# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from op_test_utils import TestDataFeeds

import onnxruntime as ort
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static


class TestAveragePoolDilations(unittest.TestCase):
    def _check(self, rank, mode, qtype, placement, quant_format=QuantFormat.QOperator, opset=19):
        shape = [1, 1, *([5] * rank)]
        dilations = [2 if mode == "dilated" or (mode == "mixed" and i == rank - 1) else 1 for i in range(rank)]
        output_shape = [1, 1, *(5 - dilation for dilation in dilations)]
        nodes = []
        initializers = []
        input_name = "input"
        has_before = placement in ("before", "both")
        has_after = placement in ("after", "both")
        if has_before or has_after:
            initializers.append(numpy_helper.from_array(np.ones([1, 1, *([1] * rank)], np.float32), "weight"))
        if has_before:
            nodes.append(helper.make_node("Conv", [input_name, "weight"], ["before"], name="before"))
            input_name = "before"
        attributes = {"kernel_shape": [2] * rank}
        if mode != "omitted":
            attributes["dilations"] = dilations
        nodes.append(helper.make_node("AveragePool", [input_name], ["pool_output"], name="pool", **attributes))
        output_name = "pool_output"
        if has_after:
            nodes.append(helper.make_node("Conv", [output_name, "weight"], ["output"], name="after"))
            output_name = "output"
        model = helper.make_model(
            helper.make_graph(
                nodes,
                "average_pool_dilations",
                [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
                [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)],
                initializers,
            ),
            opset_imports=[helper.make_opsetid("", opset)],
            ir_version=7,
        )
        data = np.linspace(-1, 1, np.prod(shape), dtype=np.float32).reshape(shape)
        feeds = [{"input": data}, {"input": np.flip(data).copy() + np.float32(0.25)}]
        onnx.checker.check_model(model, full_check=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quantized.onnx"
            quantize_static(
                model,
                path,
                TestDataFeeds(feeds),
                quant_format=quant_format,
                activation_type=qtype,
                weight_type=QuantType.QInt8,
            )
            quantized = onnx.load(path)
            onnx.checker.check_model(quantized, full_check=True)
            uses_qoperator = quant_format == QuantFormat.QOperator
            is_dilated = any(dilation != 1 for dilation in dilations)
            optimizations = [ort.GraphOptimizationLevel.ORT_DISABLE_ALL, ort.GraphOptimizationLevel.ORT_ENABLE_ALL]
            if has_before and has_after and mode != "omitted" and (is_dilated or not uses_qoperator):
                # The native QDQ AveragePool fusion also copies unsupported dilation attributes.
                # Test these valid quantized graphs independently of that existing optimizer issue.
                optimizations = [ort.GraphOptimizationLevel.ORT_DISABLE_ALL]
            for optimization in optimizations:
                options = ort.SessionOptions()
                options.graph_optimization_level = optimization
                options.intra_op_num_threads = options.inter_op_num_threads = 1
                options.log_severity_level = 3
                original = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
                actual = ort.InferenceSession(
                    quantized.SerializeToString(), options, providers=["CPUExecutionProvider"]
                )
                for feed in feeds:
                    expected = np.empty(output_shape, np.float32)
                    for position in np.ndindex(*output_shape[2:]):
                        values = [
                            feed["input"][(0, 0, *(position[i] + tap[i] * dilations[i] for i in range(rank)))]
                            for tap in itertools.product((0, 1), repeat=rank)
                        ]
                        expected[(0, 0, *position)] = np.mean(values, dtype=np.float64)
                    np.testing.assert_allclose(original.run(None, feed)[0], expected, rtol=1e-6, atol=1e-6)
                    np.testing.assert_allclose(actual.run(None, feed)[0], expected, rtol=0, atol=0.05)

            expected_pool = "AveragePool" if is_dilated or not uses_qoperator else "QLinearAveragePool"
            pool = next(n for n in quantized.graph.node if n.op_type in ("AveragePool", "QLinearAveragePool"))
            self.assertEqual(pool.op_type, expected_pool)
            if pool.op_type == "QLinearAveragePool":
                self.assertNotIn("dilations", [a.name for a in pool.attribute])
            else:
                self.assertEqual(pool.attribute, nodes[int(has_before)].attribute)
            self.assertEqual(
                sum(n.op_type == ("QLinearConv" if uses_qoperator else "Conv") for n in quantized.graph.node),
                int(has_before) + int(has_after),
            )
            if uses_qoperator and is_dilated and placement == "none":
                self.assertEqual([n.op_type for n in quantized.graph.node], ["AveragePool"])

    def test_qoperator_dilations(self):
        for rank in (1, 2, 3):
            modes = ("omitted", "unit", "dilated", "mixed") if rank > 1 else ("omitted", "unit", "dilated")
            for mode, qtype, placement in itertools.product(
                modes, (QuantType.QInt8, QuantType.QUInt8), ("none", "before", "after", "both")
            ):
                with self.subTest(rank=rank, mode=mode, qtype=qtype, placement=placement):
                    self._check(rank, mode, qtype, placement)

    def test_qdq_controls(self):
        for rank, mode, qtype in itertools.product(
            (1, 2, 3), ("omitted", "unit", "dilated"), (QuantType.QInt8, QuantType.QUInt8)
        ):
            with self.subTest(rank=rank, mode=mode, qtype=qtype):
                self._check(rank, mode, qtype, "both", QuantFormat.QDQ, opset=22)

    def test_later_opset(self):
        for mode, qtype, placement in itertools.product(
            ("omitted", "unit", "dilated"), (QuantType.QInt8, QuantType.QUInt8), ("none", "both")
        ):
            with self.subTest(mode=mode, qtype=qtype, placement=placement):
                self._check(2, mode, qtype, placement, opset=22)


if __name__ == "__main__":
    unittest.main()
