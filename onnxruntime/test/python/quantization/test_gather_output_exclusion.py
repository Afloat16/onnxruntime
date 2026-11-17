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


class TestGatherOutputExclusion(unittest.TestCase):
    def _check(self, op_type, qtype, source, excluded, axis=0, following_matmul=False):
        data = np.arange(-3, 6, dtype=np.float32).reshape(3, 3)
        indices = np.array([2, 0, -1] if op_type == "Gather" else [[2, 0, -1], [0, -2, 1], [1, 2, 0]], np.int64)
        inputs = [helper.make_tensor_value_info("indices", TensorProto.INT64, indices.shape)]
        initializers = []
        nodes = []
        data_name = "data"
        if source == "initializer":
            initializers.append(numpy_helper.from_array(data, data_name))
        else:
            inputs.append(helper.make_tensor_value_info(data_name, TensorProto.FLOAT, data.shape))
        if source == "producer" or following_matmul:
            initializers.append(numpy_helper.from_array(np.eye(3, dtype=np.float32), "weight"))
        if source == "producer":
            nodes.append(helper.make_node("MatMul", [data_name, "weight"], ["before"], name="before"))
            data_name = "before"
        nodes.append(helper.make_node(op_type, [data_name, "indices"], ["gather_out"], name="gather", axis=axis))
        output = "gather_out"
        if following_matmul:
            nodes.append(helper.make_node("MatMul", [output, "weight"], ["out"], name="after"))
            output = "out"
        model = helper.make_model(
            helper.make_graph(
                nodes,
                "gather_output_exclusion",
                inputs,
                [helper.make_tensor_value_info(output, TensorProto.FLOAT, [3, 3])],
                initializers,
            ),
            opset_imports=[helper.make_opsetid("", 21)],
            ir_version=7,
        )
        feeds = [
            {"indices": current, **({} if source == "initializer" else {"data": data})}
            for current in (indices, np.flip(indices).copy())
        ]
        onnx.checker.check_model(model, full_check=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quantized.onnx"
            quantize_static(
                model,
                path,
                TestDataFeeds(feeds),
                quant_format=QuantFormat.QDQ,
                activation_type=qtype,
                weight_type=QuantType.QInt8,
                extra_options={
                    "ForceQuantizeNoInputCheck": source == "forced",
                    "OpTypesToExcludeOutputQuantization": [op_type] if excluded else [],
                },
            )
            quantized = onnx.load(path)
            onnx.checker.check_model(quantized, full_check=True)
            gather = next(n for n in quantized.graph.node if n.op_type == op_type)
            producers = {name: n for n in quantized.graph.node for name in n.output}
            consumers = [n for n in quantized.graph.node if gather.output[0] in n.input]
            active = source != "unforced"
            input_producer = producers.get(gather.input[0])
            self.assertEqual(input_producer is not None and input_producer.op_type == "DequantizeLinear", active)
            self.assertEqual(
                sum(n.op_type == "QuantizeLinear" for n in consumers),
                int(following_matmul or (active and not excluded)),
            )
            if excluded and not following_matmul:
                self.assertEqual(gather.output[0], "gather_out")
            if following_matmul:
                matmul = next(n for n in quantized.graph.node if n.name == "after")
                self.assertEqual(producers[matmul.input[0]].op_type, "DequantizeLinear")

            for optimization in (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
                options = ort.SessionOptions()
                options.graph_optimization_level = optimization
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                options.log_severity_level = 3
                original = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
                actual = ort.InferenceSession(
                    quantized.SerializeToString(), options, providers=["CPUExecutionProvider"]
                )
                for feed in feeds:
                    expected = (
                        np.take(data, feed["indices"], axis=axis)
                        if op_type == "Gather"
                        else np.take_along_axis(data, feed["indices"], axis=axis)
                    )
                    np.testing.assert_array_equal(original.run(None, feed)[0], expected)
                    np.testing.assert_allclose(actual.run(None, feed)[0], expected, rtol=0, atol=0.08)

    def test_output_exclusion(self):
        for op_type, qtype, source, excluded, axis in itertools.product(
            ("Gather", "GatherElements"),
            (QuantType.QInt8, QuantType.QUInt8),
            ("initializer", "forced", "producer"),
            (False, True),
            (0, -1),
        ):
            with self.subTest(op_type=op_type, qtype=qtype, source=source, excluded=excluded, axis=axis):
                self._check(op_type, qtype, source, excluded, axis)

    def test_unforced_input_controls(self):
        for op_type, qtype, excluded in itertools.product(
            ("Gather", "GatherElements"), (QuantType.QInt8, QuantType.QUInt8), (False, True)
        ):
            with self.subTest(op_type=op_type, qtype=qtype, excluded=excluded):
                self._check(op_type, qtype, "unforced", excluded)

    def test_quantized_downstream_consumer(self):
        for op_type, qtype, excluded in itertools.product(
            ("Gather", "GatherElements"), (QuantType.QInt8, QuantType.QUInt8), (False, True)
        ):
            with self.subTest(op_type=op_type, qtype=qtype, excluded=excluded):
                self._check(op_type, qtype, "forced", excluded, following_matmul=True)


if __name__ == "__main__":
    unittest.main()
