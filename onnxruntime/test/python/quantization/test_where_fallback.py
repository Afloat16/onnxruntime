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


class TestWhereFallback(unittest.TestCase):
    def _check(
        self,
        qtype,
        force,
        sides,
        *,
        selection="all",
        condition_shape=(1, 1, 2, 2),
        shared=False,
        following_conv=False,
        quant_format=QuantFormat.QOperator,
    ):
        x = np.array([-1, 0, 1, 2], np.float32).reshape(1, 1, 2, 2)
        y = np.array([2, -1], np.float32).reshape(1, 1, 1, 2)
        inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, x.shape)]
        if not shared:
            inputs.append(helper.make_tensor_value_info("y", TensorProto.FLOAT, y.shape))
        inputs.append(helper.make_tensor_value_info("condition", TensorProto.BOOL, condition_shape))
        initializers = [numpy_helper.from_array(np.ones((1, 1, 1, 1), np.float32), "weight")]
        nodes = []
        operands = ["x", "x" if shared else "y"]
        conv_count = min(sides, 1) if shared else sides
        for i in range(conv_count):
            nodes.append(helper.make_node("Conv", [operands[i], "weight"], [f"conv{i}"], name=f"before{i}"))
            operands[i] = f"conv{i}"
        if shared:
            operands[1] = operands[0]
        nodes.append(helper.make_node("Where", ["condition", *operands], ["where_out"], name="where"))
        output = "where_out"
        if following_conv:
            nodes.append(helper.make_node("Conv", [output, "weight"], ["out"], name="after"))
            output = "out"
        model = helper.make_model(
            helper.make_graph(
                nodes,
                "where_fallback",
                inputs,
                [helper.make_tensor_value_info(output, TensorProto.FLOAT, x.shape)],
                initializers,
            ),
            opset_imports=[helper.make_opsetid("", 16)],
            ir_version=7,
        )
        onnx.checker.check_model(model, full_check=True)
        conditions = [
            (np.arange(np.prod(condition_shape, dtype=np.int64)) % 2 == 0).reshape(condition_shape),
            np.ones(condition_shape, np.bool_),
            np.zeros(condition_shape, np.bool_),
        ]
        feeds = [{"x": x, **({} if shared else {"y": y}), "condition": c} for c in conditions]
        selection_options = {
            "all": {},
            "exclude": {"nodes_to_exclude": ["where"]},
            "nodes": {"nodes_to_quantize": ["before0", "before1", "after"]},
            "types": {"op_types_to_quantize": ["Conv"]},
        }[selection]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quantized.onnx"
            quantize_static(
                model,
                path,
                TestDataFeeds(feeds),
                quant_format=quant_format,
                activation_type=qtype,
                weight_type=QuantType.QInt8,
                extra_options={"ForceQuantizeNoInputCheck": force},
                **selection_options,
            )
            quantized = onnx.load(path)
            onnx.checker.check_model(quantized, full_check=True)
            if quant_format == QuantFormat.QOperator:
                quantize_where = force and selection == "all"
                self.assertEqual(sum(n.op_type == "QLinearWhere" for n in quantized.graph.node), int(quantize_where))
                self.assertEqual(sum(n.op_type == "Where" for n in quantized.graph.node), int(not quantize_where))
                self.assertEqual(
                    sum(n.op_type == "QLinearConv" for n in quantized.graph.node), conv_count + int(following_conv)
                )
                if not quantize_where:
                    for name in set(operands[:sides]):
                        if name.startswith("conv"):
                            self.assertEqual(
                                sum(n.op_type == "DequantizeLinear" and name in n.output for n in quantized.graph.node),
                                1,
                            )

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
                    expected = np.where(feed["condition"], x, x if shared else y)
                    np.testing.assert_array_equal(original.run(None, feed)[0], expected)
                    np.testing.assert_allclose(actual.run(None, feed)[0], expected, rtol=0, atol=0.04)

    def test_default_fallback_and_selection(self):
        for qtype, force, sides, selection in itertools.product(
            (QuantType.QInt8, QuantType.QUInt8), (False, True), (0, 1, 2), ("all", "exclude", "nodes", "types")
        ):
            with self.subTest(qtype=qtype, force=force, sides=sides, selection=selection):
                self._check(qtype, force, sides, selection=selection)

    def test_broadcast_shared_inputs_and_following_conv(self):
        for qtype, force, shared, shape in itertools.product(
            (QuantType.QInt8, QuantType.QUInt8), (False, True), (False, True), ((), (2,), (1, 1, 2, 2))
        ):
            with self.subTest(qtype=qtype, force=force, shared=shared, shape=shape):
                self._check(qtype, force, 2, condition_shape=shape, shared=shared, following_conv=True)

    def test_qdq_controls(self):
        for qtype, force, sides in itertools.product((QuantType.QInt8, QuantType.QUInt8), (False, True), (1, 2)):
            with self.subTest(qtype=qtype, force=force, sides=sides):
                self._check(qtype, force, sides, quant_format=QuantFormat.QDQ)


if __name__ == "__main__":
    unittest.main()
