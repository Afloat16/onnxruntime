#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from op_test_utils import TestDataFeeds

import onnxruntime as ort
from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic, quantize_static


class TestQuantizationOperatorDomains(unittest.TestCase):
    @staticmethod
    def _expected(op_type, custom, value, weight):
        if custom:
            if op_type == "Add":
                return value - weight
            if op_type in ("Mul", "MatMul"):
                return value + weight
            if op_type == "Relu":
                return -value
            if op_type == "Transpose":
                return value * 2
            return value
        if op_type == "Add":
            return value + weight
        if op_type == "Mul":
            return value * weight
        if op_type == "MatMul":
            return value @ weight
        if op_type == "Relu":
            return np.maximum(value, 0)
        if op_type == "Transpose":
            return value.T
        exponentials = np.exp(value - value.max(axis=-1, keepdims=True))
        return exponentials / exponentials.sum(axis=-1, keepdims=True)

    def _check(self, op_type, custom, quant_format, surrounding, qtype, dynamic=False):
        values = np.array([[0.25, 0.75], [1.25, 2]], dtype=np.float32)
        weight = np.array([[1, 0.5], [0.25, 2]], dtype=np.float32)
        initializers = [numpy_helper.from_array(np.eye(2, dtype=np.float32), "identity_weight")]
        nodes = []
        data_name = "x"
        if surrounding in ("before", "both"):
            nodes.append(helper.make_node("MatMul", [data_name, "identity_weight"], ["before"], name="before"))
            data_name = "before"
        node_inputs = [data_name]
        if op_type in ("Add", "Mul", "MatMul"):
            initializers.append(numpy_helper.from_array(weight, "weight"))
            node_inputs.append("weight")
        nodes.append(
            helper.make_node(op_type, node_inputs, ["target_output"], domain="example" if custom else "", name="target")
        )
        output_name = "target_output"
        if surrounding in ("after", "both"):
            nodes.append(helper.make_node("MatMul", [output_name, "identity_weight"], ["y"], name="after"))
            output_name = "y"
        graph = helper.make_graph(
            nodes,
            "operator_domains",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, values.shape)],
            [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, values.shape)],
            initializers,
        )
        functions = []
        imports = [helper.make_opsetid("", 13)]
        if custom:
            imports.append(helper.make_opsetid("example", 1))
            function_inputs = ["a", "b"] if len(node_inputs) == 2 else ["a"]
            function_op = {"Add": "Sub", "Mul": "Add", "MatMul": "Add", "Relu": "Neg", "Softmax": "Identity"}
            if op_type == "Transpose":
                function_nodes = [
                    helper.make_node(
                        "Constant", [], ["two"], value=numpy_helper.from_array(np.array(2, dtype=np.float32))
                    ),
                    helper.make_node("Mul", ["a", "two"], ["out"]),
                ]
            else:
                function_nodes = [helper.make_node(function_op[op_type], function_inputs, ["out"])]
            functions.append(
                helper.make_function(
                    "example", op_type, function_inputs, ["out"], function_nodes, [helper.make_opsetid("", 13)]
                )
            )
        model = helper.make_model(graph, opset_imports=imports, functions=functions, ir_version=7)
        feeds = [{"x": values}, {"x": np.flip(values, axis=0).copy()}]
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "quantized.onnx"
            if dynamic:
                quantize_dynamic(model, output_path, weight_type=qtype)
            else:
                quantize_static(
                    model,
                    output_path,
                    TestDataFeeds(feeds),
                    quant_format=quant_format,
                    activation_type=qtype,
                    weight_type=qtype,
                )
            quantized = onnx.load(output_path)
        onnx.checker.check_model(quantized, full_check=True)
        for optimization in (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
            options = ort.SessionOptions()
            options.graph_optimization_level = optimization
            options.intra_op_num_threads = 1
            original = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
            actual = ort.InferenceSession(quantized.SerializeToString(), options, providers=["CPUExecutionProvider"])
            for feed in feeds:
                expected = self._expected(op_type, custom, feed["x"], weight)
                np.testing.assert_allclose(original.run(None, feed)[0], expected, atol=1e-6, rtol=1e-6)
                np.testing.assert_allclose(actual.run(None, feed)[0], expected, atol=0.06, rtol=0.02)
        if custom:
            target = next(node for node in quantized.graph.node if node.name == "target")
            self.assertEqual((target.domain, target.op_type), ("example", op_type))
            self.assertEqual(quantized.functions[0].SerializeToString(), model.functions[0].SerializeToString())
        if dynamic or quant_format == QuantFormat.QOperator:
            expected_count = int(surrounding in ("before", "both")) + int(surrounding in ("after", "both"))
            expected_count += int(not custom and op_type == "MatMul")
            quantized_op = "MatMulInteger" if dynamic else "QLinearMatMul"
            self.assertEqual(sum(node.op_type == quantized_op for node in quantized.graph.node), expected_count)
        elif custom:
            self.assertTrue(any(node.op_type == "QuantizeLinear" for node in quantized.graph.node))

    def test_static_operator_domains(self):
        for args in itertools.product(
            ("Add", "Mul", "Relu", "Transpose", "Softmax", "MatMul"),
            (False, True),
            (QuantFormat.QOperator, QuantFormat.QDQ),
            ("none", "before", "after", "both"),
            (QuantType.QInt8, QuantType.QUInt8),
        ):
            with self.subTest(args=args):
                self._check(*args)

    def test_dynamic_operator_domains(self):
        for custom, surrounding, qtype in itertools.product(
            (False, True), ("none", "before", "after", "both"), (QuantType.QInt8, QuantType.QUInt8)
        ):
            with self.subTest(custom=custom, surrounding=surrounding, qtype=qtype):
                self._check("MatMul", custom, QuantFormat.QOperator, surrounding, qtype, dynamic=True)


if __name__ == "__main__":
    unittest.main()
