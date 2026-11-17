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
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static


class TestSharedActivationInput(unittest.TestCase):
    def _check_activation(self, quant_format, op_type, topology, symmetric, opset=13):
        shape = [1, 1, 2, 2]
        nodes = [helper.make_node("Conv", ["x", "w"], ["pre_activation"], name="conv")]
        initializers = [numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), "w")]
        outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)]
        activation_inputs = ["pre_activation"]
        attrs = {}
        if op_type == "Clip":
            if opset < 11:
                attrs = {"min": -0.5, "max": 0.5}
            else:
                for name, value in (("min", -0.5), ("max", 0.5)):
                    initializers.append(numpy_helper.from_array(np.array(value, dtype=np.float32), name))
                    activation_inputs.append(name)
        activation = helper.make_node(op_type, activation_inputs, ["y"], name="activation", **attrs)
        if topology == "graph_output":
            outputs.append(helper.make_tensor_value_info("pre_activation", TensorProto.FLOAT, shape))
        elif topology != "single":
            branch_type = "Transpose" if topology.startswith("transpose") else "Identity"
            branch_attrs = {"perm": [0, 1, 3, 2]} if branch_type == "Transpose" else {}
            branch = helper.make_node(branch_type, ["pre_activation"], ["side"], name="branch", **branch_attrs)
            outputs.append(helper.make_tensor_value_info("side", TensorProto.FLOAT, shape))
            if topology.endswith("before"):
                nodes.append(branch)
        nodes.append(activation)
        if topology.endswith("after"):
            nodes.append(branch)
        model = helper.make_model(
            helper.make_graph(
                nodes,
                "shared_activation_input",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
                outputs,
                initializers,
            ),
            opset_imports=[helper.make_opsetid("", opset)],
            ir_version=7,
        )
        onnx.checker.check_model(model, full_check=True)
        data = np.array([-2, -1, 1, 2], dtype=np.float32).reshape(shape)
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "quantized.onnx"
            quantize_static(
                model,
                output_path,
                TestDataFeeds([{"x": data}]),
                quant_format=quant_format,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QUInt8,
                extra_options={"ActivationSymmetric": symmetric},
            )
            quantized = onnx.load(output_path)
        onnx.checker.check_model(quantized, full_check=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        original_session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
        for optimization in (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
            options.graph_optimization_level = optimization
            quantized_session = ort.InferenceSession(
                quantized.SerializeToString(), options, providers=["CPUExecutionProvider"]
            )
            for values in (data, np.flip(data, axis=-1).copy(), data * 0.5):
                with self.subTest(optimization=optimization, values=values.flatten().tolist()):
                    expected = [np.maximum(values, 0) if op_type == "Relu" else np.clip(values, -0.5, 0.5)]
                    if topology != "single":
                        expected.append(values.transpose(0, 1, 3, 2) if topology.startswith("transpose") else values)
                    original = original_session.run(None, {"x": values})
                    actual = quantized_session.run(None, {"x": values})
                    for original_output, actual_output, expected_output in zip(original, actual, expected, strict=True):
                        np.testing.assert_array_equal(original_output, expected_output)
                        np.testing.assert_allclose(actual_output, expected_output, rtol=0, atol=0.025)
        if topology == "single" and not symmetric:
            self.assertFalse(any(n.op_type == op_type for n in quantized.graph.node))
        else:
            self.assertTrue(any(n.op_type == op_type for n in quantized.graph.node))
        if quant_format == QuantFormat.QOperator:
            self.assertTrue(any(n.op_type == "QLinearConv" for n in quantized.graph.node))
        else:
            self.assertTrue(any(n.op_type == "QuantizeLinear" for n in quantized.graph.node))

    def test_shared_activation_input(self):
        for quant_format, op_type, topology, symmetric in itertools.product(
            (QuantFormat.QOperator, QuantFormat.QDQ),
            ("Relu", "Clip"),
            ("graph_output", "identity_before", "identity_after", "transpose_before", "transpose_after", "single"),
            (False, True),
        ):
            with self.subTest(quant_format=quant_format, op_type=op_type, topology=topology, symmetric=symmetric):
                self._check_activation(quant_format, op_type, topology, symmetric)

    def test_clip_graph_output_across_opsets(self):
        for quant_format, opset, symmetric in itertools.product(
            (QuantFormat.QOperator, QuantFormat.QDQ), (10, 21), (False, True)
        ):
            with self.subTest(quant_format=quant_format, opset=opset, symmetric=symmetric):
                self._check_activation(quant_format, "Clip", "graph_output", symmetric, opset)


if __name__ == "__main__":
    unittest.main()
