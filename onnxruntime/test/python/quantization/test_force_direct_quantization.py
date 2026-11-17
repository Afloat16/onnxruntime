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
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_dynamic, quantize_static


class TestForceDirectQuantization(unittest.TestCase):
    @staticmethod
    def _transform(op_type, value):
        if op_type == "Transpose":
            return value.transpose(0, 2, 3, 1)
        if op_type in ("Reshape", "Flatten"):
            return value.reshape(2, 12)
        if op_type == "Squeeze":
            return np.squeeze(value, axis=0)
        if op_type == "Unsqueeze":
            return np.expand_dims(value, axis=-1)
        if op_type == "MaxPool":
            return np.lib.stride_tricks.sliding_window_view(value, (2, 2), axis=(2, 3)).max(axis=(-2, -1))
        return value.repeat(2, axis=2).repeat(2, axis=3)

    def _model(self, op_type, source, dynamic=False, integer=False):
        values = (np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4) - 12) / 8
        if integer:
            values = values.astype(np.int64)
        proto_type = TensorProto.INT64 if integer else TensorProto.FLOAT
        inputs = [helper.make_tensor_value_info("control", TensorProto.INT64, [])]
        nodes = [helper.make_node("Identity", ["control"], ["control_output"])]
        initializers = []
        data_name = "x"
        if source == "initializer":
            initializers.append(numpy_helper.from_array(values, data_name))
        else:
            inputs.append(helper.make_tensor_value_info(data_name, proto_type, values.shape))
        if source == "producer":
            weights = np.eye(4, dtype=np.float32) if dynamic else np.eye(2, dtype=np.float32).reshape(2, 2, 1, 1)
            initializers.append(numpy_helper.from_array(weights, "weights"))
            nodes.append(helper.make_node("MatMul" if dynamic else "Conv", [data_name, "weights"], ["before"]))
            data_name = "before"

        node_inputs = [data_name]
        attrs = {}
        if op_type == "Transpose":
            attrs["perm"] = [0, 2, 3, 1]
        elif op_type in ("Reshape", "Squeeze", "Unsqueeze"):
            argument = {"Reshape": [2, 12], "Squeeze": [0], "Unsqueeze": [-1]}[op_type]
            initializers.append(numpy_helper.from_array(np.array(argument, dtype=np.int64), "argument"))
            node_inputs.append("argument")
        elif op_type == "Flatten":
            attrs["axis"] = 2
        elif op_type == "MaxPool":
            attrs["kernel_shape"] = [2, 2]
        else:
            initializers.append(numpy_helper.from_array(np.array([1, 1, 2, 2], dtype=np.float32), "scales"))
            node_inputs += ["", "scales"]
            attrs["mode"] = "nearest"
        nodes.append(helper.make_node(op_type, node_inputs, ["y"], name="target", **attrs))
        outputs = [
            helper.make_tensor_value_info("y", proto_type, self._transform(op_type, values).shape),
            helper.make_tensor_value_info("control_output", TensorProto.INT64, []),
        ]
        graph = helper.make_graph(nodes, "forced_direct", inputs, outputs, initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=7)
        feeds = []
        for index, data in enumerate((values, np.flip(values, axis=-1).copy())):
            feed = {"control": np.array(index, dtype=np.int64)}
            if source != "initializer":
                feed["x"] = data
            feeds.append(feed)
        return model, feeds, values

    def _check(self, op_type, source, force, exclude, qtype, dynamic=False, integer=False):
        model, feeds, constant = self._model(op_type, source, dynamic, integer)
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "quantized.onnx"
            options = {
                "nodes_to_exclude": ["target"] if exclude else [],
                "extra_options": {"ForceQuantizeNoInputCheck": force},
                "weight_type": qtype,
            }
            if dynamic:
                quantize_dynamic(model, output_path, **options)
            else:
                quantize_static(
                    model,
                    output_path,
                    TestDataFeeds(feeds),
                    quant_format=QuantFormat.QOperator,
                    activation_type=qtype,
                    **options,
                )
            quantized = onnx.load(output_path)
            onnx.checker.check_model(quantized, full_check=True)
            target = next(node for node in quantized.graph.node if node.name == "target")
            expected_quantized = not integer and not exclude and (force or (source == "producer" and not dynamic))
            self.assertEqual(target.output[0] != "y", expected_quantized)
            inferred = onnx.shape_inference.infer_shapes(quantized)
            types = {
                value.name: value.type.tensor_type.elem_type
                for value in itertools.chain(inferred.graph.input, inferred.graph.value_info, inferred.graph.output)
            }
            types.update({value.name: value.data_type for value in inferred.graph.initializer})
            expected_type = (
                (TensorProto.UINT8 if dynamic or qtype == QuantType.QUInt8 else TensorProto.INT8)
                if expected_quantized
                else (TensorProto.INT64 if integer else TensorProto.FLOAT)
            )
            self.assertEqual(types[target.input[0]], expected_type)
            self.assertEqual(types[target.output[0]], expected_type)
            if expected_quantized:
                dequantize = next(node for node in quantized.graph.node if "y" in node.output)
                self.assertEqual(dequantize.op_type, "DequantizeLinear")
                self.assertEqual(dequantize.input[0], target.output[0])
            if source == "producer":
                self.assertIn(
                    "MatMulInteger" if dynamic else "QLinearConv", [node.op_type for node in quantized.graph.node]
                )
            for optimization in (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = optimization
                session_options.intra_op_num_threads = 1
                original_session = ort.InferenceSession(
                    model.SerializeToString(), session_options, providers=["CPUExecutionProvider"]
                )
                quantized_session = ort.InferenceSession(
                    quantized.SerializeToString(), session_options, providers=["CPUExecutionProvider"]
                )
                for feed in feeds:
                    expected = self._transform(op_type, constant if source == "initializer" else feed["x"])
                    original = original_session.run(None, feed)
                    actual = quantized_session.run(None, feed)
                    np.testing.assert_allclose(original[0], expected, rtol=0, atol=1e-6)
                    np.testing.assert_allclose(actual[0], expected, rtol=0, atol=0.03)
                    np.testing.assert_array_equal(actual[1], feed["control"])

    def test_static_forced_direct_operators(self):
        operators = ("Transpose", "Reshape", "Squeeze", "Unsqueeze", "Flatten", "MaxPool", "Resize")
        for args in itertools.product(
            operators,
            ("input", "initializer", "producer"),
            (False, True),
            (False, True),
            (QuantType.QInt8, QuantType.QUInt8),
        ):
            with self.subTest(args=args):
                self._check(*args)

    def test_dynamic_forced_transpose(self):
        for args in itertools.product(
            ("input", "initializer", "producer"), (False, True), (False, True), (QuantType.QInt8, QuantType.QUInt8)
        ):
            with self.subTest(args=args):
                self._check("Transpose", *args, dynamic=True)

    def test_integer_inputs_remain_integer(self):
        for dynamic, source in itertools.product((False, True), ("input", "initializer")):
            with self.subTest(dynamic=dynamic, source=source):
                self._check("Transpose", source, True, False, QuantType.QUInt8, dynamic=dynamic, integer=True)

    def test_unsupported_float8_operators_remain_float(self):
        for op_type, source, force in itertools.product(("Resize", "MaxPool"), ("input", "initializer"), (False, True)):
            with self.subTest(op_type=op_type, source=source, force=force):
                model, feeds, _ = self._model(op_type, source)
                with tempfile.TemporaryDirectory() as directory:
                    output_path = Path(directory) / "quantized.onnx"
                    quantize_static(
                        model,
                        output_path,
                        TestDataFeeds(feeds),
                        quant_format=QuantFormat.QOperator,
                        activation_type=QuantType.QFLOAT8E4M3FN,
                        weight_type=QuantType.QFLOAT8E4M3FN,
                        calibrate_method=CalibrationMethod.Distribution,
                        extra_options={"ForceQuantizeNoInputCheck": force},
                    )
                    quantized = onnx.load(output_path)
                onnx.checker.check_model(quantized, full_check=True)
                self.assertEqual([node.op_type for node in quantized.graph.node], ["Identity", op_type])
                for optimization in (
                    ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                ):
                    options = ort.SessionOptions()
                    options.graph_optimization_level = optimization
                    options.intra_op_num_threads = 1
                    original = ort.InferenceSession(
                        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
                    )
                    actual = ort.InferenceSession(
                        quantized.SerializeToString(), options, providers=["CPUExecutionProvider"]
                    )
                    for feed in feeds:
                        for expected_output, actual_output in zip(
                            original.run(None, feed), actual.run(None, feed), strict=True
                        ):
                            np.testing.assert_array_equal(actual_output, expected_output)


if __name__ == "__main__":
    unittest.main()
