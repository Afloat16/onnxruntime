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
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.base_quantizer import BaseQuantizer


class TestInitializerOverrideType(unittest.TestCase):
    @staticmethod
    def _fixture(qtype, dtype, explicit, shape):
        values = np.linspace(-1.5, 1.5, int(np.prod(shape)), dtype=dtype).reshape(shape)
        override = {"quant_type": qtype}
        if explicit:
            midpoint = {QuantType.QUInt8: 128, QuantType.QUInt16: 32768}.get(qtype, 0)
            override.update(
                scale=np.array(0.25, dtype=dtype),
                zero_point=np.array(midpoint, dtype=helper.tensor_dtype_to_np_dtype(qtype.tensor_type)),
            )
        graph = helper.make_graph(
            [helper.make_node("Transpose", ["weight"], ["y"])],
            "initializer_override",
            [],
            [helper.make_tensor_value_info("y", helper.np_dtype_to_tensor_dtype(values.dtype), list(reversed(shape)))],
            [numpy_helper.from_array(values, "weight")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
        return model, values, {"TensorQuantOverrides": {"weight": [override]}}

    def test_initializer_storage_and_dequantization(self):
        qtypes = (
            QuantType.QInt8,
            QuantType.QUInt8,
            QuantType.QInt16,
            QuantType.QUInt16,
            QuantType.QFLOAT8E4M3FN,
        )
        for global_qtype, qtype, dtype, explicit, shape in itertools.product(
            (QuantType.QUInt8, QuantType.QFLOAT8E4M3FN),
            qtypes,
            (np.float32, np.float16),
            (False, True),
            ((5,), (2, 3)),
        ):
            with self.subTest(global_qtype=global_qtype, qtype=qtype, dtype=dtype, explicit=explicit, shape=shape):
                model, values, options = self._fixture(qtype, dtype, explicit, shape)
                quantizer = BaseQuantizer(
                    model, False, False, global_qtype, global_qtype, None, [], [], ["Transpose"], options
                )
                weight = quantizer.model.get_initializer("weight")
                qname, zp_name, scale_name = quantizer.quantize_initializer_impl(weight, global_qtype.tensor_type)
                tensors = {name: quantizer.model.get_initializer(name) for name in (qname, zp_name, scale_name)}
                self.assertEqual(tensors[qname].data_type, qtype.tensor_type)
                self.assertEqual(tensors[zp_name].data_type, qtype.tensor_type)
                self.assertEqual(list(tensors[qname].dims), list(shape))
                bits = 16 if qtype in (QuantType.QInt16, QuantType.QUInt16) else 8
                self.assertEqual(len(tensors[qname].raw_data), (values.size * bits + 7) // 8)
                decoded = numpy_helper.to_array(tensors[qname]).astype(np.float32)
                scale = numpy_helper.to_array(tensors[scale_name]).astype(np.float32)
                zero = numpy_helper.to_array(tensors[zp_name]).astype(np.float32)
                if qtype != QuantType.QFLOAT8E4M3FN:
                    signed = qtype in (QuantType.QInt8, QuantType.QInt16)
                    minimum = -(2 ** (bits - 1)) if signed else 0
                    maximum = 2 ** (bits - int(signed)) - 1
                    expected_codes = np.clip(np.rint(values.astype(np.float32) / scale) + zero, minimum, maximum)
                    np.testing.assert_array_equal(decoded, expected_codes)
                np.testing.assert_array_equal(numpy_helper.to_array(weight), values)
                dequantized_model = helper.make_model(
                    helper.make_graph(
                        [helper.make_node("DequantizeLinear", [qname, scale_name, zp_name], ["y"])],
                        "stored_initializer",
                        [],
                        [helper.make_tensor_value_info("y", helper.np_dtype_to_tensor_dtype(values.dtype), shape)],
                        list(tensors.values()),
                    ),
                    opset_imports=[helper.make_opsetid("", 21)],
                    ir_version=10,
                )
                onnx.checker.check_model(dequantized_model, full_check=True)
                reloaded = onnx.load_from_string(dequantized_model.SerializeToString())
                expected = ((decoded - zero) * scale).astype(dtype)
                for optimization in (
                    ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                ):
                    session_options = ort.SessionOptions()
                    session_options.graph_optimization_level = optimization
                    session_options.intra_op_num_threads = 1
                    session = ort.InferenceSession(
                        reloaded.SerializeToString(), session_options, providers=["CPUExecutionProvider"]
                    )
                    actual = session.run(None, {})[0]
                    self.assertEqual(actual.dtype, values.dtype)
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)

    def test_static_quantization_honors_integer_override(self):
        for qtype, explicit, shape in itertools.product(
            (QuantType.QInt8, QuantType.QUInt8),
            (False, True),
            ((5,), (2, 3)),
        ):
            with self.subTest(qtype=qtype, explicit=explicit, shape=shape):
                model, values, options = self._fixture(qtype, np.float32, explicit, shape)
                model.graph.input.append(helper.make_tensor_value_info("control", TensorProto.INT64, []))
                model.graph.node.insert(0, helper.make_node("Identity", ["control"], ["control_output"]))
                model.graph.output.append(helper.make_tensor_value_info("control_output", TensorProto.INT64, []))
                options["ForceQuantizeNoInputCheck"] = True
                feeds = [{"control": np.array(value, dtype=np.int64)} for value in (0, 1)]
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "quantized.onnx"
                    quantize_static(
                        model,
                        path,
                        TestDataFeeds(feeds),
                        quant_format=QuantFormat.QOperator,
                        activation_type=QuantType.QFLOAT8E4M3FN,
                        weight_type=QuantType.QFLOAT8E4M3FN,
                        calibrate_method=CalibrationMethod.Distribution,
                        extra_options=options,
                    )
                    quantized = onnx.load(path)
                onnx.checker.check_model(quantized, full_check=True)
                tensors = {tensor.name: tensor for tensor in quantized.graph.initializer}
                self.assertEqual(tensors["weight_quantized"].data_type, qtype.tensor_type)
                tolerance = float(numpy_helper.to_array(tensors["weight_scale"])) / 2 + 2 * np.finfo(np.float32).eps
                for optimization in (
                    ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                ):
                    session_options = ort.SessionOptions()
                    session_options.graph_optimization_level = optimization
                    session_options.intra_op_num_threads = 1
                    original = ort.InferenceSession(
                        model.SerializeToString(), session_options, providers=["CPUExecutionProvider"]
                    )
                    actual = ort.InferenceSession(
                        quantized.SerializeToString(), session_options, providers=["CPUExecutionProvider"]
                    )
                    for feed in feeds:
                        original_outputs, actual_outputs = original.run(None, feed), actual.run(None, feed)
                        np.testing.assert_array_equal(original_outputs[0], values.T)
                        np.testing.assert_allclose(actual_outputs[0], values.T, rtol=0, atol=tolerance)
                        np.testing.assert_array_equal(actual_outputs[1], feed["control"])


if __name__ == "__main__":
    unittest.main()
