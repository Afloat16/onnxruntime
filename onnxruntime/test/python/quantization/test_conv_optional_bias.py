# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_static


class _DataReader(CalibrationDataReader):
    def __init__(self, feeds):
        self.feeds = iter(feeds)

    def get_next(self):
        return next(self.feeds, None)


class TestConvOptionalBias(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.model_dir = Path(self.temp_dir.name)
        data = np.linspace(-1, 1, 32, dtype=np.float32).reshape(1, 2, 4, 4)
        self.feeds = [{"input": data}, {"input": -data}]
        self.session_options = SessionOptions()
        self.session_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
        self.session_options.intra_op_num_threads = 1

    @staticmethod
    def _make_model(op_type, bias_name):
        weight_shape = (3, 2, 2, 2) if op_type == "Conv" else (2, 3, 2, 2)
        output_shape = [1, 3, 3, 3] if op_type == "Conv" else [1, 3, 5, 5]
        weights = np.linspace(-0.5, 0.5, 24, dtype=np.float32).reshape(weight_shape)
        initializers = [numpy_helper.from_array(weights, "weight")]
        if bias_name:
            initializers.append(numpy_helper.from_array(np.array([0.25, -0.5, 0.75], dtype=np.float32), bias_name))
        inputs = ["input", "weight"] + ([] if bias_name is None else [bias_name])
        graph = helper.make_graph(
            [helper.make_node(op_type, inputs, ["output"], name="conv")],
            "conv_optional_bias",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 4, 4])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
            initializers,
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=7)

    def _check_quantization(self, op_type, quant_format=None, per_channel=False, weight_type=QuantType.QUInt8):
        outputs = {}
        for bias_name in [None, "", "bias"]:
            with self.subTest(bias_name=bias_name):
                model = self._make_model(op_type, bias_name)
                onnx.checker.check_model(model, full_check=True)
                input_path = self.model_dir / "input.onnx"
                output_path = self.model_dir / "quantized.onnx"
                onnx.save(model, input_path)
                if quant_format is None:
                    quantize_dynamic(input_path, output_path, weight_type=weight_type)
                else:
                    with patch("logging.warning", wraps=logging.warning) as log_warning:
                        quantize_static(
                            input_path,
                            output_path,
                            _DataReader(self.feeds),
                            quant_format=quant_format,
                            activation_type=QuantType.QUInt8,
                            weight_type=weight_type,
                            per_channel=per_channel,
                        )
                    self.assertNotIn(
                        call("Expected bias '' to be an initializer"),
                        log_warning.call_args_list,
                    )
                quantized = onnx.load(output_path)
                onnx.checker.check_model(quantized, full_check=True)
                expected_op = (
                    "ConvInteger"
                    if quant_format is None
                    else ("QLinearConv" if quant_format == QuantFormat.QOperator else op_type)
                )
                self.assertEqual(sum(node.op_type == expected_op for node in quantized.graph.node), 1)
                quantized_session = InferenceSession(
                    quantized.SerializeToString(), self.session_options, providers=["CPUExecutionProvider"]
                )
                original_session = InferenceSession(
                    model.SerializeToString(), self.session_options, providers=["CPUExecutionProvider"]
                )
                outputs[bias_name] = [quantized_session.run(None, feed)[0] for feed in self.feeds]
                for feed, actual in zip(self.feeds, outputs[bias_name], strict=True):
                    expected = original_session.run(None, feed)[0]
                    np.testing.assert_allclose(actual, expected, rtol=0.01, atol=0.03)
        if None in outputs and "" in outputs:
            for omitted, empty in zip(outputs[None], outputs[""], strict=True):
                np.testing.assert_array_equal(omitted, empty)

    def test_dynamic_conv_optional_bias(self):
        self._check_quantization("Conv")

    def test_static_conv_optional_bias(self):
        for op_type, quant_format in [
            ("Conv", QuantFormat.QOperator),
            ("Conv", QuantFormat.QDQ),
            ("ConvTranspose", QuantFormat.QDQ),
        ]:
            for per_channel in [False, True]:
                for weight_type in [QuantType.QUInt8, QuantType.QInt8]:
                    with self.subTest(
                        op_type=op_type, quant_format=quant_format, per_channel=per_channel, weight_type=weight_type
                    ):
                        self._check_quantization(op_type, quant_format, per_channel, weight_type)


if __name__ == "__main__":
    unittest.main()
