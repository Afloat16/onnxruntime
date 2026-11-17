# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from onnxruntime.quantization import QuantType, quantize_dynamic


class TestAttentionQuantizationFallback(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.model_dir = Path(self.temp_dir.name)
        self.data = np.arange(12, dtype=np.float32).reshape(1, 3, 4) / 4
        self.session_options = SessionOptions()
        self.session_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
        self.session_options.intra_op_num_threads = 1

    @staticmethod
    def _make_contrib_model(kind, with_matmul=False):
        initializers = [numpy_helper.from_array(np.tile(np.eye(4, dtype=np.float32), (1, 3)), "weight")]
        inputs = ["projected" if with_matmul else "input", "weight"]
        if kind != "omitted_bias":
            inputs.append("" if kind == "empty_bias" else "bias")
            if kind != "empty_bias":
                initializers.append(numpy_helper.from_array(np.zeros(12, dtype=np.float32), "bias"))
        attributes = {"num_heads": 2}
        if kind == "attention_bias":
            attention_bias = np.zeros((1, 2, 3, 3), dtype=np.float32)
            attention_bias[:, :, :, 0] = 20
            initializers.append(numpy_helper.from_array(attention_bias, "attention_bias"))
            inputs.extend(["", "", "attention_bias"])
        elif kind == "past_sequence_length":
            initializers.append(numpy_helper.from_array(np.array(0, dtype=np.int32), "past_sequence_length"))
            inputs.extend(["", "", "", "past_sequence_length"])
        elif kind == "rotary_embedding_dim":
            attributes["rotary_embedding_dim"] = 0
        elif kind == "empty_optional_inputs":
            inputs.extend(["", "", "", ""])
        elif kind == "mask":
            initializers.append(numpy_helper.from_array(np.array([2], dtype=np.int32), "mask"))
            inputs.append("mask")
        nodes = []
        if with_matmul:
            initializers.append(numpy_helper.from_array(np.eye(4, dtype=np.float32), "projection_weight"))
            nodes.append(helper.make_node("MatMul", ["input", "projection_weight"], ["projected"], name="projection"))
        nodes.append(
            helper.make_node("Attention", inputs, ["output"], domain="com.microsoft", name="attention", **attributes)
        )
        graph = helper.make_graph(
            nodes,
            "attention_fallback",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4])],
            initializers,
        )
        return helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("com.microsoft", 1)], ir_version=7
        )

    def _quantize(self, model, data, weight_type, per_channel):
        onnx.checker.check_model(model, full_check=True)
        input_path = self.model_dir / "input.onnx"
        output_path = self.model_dir / "quantized.onnx"
        onnx.save(model, input_path)
        feeds = [{"input": data}, {"input": -data}]
        quantize_dynamic(input_path, output_path, weight_type=weight_type, per_channel=per_channel)
        quantized = onnx.load(output_path)
        onnx.checker.check_model(quantized, full_check=True)
        original_session = InferenceSession(
            model.SerializeToString(), self.session_options, providers=["CPUExecutionProvider"]
        )
        quantized_session = InferenceSession(
            quantized.SerializeToString(), self.session_options, providers=["CPUExecutionProvider"]
        )
        for feed in feeds:
            expected = original_session.run(None, feed)[0]
            actual = quantized_session.run(None, feed)[0]
            np.testing.assert_allclose(actual, expected, rtol=0.01, atol=0.03)
        return quantized

    def test_contrib_attention_fallback(self):
        for kind in ["omitted_bias", "empty_bias", "attention_bias", "past_sequence_length", "rotary_embedding_dim"]:
            for weight_type in [QuantType.QInt8, QuantType.QUInt8]:
                for per_channel in [False, True]:
                    with self.subTest(kind=kind, weight_type=weight_type, per_channel=per_channel):
                        model = self._make_contrib_model(kind)
                        quantized = self._quantize(model, self.data, weight_type, per_channel)
                        self.assertEqual(list(quantized.graph.node), list(model.graph.node))

    @unittest.skipIf(onnx.defs.onnx_opset_version() < 23, "Standard-domain Attention requires opset 23")
    def test_standard_attention_fallback(self):
        data = self.data.reshape(1, 1, 3, 4)
        graph = helper.make_graph(
            [helper.make_node("Attention", ["input", "key", "value"], ["output"], name="attention")],
            "standard_attention",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 3, 4])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 3, 4])],
            [numpy_helper.from_array(data, "key"), numpy_helper.from_array(data, "value")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)], ir_version=10)
        for weight_type in [QuantType.QInt8, QuantType.QUInt8]:
            for per_channel in [False, True]:
                with self.subTest(weight_type=weight_type, per_channel=per_channel):
                    quantized = self._quantize(model, data, weight_type, per_channel)
                    self.assertEqual(list(quantized.graph.node), list(model.graph.node))

    def test_supported_attention_still_quantizes(self):
        for kind in ["ordinary", "empty_optional_inputs", "mask"]:
            for weight_type in [QuantType.QInt8, QuantType.QUInt8]:
                for per_channel in [False, True]:
                    with self.subTest(kind=kind, weight_type=weight_type, per_channel=per_channel):
                        quantized = self._quantize(self._make_contrib_model(kind), self.data, weight_type, per_channel)
                        self.assertEqual(sum(node.op_type == "QAttention" for node in quantized.graph.node), 1)

    def test_fallback_after_quantized_matmul(self):
        for weight_type in [QuantType.QInt8, QuantType.QUInt8]:
            for per_channel in [False, True]:
                with self.subTest(weight_type=weight_type, per_channel=per_channel):
                    quantized = self._quantize(
                        self._make_contrib_model("attention_bias", with_matmul=True),
                        self.data,
                        weight_type,
                        per_channel,
                    )
                    nodes = list(quantized.graph.node)
                    self.assertEqual(sum(node.op_type == "MatMulInteger" for node in nodes), 1)
                    self.assertFalse(any(node.op_type == "QAttention" for node in nodes))
                    attention = next(node for node in nodes if node.op_type == "Attention")
                    self.assertEqual(attention.input[5], "attention_bias")


if __name__ == "__main__":
    unittest.main()
