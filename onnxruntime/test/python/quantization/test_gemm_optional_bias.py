# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from op_test_utils import TestDataFeeds

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic, quantize_static


class TestGemmOptionalBias(unittest.TestCase):
    @staticmethod
    def _make_model(opset, trans_b, bias=None):
        weights = np.array([[0.25, 0.5, -0.25, 1.0], [-0.5, 0.75, 1.25, -1.0], [1.5, -0.75, 0.25, 0.5]], np.float32)
        initializers = [numpy_helper.from_array(weights.T if trans_b else weights, "weight")]
        inputs = ["input", "weight"]
        if bias is not None:
            inputs.append("bias")
            initializers.append(numpy_helper.from_array(bias, "bias"))
        return helper.make_model(
            helper.make_graph(
                [helper.make_node("Gemm", inputs, ["output"], name="gemm", transB=trans_b)],
                "optional_bias",
                [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3])],
                [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])],
                initializers,
            ),
            opset_imports=[helper.make_opsetid("", opset)],
            ir_version=10,
        )

    @staticmethod
    def _quantize(model, path, mode, per_channel, feeds):
        if mode == "dynamic":
            quantize_dynamic(model, path, weight_type=QuantType.QInt8, per_channel=per_channel)
        else:
            quantize_static(
                model,
                path,
                TestDataFeeds([feeds]),
                quant_format=QuantFormat.QDQ if mode == "qdq" else QuantFormat.QOperator,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=per_channel,
            )
        quantized = onnx.load(path)
        onnx.checker.check_model(quantized, full_check=True)
        return quantized

    @staticmethod
    def _run(model, feeds, optimization_level):
        options = SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = optimization_level
        return InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"]).run(
            None, feeds
        )[0]

    def _check_missing_bias(self, mode, opsets):
        feeds = {"input": np.array([[1.0, 0.5, -0.25], [-0.5, 0.25, 1.0]], dtype=np.float32)}
        for opset, trans_b, per_channel in itertools.product(opsets, (0, 1), (False, True)):
            with (
                self.subTest(mode=mode, opset=opset, trans_b=trans_b, per_channel=per_channel),
                tempfile.TemporaryDirectory() as directory,
            ):
                omitted = self._make_model(opset, trans_b)
                explicit_empty = copy.deepcopy(omitted)
                explicit_empty.graph.node[0].input.append("")
                for model in (omitted, explicit_empty):
                    onnx.checker.check_model(model, full_check=True)
                expected = self._run(omitted, feeds, GraphOptimizationLevel.ORT_DISABLE_ALL)
                np.testing.assert_array_equal(
                    self._run(explicit_empty, feeds, GraphOptimizationLevel.ORT_DISABLE_ALL), expected
                )
                reference = self._quantize(omitted, Path(directory) / "omitted.onnx", mode, per_channel, feeds)
                actual = self._quantize(explicit_empty, Path(directory) / "empty.onnx", mode, per_channel, feeds)
                expected_op = {"dynamic": "MatMulInteger", "qoperator": "QGemm", "qdq": "Gemm"}[mode]
                self.assertEqual(sum(node.op_type == expected_op for node in actual.graph.node), 1)
                if mode != "qdq":
                    self.assertEqual(actual.SerializeToString(), reference.SerializeToString())
                for level in (GraphOptimizationLevel.ORT_DISABLE_ALL, GraphOptimizationLevel.ORT_ENABLE_ALL):
                    reference_output = self._run(reference, feeds, level)
                    actual_output = self._run(actual, feeds, level)
                    np.testing.assert_array_equal(actual_output, reference_output)
                    np.testing.assert_allclose(actual_output, expected, rtol=0.02, atol=0.03)

    def test_dynamic_explicit_empty_bias(self):
        self._check_missing_bias("dynamic", (11, 13, 21))

    def test_qoperator_explicit_empty_bias(self):
        self._check_missing_bias("qoperator", (11, 13, 21))

    def test_qdq_explicit_empty_bias(self):
        self._check_missing_bias("qdq", (13,))

    def test_present_bias_is_preserved(self):
        feeds = {"input": np.array([[1.0, 0.5, -0.25], [-0.5, 0.25, 1.0]], dtype=np.float32)}
        for mode, trans_b in itertools.product(("dynamic", "qoperator", "qdq"), (0, 1)):
            with self.subTest(mode=mode, trans_b=trans_b), tempfile.TemporaryDirectory() as directory:
                bias = np.array([0.125, 0.25, 0.375, 0.5], dtype=np.float32)
                model = self._make_model(13, trans_b, bias)
                expected = self._run(model, feeds, GraphOptimizationLevel.ORT_DISABLE_ALL)
                quantized = self._quantize(model, Path(directory) / "quantized.onnx", mode, True, feeds)
                actual = self._run(quantized, feeds, GraphOptimizationLevel.ORT_DISABLE_ALL)
                np.testing.assert_allclose(actual, expected, rtol=0.02, atol=0.03)


if __name__ == "__main__":
    unittest.main()
