# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import onnxruntime
from onnxruntime.quantization.qdq_loss_debug import create_weight_matching


class TestWeightMatchingDefaults(unittest.TestCase):
    def _check_weight_matching(self, shape, dtype, axis, zero_point, per_channel=True):
        weights = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        if np.issubdtype(dtype, np.signedinteger):
            weights -= weights.size // 2
        effective_axis = 1 if axis is None else axis
        scale = (
            np.arange(1, shape[effective_axis] + 1, dtype=np.float32) / 4
            if per_channel
            else np.array(0.25, dtype=np.float32)
        )
        inputs = ["weight_quantized", "scale"]
        initializers = [
            numpy_helper.from_array(weights, "weight_quantized"),
            numpy_helper.from_array(scale, "scale"),
        ]
        if zero_point == "explicit":
            inputs.append("zero_point")
            initializers.append(numpy_helper.from_array(np.zeros(scale.shape, dtype=dtype), "zero_point"))
        elif zero_point == "empty":
            inputs.append("")

        node = helper.make_node("DequantizeLinear", inputs, ["output"], **({} if axis is None else {"axis": axis}))
        qdq_model = helper.make_model(
            helper.make_graph(
                [node],
                "qdq_weights",
                [],
                [helper.make_tensor_value_info("output", TensorProto.FLOAT, list(shape))],
                initializers,
            ),
            opset_imports=[helper.make_opsetid("", 13)],
        )
        onnx.checker.check_model(qdq_model, full_check=True)
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        expected = onnxruntime.InferenceSession(
            qdq_model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        ).run(None, {})[0]

        float_model = helper.make_model(
            helper.make_graph(
                [],
                "float_weights",
                [],
                [helper.make_tensor_value_info("weight", TensorProto.FLOAT, list(shape))],
                [numpy_helper.from_array(expected, "weight")],
            ),
            opset_imports=[helper.make_opsetid("", 13)],
        )
        with tempfile.TemporaryDirectory() as directory:
            float_path = Path(directory) / "float.onnx"
            qdq_path = Path(directory) / "qdq.onnx"
            onnx.save(float_model, float_path)
            onnx.save(qdq_model, qdq_path)
            matched = create_weight_matching(str(float_path), str(qdq_path))

        self.assertEqual(set(matched), {"weight"})
        actual = matched["weight"]["dequantized"]
        self.assertEqual(actual.dtype, expected.dtype)
        np.testing.assert_array_equal(actual, expected, strict=True)
        np.testing.assert_array_equal(matched["weight"]["float"], expected, strict=True)

    def test_weight_matching_uses_default_axis(self):
        for shape in ((2, 3, 4), (2, 3, 3)):
            for dtype in (np.uint8, np.int8):
                with self.subTest(shape=shape, dtype=dtype):
                    self._check_weight_matching(shape, dtype, axis=None, zero_point="explicit")

    def test_weight_matching_accepts_omitted_zero_point(self):
        for zero_point in ("trailing", "empty"):
            for dtype in (np.uint8, np.int8):
                for axis in (None, 0, -1):
                    for per_channel in (False, True):
                        with self.subTest(zero_point=zero_point, dtype=dtype, axis=axis, per_channel=per_channel):
                            self._check_weight_matching((2, 3, 4), dtype, axis, zero_point, per_channel)

    def test_weight_matching_preserves_explicit_parameters(self):
        for dtype in (np.uint8, np.int8):
            for axis in (0, 1, -1):
                for per_channel in (False, True):
                    with self.subTest(dtype=dtype, axis=axis, per_channel=per_channel):
                        self._check_weight_matching((2, 3, 4), dtype, axis, "explicit", per_channel)


if __name__ == "__main__":
    unittest.main()
