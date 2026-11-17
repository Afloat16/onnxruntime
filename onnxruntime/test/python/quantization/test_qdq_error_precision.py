# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

import onnxruntime
from onnxruntime.quantization.qdq_loss_debug import (
    compute_activation_error,
    compute_signal_to_quantization_noice_ratio,
    compute_weight_error,
    create_weight_matching,
)


class TestQuantizationErrorPrecision(unittest.TestCase):
    def _check_score(self, left, right, places=10):
        left_values = np.concatenate(left).ravel() if isinstance(left, list) else left.ravel()
        right_values = np.concatenate(right).ravel() if isinstance(right, list) else right.ravel()
        signal = math.hypot(*(abs(value.item()) for value in left_values))
        noise = math.hypot(*(abs(x.item() - y.item()) for x, y in zip(left_values, right_values, strict=True)))
        epsilon = np.finfo(float).eps
        expected = 20 * math.log10(max(signal, epsilon) / max(noise, epsilon))
        with np.errstate(over="raise", invalid="raise"):
            actual = compute_signal_to_quantization_noice_ratio(left, right)
        self.assertTrue(math.isfinite(actual))
        self.assertAlmostEqual(actual, expected, places=places)
        return expected

    def test_low_precision_norms_and_subtraction(self):
        cases = [
            (np.float16, [400, 800], [200, 400]),
            (np.float16, [400, 800], [400, 800]),
            (np.float16, [0.0001, 0.0002], [0.00005, 0.0001]),
            (np.float16, [65504, -65504], [-65504, 65504]),
            (np.float32, [1e20, 2e20], [5e19, 1e20]),
            (np.complex64, [1e20 + 2e20j, -2e20 + 1e20j], [5e19 + 1e20j, -1e20 + 5e19j]),
        ]
        for dtype, left, right in cases:
            with self.subTest(dtype=dtype, left=left, right=right):
                self._check_score(np.array(left, dtype=dtype), np.array(right, dtype=dtype))

    def test_integer_subtraction_does_not_wrap(self):
        for dtype in [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]:
            with self.subTest(dtype=dtype):
                limits = np.iinfo(dtype)
                left = np.array([limits.min, limits.max], dtype=dtype)
                self._check_score(left, left[::-1])

    def test_large_integer_unit_differences(self):
        for left_type in [np.int64, np.uint64]:
            for right_type in [np.int64, np.uint64]:
                with self.subTest(left_type=left_type, right_type=right_type):
                    left = np.array([2**60, 2**60 + 4], dtype=left_type)
                    right = np.array([2**60 - 1, 2**60 + 3], dtype=right_type)
                    self._check_score(left, right)

    def test_regular_values_and_epsilon_floor(self):
        for dtype in [np.float16, np.float32, np.float64, np.complex64, np.complex128]:
            for values in [([0, 0], [0, 0]), ([1, 2], [1, 2]), ([4, 8], [2, 4])]:
                with self.subTest(dtype=dtype, values=values):
                    self._check_score(np.array(values[0], dtype=dtype), np.array(values[1], dtype=dtype), places=2)
        self._check_score(np.array([400, 800], dtype=np.float16), np.array([200, 400], dtype=np.float32))

    def test_activation_error_batches(self):
        for dtype, magnitude in [(np.float16, 400), (np.float32, 1e20)]:
            with self.subTest(dtype=dtype):
                left = [np.full((2, 3), magnitude * factor, dtype=dtype) for factor in [1, 2]]
                right = [array * 0.5 for array in left]
                expected = self._check_score(left, right)
                actual = compute_activation_error({"activation": {"pre_qdq": left, "post_qdq": right, "float": left}})
                self.assertAlmostEqual(actual["activation"]["qdq_err"], expected, places=10)
                self.assertAlmostEqual(actual["activation"]["xmodel_err"], expected, places=10)

    def test_dequantized_weight_error(self):
        for dtype, scale_value in [(np.float16, 512), (np.float32, 1e18)]:
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory(prefix="qdq_error_precision.") as directory:
                elem_type = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
                quantized = np.array([[100, -100], [64, -64]], dtype=np.int8)
                scale = np.array(scale_value, dtype=dtype)
                dequantized = quantized.astype(dtype) * scale
                weights = np.nextafter(dequantized, np.full((2, 2), np.inf, dtype=dtype))
                output = helper.make_tensor_value_info("output", elem_type, [2, 2])
                float_model = helper.make_model(
                    helper.make_graph(
                        [helper.make_node("Identity", ["weight"], ["output"])],
                        "float_weight",
                        [],
                        [output],
                        [numpy_helper.from_array(weights, "weight")],
                    ),
                    opset_imports=[helper.make_opsetid("", 19)],
                    ir_version=10,
                )
                quantized_model = helper.make_model(
                    helper.make_graph(
                        [helper.make_node("DequantizeLinear", ["weight_quantized", "scale", "zero_point"], ["output"])],
                        "quantized_weight",
                        [],
                        [output],
                        [
                            numpy_helper.from_array(quantized, "weight_quantized"),
                            numpy_helper.from_array(scale, "scale"),
                            numpy_helper.from_array(np.array(0, dtype=np.int8), "zero_point"),
                        ],
                    ),
                    opset_imports=[helper.make_opsetid("", 19)],
                    ir_version=10,
                )
                float_path, quantized_path = Path(directory) / "float.onnx", Path(directory) / "quantized.onnx"
                for model, path in [(float_model, float_path), (quantized_model, quantized_path)]:
                    onnx.checker.check_model(model, full_check=True)
                    onnx.save(model, path)
                matched = create_weight_matching(str(float_path), str(quantized_path))
                options = onnxruntime.SessionOptions()
                options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
                options.intra_op_num_threads = options.inter_op_num_threads = 1
                session = onnxruntime.InferenceSession(str(quantized_path), options, providers=["CPUExecutionProvider"])
                np.testing.assert_array_equal(matched["weight"]["dequantized"], session.run(None, {})[0])
                expected = self._check_score(weights, dequantized)
                self.assertAlmostEqual(compute_weight_error(matched)["weight"], expected, places=10)


if __name__ == "__main__":
    unittest.main()
