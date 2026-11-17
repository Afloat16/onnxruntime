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
from onnx import TensorProto, helper, numpy_helper
from op_test_utils import TestDataFeeds

from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.qdq_loss_debug import (
    compute_activation_error,
    compute_signal_to_quantization_noice_ratio,
    compute_weight_error,
    create_weight_matching,
)


class TestScalarQuantizationError(unittest.TestCase):
    def test_quantized_scalar_weight_error(self):
        for op_type in ("Add", "Mul"):
            for weight in (-2.5, 0.0, 1.25):
                with self.subTest(op_type=op_type, weight=weight), tempfile.TemporaryDirectory() as directory:
                    model = helper.make_model(
                        helper.make_graph(
                            [helper.make_node(op_type, ["input", "weight"], ["output"])],
                            "scalar_weight",
                            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 2])],
                            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 2])],
                            [numpy_helper.from_array(np.array(weight, dtype=np.float32), "weight")],
                        ),
                        opset_imports=[helper.make_opsetid("", 13)],
                        ir_version=10,
                    )
                    float_path = Path(directory) / "float.onnx"
                    quant_path = Path(directory) / "quant.onnx"
                    onnx.save(model, float_path)
                    reader = TestDataFeeds([{"input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}])
                    quantize_static(
                        float_path,
                        quant_path,
                        reader,
                        quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8,
                        weight_type=QuantType.QInt8,
                    )
                    onnx.checker.check_model(onnx.load(quant_path), full_check=True)
                    matched = create_weight_matching(str(float_path), str(quant_path))
                    self.assertEqual(set(matched), {"weight"})
                    self.assertEqual(matched["weight"]["float"].shape, ())
                    self.assertEqual(matched["weight"]["dequantized"].shape, ())
                    expected = float(matched["weight"]["float"])
                    actual = float(matched["weight"]["dequantized"])
                    epsilon = np.finfo("float").eps
                    expected_score = 20 * math.log10(max(abs(expected), epsilon) / max(abs(expected - actual), epsilon))
                    scores = compute_weight_error(matched)
                    self.assertEqual(set(scores), {"weight"})
                    self.assertAlmostEqual(scores["weight"], expected_score, places=5)

    def test_scalar_tensor_representations(self):
        for dtype in (np.float16, np.float32, np.float64):
            for representation in ("array", "scalar", "array_list", "scalar_list"):
                with self.subTest(dtype=dtype, representation=representation):
                    left, right = np.array(4.0, dtype=dtype), np.array(2.0, dtype=dtype)
                    if representation.startswith("scalar"):
                        left, right = left[()], right[()]
                    if representation.endswith("list"):
                        left, right = [left], [right]
                    score = compute_signal_to_quantization_noice_ratio(left, right)
                    self.assertAlmostEqual(score, 20 * math.log10(2.0))

    def test_activation_error_for_tensor_sequences(self):
        arrays = np.arange(1.0, 25.0).reshape(4, 6)
        cases = [
            [np.array(4.0), np.array(8.0)],
            [np.ones((1, 2, 3)), np.ones((1, 4, 3))],
            [arrays[:, ::2], arrays[:, 1::2]],
            [arrays[:2], arrays[2:]],
        ]
        for left in cases:
            with self.subTest(shapes=[item.shape for item in left]):
                right = [item * 0.5 for item in left]
                errors = compute_activation_error({"activation": {"pre_qdq": left, "post_qdq": right, "float": left}})
                self.assertEqual(set(errors["activation"]), {"qdq_err", "xmodel_err"})
                for score in errors["activation"].values():
                    self.assertAlmostEqual(score, 20 * math.log10(2.0))

    def test_unequal_tensor_counts_still_raise(self):
        with self.assertRaisesRegex(RuntimeError, "Unequal number of tensors"):
            compute_signal_to_quantization_noice_ratio([np.array([1.0]), np.array([2.0])], [np.array([1.0])])


if __name__ == "__main__":
    unittest.main()
