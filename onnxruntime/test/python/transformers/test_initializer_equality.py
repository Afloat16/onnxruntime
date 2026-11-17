# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from parity_utilities import find_transformers_source

if find_transformers_source():
    from onnx_model import OnnxModel
else:
    from onnxruntime.transformers.onnx_model import OnnxModel


class TestInitializerEquality(unittest.TestCase):
    def test_discrete_values_ignore_floating_tolerances(self):
        type_names = (
            "INT8",
            "UINT8",
            "INT16",
            "UINT16",
            "INT32",
            "UINT32",
            "INT64",
            "UINT64",
            "INT4",
            "UINT4",
            "INT2",
            "UINT2",
            "BOOL",
            "STRING",
        )
        for type_name in type_names:
            if not hasattr(TensorProto, type_name):
                continue
            with self.subTest(type=type_name):
                data_type = getattr(TensorProto, type_name)
                values = ["first", "second"] if data_type == TensorProto.STRING else [0, 1]
                first = helper.make_tensor("first", data_type, [1], [values[0]])
                second = helper.make_tensor("second", data_type, [1], [values[1]])
                duplicate = helper.make_tensor("duplicate", data_type, [1], [values[0]])
                self.assertFalse(OnnxModel.has_same_value(first, second, rtol=1, atol=1))
                self.assertTrue(OnnxModel.has_same_value(first, duplicate, rtol=1, atol=1))

    def test_large_integers_are_distinct_with_default_tolerances(self):
        cases = (
            (TensorProto.INT32, 1_000_000),
            (TensorProto.UINT32, 1_000_000),
            (TensorProto.INT64, 2**53),
            (TensorProto.UINT64, 2**63),
        )
        for data_type, value in cases:
            first = helper.make_tensor("first", data_type, [1], [value])
            second = helper.make_tensor("second", data_type, [1], [value + 1])
            for cache_mode in ("none", "empty", "populated", "collision"):
                with self.subTest(type=data_type, cache=cache_mode):
                    cache1, cache2 = None, None
                    if cache_mode == "empty":
                        cache1, cache2 = {}, {}
                    elif cache_mode == "populated":
                        cache1 = {first.name: OnnxModel.to_data_hash(first)}
                        cache2 = {second.name: OnnxModel.to_data_hash(second)}
                    elif cache_mode == "collision":
                        cache1, cache2 = {first.name: 0}, {second.name: 0}
                    self.assertFalse(OnnxModel.has_same_value(first, second, cache1, cache2))

    def test_equal_integer_encodings_and_metadata(self):
        for dtype in (np.int64, np.uint64):
            with self.subTest(dtype=dtype):
                data = np.array([0, 2**53 + 1], dtype=dtype)
                raw = numpy_helper.from_array(data, "raw")
                repeated = helper.make_tensor("repeated", raw.data_type, [2], data.tolist())
                cache = {}
                self.assertTrue(OnnxModel.has_same_value(raw, repeated, cache, cache))
                self.assertEqual(set(cache), {"raw", "repeated"})
                reshaped = helper.make_tensor("reshaped", raw.data_type, [1, 2], data.tolist())
                self.assertFalse(OnnxModel.has_same_value(raw, reshaped))
        self.assertFalse(
            OnnxModel.has_same_value(
                helper.make_tensor("int", TensorProto.INT64, [1], [1]),
                helper.make_tensor("float", TensorProto.FLOAT, [1], [1]),
            )
        )

    def test_floating_and_complex_tolerances_are_preserved(self):
        for dtype in (np.float16, np.float32, np.float64, np.complex64, np.complex128):
            with self.subTest(dtype=dtype):
                first = numpy_helper.from_array(np.array([1.0], dtype=dtype), "first")
                nearby = numpy_helper.from_array(np.array([1.000001], dtype=dtype), "nearby")
                distant = numpy_helper.from_array(np.array([1.1], dtype=dtype), "distant")
                self.assertTrue(OnnxModel.has_same_value(first, nearby))
                self.assertFalse(OnnxModel.has_same_value(first, distant))
                self.assertTrue(OnnxModel.has_same_value(first, distant, rtol=0.2))
        for values, expected in (([np.inf], True), ([np.nan], False), ([-0.0], True)):
            with self.subTest(values=values):
                first = numpy_helper.from_array(np.array(values), "first")
                second = numpy_helper.from_array(np.array(values) + 0.0, "second")
                self.assertEqual(OnnxModel.has_same_value(first, second), expected)

    def test_deduplication_preserves_distinct_discrete_outputs(self):
        cases = (
            (TensorProto.INT32, [1_000_000], [1_000_001]),
            (TensorProto.UINT32, [1_000_000], [1_000_001]),
            (TensorProto.INT64, [2**53], [2**53 + 1]),
            (TensorProto.UINT64, [2**63], [2**63 + 1]),
            (TensorProto.STRING, ["first"], ["second"]),
            (TensorProto.BOOL, [False], [True]),
        )
        for data_type, first_values, second_values in cases:
            for use_cache in (False, True):
                with self.subTest(type=data_type, cache=use_cache):
                    names = ("first", "second", "duplicate")
                    initializers = [
                        helper.make_tensor(name, data_type, [1], values)
                        for name, values in zip(names, (first_values, second_values, first_values), strict=True)
                    ]
                    model = helper.make_model(
                        helper.make_graph(
                            [helper.make_node("Identity", [name], [f"{name}_out"]) for name in names],
                            "discrete_initializers",
                            [],
                            [helper.make_tensor_value_info(f"{name}_out", data_type, [1]) for name in names],
                            initializers,
                        ),
                        opset_imports=[helper.make_opsetid("", 13)],
                    )
                    onnx.checker.check_model(model, full_check=True)
                    optimized = OnnxModel(model)
                    optimized.remove_duplicated_initializer({} if use_cache else None)
                    onnx.checker.check_model(optimized.model, full_check=True)
                    self.assertEqual([tensor.name for tensor in optimized.model.graph.initializer], ["first", "second"])
                    self.assertEqual(
                        [node.input[0] for node in optimized.model.graph.node], ["first", "second", "first"]
                    )
                    np.testing.assert_array_equal(
                        numpy_helper.to_array(optimized.get_initializer("first")), first_values
                    )
                    np.testing.assert_array_equal(
                        numpy_helper.to_array(optimized.get_initializer("second")), second_values
                    )


if __name__ == "__main__":
    unittest.main()
