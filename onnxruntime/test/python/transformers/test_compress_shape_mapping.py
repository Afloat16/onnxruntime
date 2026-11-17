# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import unittest

import numpy as np
from onnx import TensorProto, helper, numpy_helper
from parity_utilities import find_transformers_source

if find_transformers_source():
    from shape_infer_helper import SymbolicShapeInferenceHelper
else:
    from onnxruntime.transformers.shape_infer_helper import SymbolicShapeInferenceHelper


class TestCompressShapeMapping(unittest.TestCase):
    def test_resolved_extent_with_explicit_or_flattened_axis(self):
        for axis in (None, 0, -1):
            for extent, expected in ((0, 0), (2, 1), (4, 2)):
                with self.subTest(axis=axis, extent=extent):
                    model = helper.make_model(
                        helper.make_graph(
                            [
                                helper.make_node(
                                    "Compress",
                                    ["input", "condition"],
                                    ["output"],
                                    **({"axis": axis} if axis is not None else {}),
                                )
                            ],
                            "mapped_compress",
                            [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["B"])],
                            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None])],
                            [numpy_helper.from_array(np.array([True, False, True]), "condition")],
                        ),
                        opset_imports=[helper.make_opsetid("", 13)],
                    )
                    inference = SymbolicShapeInferenceHelper(model)
                    self.assertTrue(inference.infer({"B": extent}))
                    self.assertEqual(inference.get_edge_shape("output"), [expected])


if __name__ == "__main__":
    unittest.main()
