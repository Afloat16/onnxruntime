# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import onnxruntime
from onnxruntime.quantization.onnx_model import ONNXModel


class TestFloat8InitializerValidation(unittest.TestCase):
    @staticmethod
    def _tensor(data, name="weight", dtype=TensorProto.FLOAT8E4M3FN):
        tensor = TensorProto()
        tensor.name = name
        tensor.data_type = dtype
        tensor.dims.extend([len(data)])
        tensor.raw_data = data
        return tensor

    @staticmethod
    def _model(initializers=()):
        return ONNXModel(helper.make_model(helper.make_graph([], "validation", [], [], initializers)))

    def test_all_float8_codes_follow_decoded_nan_classification(self):
        for code in range(256):
            with self.subTest(code=code):
                tensor = self._tensor(bytes([code]))
                original = tensor.SerializeToString()
                is_nan = bool(np.isnan(numpy_helper.to_array(tensor).astype(np.float32)[0]))
                model = self._model()
                if is_nan:
                    with self.assertRaisesRegex(ValueError, "Initializer 'weight' has nan"):
                        model.add_initializer(tensor)
                    self.assertEqual(len(model.initializer()), 0)
                else:
                    model.add_initializer(tensor)
                    self.assertEqual(model.initializer()[0].SerializeToString(), original)
                self.assertEqual(tensor.SerializeToString(), original)

    def test_nan_position_and_insertion_methods(self):
        valid = bytes(code for code in range(256) if code not in (0x7F, 0xFF))
        for nan_code in (0x7F, 0xFF):
            for position in (0, len(valid) // 2, len(valid)):
                for method in ("add", "extend"):
                    with self.subTest(nan_code=nan_code, position=position, method=method):
                        model = self._model([self._tensor(b"\x00", "existing")])
                        original = model.model.SerializeToString()
                        data = valid[:position] + bytes([nan_code]) + valid[position:]
                        tensor = self._tensor(data)
                        with self.assertRaisesRegex(ValueError, "Initializer 'weight' has nan"):
                            if method == "add":
                                model.add_initializer(tensor)
                            else:
                                model.initializer_extend([tensor])
                        self.assertEqual(model.model.SerializeToString(), original)

    def test_initializer_extend_checks_existing_weights(self):
        for nan_code in (0x7F, 0xFF):
            with self.subTest(nan_code=nan_code):
                model = self._model([self._tensor(bytes([nan_code]), "existing")])
                original = model.model.SerializeToString()
                with self.assertRaisesRegex(ValueError, "Initializer 'existing' has nan"):
                    model.initializer_extend([self._tensor(b"\x00")])
                self.assertEqual(model.model.SerializeToString(), original)

    def test_empty_and_other_storage_formats_are_preserved(self):
        empty = self._tensor(b"")
        typed = self._tensor(b"\x00\x80")
        typed.ClearField("raw_data")
        typed.int32_data.extend([0, 128])
        external = self._tensor(b"\x00")
        onnx.external_data_helper.set_external_data(external, location="unloaded_float8.bin")
        external.ClearField("raw_data")
        for tensor in (
            empty,
            typed,
            external,
            self._tensor(b"\x7f\xff", dtype=TensorProto.INT8),
            self._tensor(b"\x7f\xff", dtype=TensorProto.UINT8),
        ):
            with self.subTest(dtype=tensor.data_type, tensor=tensor):
                original = tensor.SerializeToString()
                model = self._model()
                model.add_initializer(tensor)
                self.assertEqual(model.initializer()[0].SerializeToString(), original)

    def test_valid_float8_payload_runs_after_insertion(self):
        data = bytes(code for code in range(256) if code not in (0x7F, 0xFF))
        tensor = self._tensor(data)
        model = ONNXModel(
            helper.make_model(
                helper.make_graph(
                    [helper.make_node("Cast", ["weight"], ["output"], to=TensorProto.FLOAT)],
                    "float8_cast",
                    [],
                    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [len(data)])],
                ),
                opset_imports=[helper.make_opsetid("", 21)],
                ir_version=10,
            )
        )
        model.add_initializer(tensor)
        onnx.checker.check_model(model.model, full_check=True)
        reloaded = onnx.load_from_string(model.model.SerializeToString())
        self.assertEqual(reloaded.graph.initializer[0].raw_data, data)
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = onnxruntime.InferenceSession(
            reloaded.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        actual = session.run(None, {})[0]
        expected = numpy_helper.to_array(tensor).astype(np.float32)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


if __name__ == "__main__":
    unittest.main()
