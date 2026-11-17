# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
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
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static


class TestPadSharedInitializer(unittest.TestCase):
    def test_shared_padding_constant(self):
        for consumer, activation_type, named in itertools.product(
            ("single", "identity", "output", "pad", "add"), (QuantType.QUInt8, QuantType.QInt8), (False, True)
        ):
            with self.subTest(consumer=consumer, activation_type=activation_type, named=named):
                self.check_padding_constant(consumer, activation_type, named)

    def test_padding_constant_does_not_shadow_subgraphs(self):
        for consumer, activation_type, named in itertools.product(
            ("if", "nested_if"), (QuantType.QUInt8, QuantType.QInt8), (False, True)
        ):
            with self.subTest(consumer=consumer, activation_type=activation_type, named=named):
                self.check_padding_constant(consumer, activation_type, named)

    def check_padding_constant(self, consumer, activation_type, named):
        inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])]
        outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])]
        nodes = [
            helper.make_node("Add", ["X", "X"], ["A"]),
            helper.make_node("Pad", ["A", "pads", "constant"], ["Y"], name="pad" if named else ""),
        ]
        initializers = [
            numpy_helper.from_array(np.array([1, 1], dtype=np.int64), "pads"),
            numpy_helper.from_array(np.array(0.5, dtype=np.float32), "constant"),
        ]
        feeds = {"X": np.array([-1, 1], dtype=np.float32)}

        if consumer == "identity":
            nodes.append(helper.make_node("Identity", ["constant"], ["constant_output"]))
            outputs.append(helper.make_tensor_value_info("constant_output", TensorProto.FLOAT, []))
        elif consumer == "output":
            outputs.append(helper.make_tensor_value_info("constant", TensorProto.FLOAT, []))
        elif consumer == "pad":
            inputs.append(helper.make_tensor_value_info("W", TensorProto.FLOAT, [2]))
            feeds["W"] = np.array([-10, 10], dtype=np.float32)
            nodes.extend(
                [
                    helper.make_node("Add", ["W", "W"], ["B"]),
                    helper.make_node("Pad", ["B", "pads", "constant"], ["Z"], name="pad2" if named else ""),
                ]
            )
            outputs.append(helper.make_tensor_value_info("Z", TensorProto.FLOAT, [4]))
            # Existing names must not be reused for a padding value, including
            # names occupied by graph inputs/outputs instead of initializers.
            initializers.append(numpy_helper.from_array(np.array(7, dtype=np.float32), "Y_pad_value"))
            nodes.append(helper.make_node("Identity", ["Y_pad_value"], ["Y_pad_value_0"]))
            outputs.append(helper.make_tensor_value_info("Y_pad_value_0", TensorProto.FLOAT, []))
        elif consumer == "add":
            nodes.append(helper.make_node("Add", ["X", "constant"], ["Z"]))
            outputs.append(helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2]))
        elif consumer in ("if", "nested_if"):
            inputs.append(helper.make_tensor_value_info("condition", TensorProto.BOOL, []))
            feeds["condition"] = np.array(True)
            then_branch = helper.make_graph(
                [
                    helper.make_node(
                        "Constant",
                        [],
                        ["Y_pad_value"],
                        value=numpy_helper.from_array(np.array(7, dtype=np.float32)),
                    )
                ],
                "then",
                [],
                [helper.make_tensor_value_info("Y_pad_value", TensorProto.FLOAT, [])],
            )
            else_branch = helper.make_graph(
                [],
                "else",
                [],
                [helper.make_tensor_value_info("Y_pad_value_0", TensorProto.FLOAT, [])],
                [numpy_helper.from_array(np.array(9, dtype=np.float32), "Y_pad_value_0")],
            )
            if consumer == "nested_if":
                then_branch = helper.make_graph(
                    [
                        helper.make_node(
                            "If", ["condition"], ["nested_output"], then_branch=then_branch, else_branch=else_branch
                        )
                    ],
                    "nested",
                    [],
                    [helper.make_tensor_value_info("nested_output", TensorProto.FLOAT, [])],
                )
            nodes.append(helper.make_node("If", ["condition"], ["Z"], then_branch=then_branch, else_branch=else_branch))
            outputs.append(helper.make_tensor_value_info("Z", TensorProto.FLOAT, []))

        model = helper.make_model(
            helper.make_graph(nodes, "shared-pad-constant", inputs, outputs, initializers),
            opset_imports=[helper.make_opsetid("", 13)],
            ir_version=7,
        )
        onnx.checker.check_model(model, full_check=True)
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        with tempfile.TemporaryDirectory() as directory:
            float_path = Path(directory) / "float.onnx"
            quant_path = Path(directory) / "quant.onnx"
            onnx.save(model, float_path)
            quantize_static(
                float_path,
                quant_path,
                TestDataFeeds([feeds]),
                quant_format=QuantFormat.QOperator,
                activation_type=activation_type,
                weight_type=activation_type,
                op_types_to_quantize=["Add", "Pad"],
            )
            quantized = onnx.load(quant_path)
            onnx.checker.check_model(quantized, full_check=True)
            actual = ort.InferenceSession(
                str(quant_path), sess_options=session_options, providers=["CPUExecutionProvider"]
            ).run(None, feeds)
            if consumer in ("if", "nested_if"):
                np.testing.assert_array_equal(actual[1], np.array(7, dtype=np.float32))
                feeds["condition"] = np.array(False)
                false_result = ort.InferenceSession(
                    str(quant_path), sess_options=session_options, providers=["CPUExecutionProvider"]
                ).run(None, feeds)
                np.testing.assert_array_equal(false_result[1], np.array(9, dtype=np.float32))

        quant_inits = {value.name: numpy_helper.to_array(value) for value in quantized.graph.initializer}
        pads = [node for node in quantized.graph.node if node.op_type == "Pad"]
        self.assertEqual(len(pads), 2 if consumer == "pad" else 1)
        self.assertEqual(len({node.input[2] for node in pads}), len(pads))
        output_scales = {}
        for node in quantized.graph.node:
            if node.op_type == "DequantizeLinear":
                output_scales[node.output[0]] = quant_inits[node.input[1]].item()

        # The two Pad inputs have ranges differing by a factor of ten, so
        # sharing the first padding value's quantized bytes would be incorrect.
        for output_name, expected, actual_index in (
            [("Y", [0.5, -2, 2, 0.5], 0)]
            + ([("Z", [0.5, -20, 20, 0.5], 1)] if consumer == "pad" else [])
            + ([("Z", [-0.5, 1.5], 1)] if consumer == "add" else [])
        ):
            np.testing.assert_allclose(actual[actual_index], expected, atol=output_scales[output_name], rtol=0)
        if consumer in ("identity", "output"):
            np.testing.assert_array_equal(quant_inits["constant"], np.array(0.5, dtype=np.float32))
            np.testing.assert_array_equal(actual[1], np.array(0.5, dtype=np.float32))
        else:
            self.assertNotIn("constant", quant_inits)  # Normal cleanup still removes the unused original.
        if consumer == "pad":
            np.testing.assert_array_equal(quant_inits["Y_pad_value"], np.array(7, dtype=np.float32))
            np.testing.assert_array_equal(actual[2], np.array(7, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
