#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""Tests for float16 conversion (convert_float_to_float16)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from parameterized import parameterized
from parity_utilities import find_transformers_source

if find_transformers_source():
    from float16 import (
        _npfloat16_to_int,
        convert_float_to_float16,
        convert_np_to_float16,
        convert_tensor_float_to_float16,
    )
else:
    from onnxruntime.transformers.float16 import (
        _npfloat16_to_int,
        convert_float_to_float16,
        convert_np_to_float16,
        convert_tensor_float_to_float16,
    )


def _make_resize_model_opset11(num_resize_nodes=2, use_empty_names=True, share_scales=False):
    """Create a minimal ONNX model with multiple Resize nodes (opset 11+).

    Resize opset 11+: inputs are [X, roi, scales, sizes].
    Scales (index 2) must stay float32 per ALWAYS_FLOAT_INPUTS; roi (index 1) allows fp16.
    """
    graph_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])
    graph_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 8, 8])

    nodes = []
    prev_output = "input"
    for idx in range(num_resize_nodes):
        roi_name = f"roi_{idx}"
        scales_name = "scales" if share_scales else f"scales_{idx}"
        output_name = f"resize_out_{idx}" if idx < num_resize_nodes - 1 else "output"

        node = helper.make_node(
            "Resize",
            inputs=[prev_output, roi_name, scales_name],
            outputs=[output_name],
            name="" if use_empty_names else f"Resize_{idx}",
            mode="nearest",
        )
        nodes.append(node)
        prev_output = output_name

    initializers = []
    for idx in range(num_resize_nodes):
        roi = numpy_helper.from_array(np.array([], dtype=np.float32), name=f"roi_{idx}")
        initializers.append(roi)
        if not share_scales or idx == 0:
            scales = numpy_helper.from_array(
                np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name="scales" if share_scales else f"scales_{idx}"
            )
            initializers.append(scales)

    value_info = [helper.make_tensor_value_info("scales", TensorProto.FLOAT, [4])] if share_scales else None
    graph = helper.make_graph(
        nodes, "resize_test", [graph_input], [graph_output], initializer=initializers, value_info=value_info
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model = onnx.shape_inference.infer_shapes(model)
    return model


def _make_resize_model_opset10(num_resize_nodes=1, use_empty_names=True):
    """Create a minimal ONNX model with Resize nodes using opset 10.

    Resize opset 10: inputs are [X, scales].
    Scales (index 1) must stay float32.
    """
    graph_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])
    graph_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 8, 8])

    nodes = []
    prev_output = "input"
    initializers = []
    for idx in range(num_resize_nodes):
        scales_name = f"scales_{idx}"
        output_name = f"resize_out_{idx}" if idx < num_resize_nodes - 1 else "output"

        node = helper.make_node(
            "Resize",
            inputs=[prev_output, scales_name],
            outputs=[output_name],
            name="" if use_empty_names else f"Resize_{idx}",
            mode="nearest",
        )
        nodes.append(node)
        prev_output = output_name

        scales = numpy_helper.from_array(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name=scales_name)
        initializers.append(scales)

    graph = helper.make_graph(nodes, "resize_opset10_test", [graph_input], [graph_output], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    model = onnx.shape_inference.infer_shapes(model)
    return model


def _make_blocked_node_model(num_nodes=2, use_empty_names=True):
    """Create a model with multiple blocked op nodes (using Upsample, which is in DEFAULT_OP_BLOCK_LIST).

    Tests that Cast nodes for blocked ops also get unique names.
    """
    graph_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])
    graph_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 16, 16])

    nodes = []
    prev_output = "input"
    for idx in range(num_nodes):
        scales_name = f"scales_{idx}"
        output_name = f"upsample_out_{idx}" if idx < num_nodes - 1 else "output"

        node = helper.make_node(
            "Upsample",
            inputs=[prev_output, scales_name],
            outputs=[output_name],
            name="" if use_empty_names else f"Upsample_{idx}",
            mode="nearest",
        )
        nodes.append(node)
        prev_output = output_name

    initializers = []
    for idx in range(num_nodes):
        scales = numpy_helper.from_array(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name=f"scales_{idx}")
        initializers.append(scales)

    graph = helper.make_graph(nodes, "blocked_node_test", [graph_input], [graph_output], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 9)])
    model = onnx.shape_inference.infer_shapes(model)
    return model


def _make_mixed_float_bfloat16_model():
    """Create a model where one value feeds both a FLOAT and a BFLOAT16 cast."""
    data_input = helper.make_tensor_value_info("data", TensorProto.FLOAT, [1, 1, 2, 2])
    shared_input = helper.make_tensor_value_info("shared", TensorProto.FLOAT, [4])
    resize_output = helper.make_tensor_value_info("resize_output", TensorProto.FLOAT, [1, 1, 4, 4])
    upsample_output = helper.make_tensor_value_info("upsample_output", TensorProto.FLOAT, [1, 1, 4, 4])
    nodes = [
        helper.make_node("Resize", ["data", "roi", "shared"], ["resize_output"], mode="nearest"),
        helper.make_node("Upsample", ["data", "shared"], ["upsample_output"], mode="nearest"),
    ]
    roi = numpy_helper.from_array(np.array([], dtype=np.float32), name="roi")
    graph = helper.make_graph(
        nodes,
        "mixed_float_bfloat16_test",
        [data_input, shared_input],
        [resize_output, upsample_output],
        initializer=[roi],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def _make_constant_of_shape_model():
    """Create a model with a ConstantOfShape node that omits its optional value attribute."""
    graph_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    graph_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    nodes = [
        helper.make_node("Shape", ["input"], ["shape"]),
        helper.make_node("ConstantOfShape", ["shape"], ["constant"]),
        helper.make_node("Add", ["input", "constant"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "constant_of_shape_test", [graph_input], [graph_output])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=8)


class TestFloat16ValueConversion(unittest.TestCase):
    def test_all_float16_bit_patterns(self):
        bits = np.arange(65536, dtype=np.uint16)
        values = bits.view(np.float16)
        self.assertEqual(_npfloat16_to_int(values), list(range(65536)))
        self.assertEqual(_npfloat16_to_int(values[::-7]), bits[::-7].tolist())
        self.assertEqual(_npfloat16_to_int([]), [])

    def test_tensor_storage_preserves_boundary_values(self):
        values = np.array(
            [0.0, -0.0, 1e-20, -1e-20, 1e10, -1e10, np.inf, -np.inf, np.nan, -np.nan, 1.5, -2.0],
            dtype=np.float32,
        ).reshape(3, 4)
        expected = np.array(
            [0x0000, 0x8000, 0x0001, 0x8001, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x3E00, 0xC000],
            dtype=np.uint16,
        ).reshape(3, 4)
        for raw in (False, True):
            with self.subTest(raw=raw):
                tensor = (
                    numpy_helper.from_array(values, "weights")
                    if raw
                    else helper.make_tensor("weights", TensorProto.FLOAT, values.shape, values.ravel())
                )
                tensor.doc_string = "Preserve tensor metadata"
                result = convert_tensor_float_to_float16(tensor)
                self.assertIs(result, tensor)
                self.assertEqual(result.name, "weights")
                self.assertEqual(list(result.dims), [3, 4])
                self.assertEqual(result.doc_string, "Preserve tensor metadata")
                self.assertEqual(result.data_type, TensorProto.FLOAT16)
                np.testing.assert_array_equal(numpy_helper.to_array(result).view(np.uint16), expected)
                self.assertEqual(len(result.float_data), 0)

    def test_debug_diagnostics_preserve_conversion(self):
        values = np.array([1e-20, -1e-20, 1e10, -1e10], dtype=np.float32)
        with self.assertNoLogs(convert_np_to_float16.__module__, level="WARNING"):
            quiet = convert_np_to_float16(values, min_positive_val=1e-4, max_finite_val=10000)
        with self.assertLogs(convert_np_to_float16.__module__, level="DEBUG") as captured:
            verbose = convert_np_to_float16(values, min_positive_val=1e-4, max_finite_val=10000)
        self.assertEqual(len(captured.records), 4)
        self.assertTrue(all("will be truncated" in record.getMessage() for record in captured.records))
        np.testing.assert_array_equal(quiet.view(np.uint16), verbose.view(np.uint16))
        np.testing.assert_array_equal(quiet, np.array([1e-4, -1e-4, 10000, -10000], dtype=np.float16))


class TestFloat16Conversion(unittest.TestCase):
    """Tests for convert_float_to_float16 correctness."""

    def _get_all_node_names(self, model):
        """Return all node names in the model graph."""
        return [n.name for n in model.graph.node]

    def _get_all_output_names(self, model):
        """Return all output tensor names from all nodes."""
        names = []
        for n in model.graph.node:
            names.extend(n.output)
        return names

    def _get_initializer(self, model, name):
        """Find an initializer by name."""
        for init in model.graph.initializer:
            if init.name == name:
                return init
        return None

    def test_resize_opset11_cast_naming_unique(self):
        """Multiple unnamed Resize nodes should produce uniquely named Cast nodes."""
        model = _make_resize_model_opset11(num_resize_nodes=3, use_empty_names=True)
        converted = convert_float_to_float16(model, keep_io_types=True)

        node_names = self._get_all_node_names(converted)
        # Filter to only non-empty names (original nodes may have empty names)
        cast_names = [n for n in node_names if n and "cast" in n.lower()]
        self.assertEqual(len(cast_names), len(set(cast_names)), f"Duplicate Cast node names found: {cast_names}")

        output_names = self._get_all_output_names(converted)
        cast_outputs = [n for n in output_names if "cast" in n.lower()]
        self.assertEqual(
            len(cast_outputs), len(set(cast_outputs)), f"Duplicate Cast output names found: {cast_outputs}"
        )

    def test_resize_opset11_shared_scales_reuses_cast(self):
        """A shared Resize scales input should use one Cast node for all consumers."""
        model = _make_resize_model_opset11(num_resize_nodes=2, use_empty_names=False, share_scales=True)
        converted = convert_float_to_float16(model, keep_io_types=True)

        scales_casts = [n for n in converted.graph.node if n.op_type == "Cast" and n.input[0] == "scales"]
        self.assertEqual(len(scales_casts), 1)

        resize_nodes = [n for n in converted.graph.node if n.op_type == "Resize"]
        self.assertEqual([n.input[2] for n in resize_nodes], [scales_casts[0].output[0]] * len(resize_nodes))

    def test_resize_opset11_scales_initializer_stays_fp32(self):
        """Resize scales initializer (input index 2) should stay float32 after conversion.

        When scales is an initializer and ALWAYS_FLOAT_INPUTS protects index 2,
        the initializer should not be converted to float16.
        Roi (index 1) is NOT protected for opset 11+ and may be converted to fp16.
        """
        model = _make_resize_model_opset11(num_resize_nodes=1, use_empty_names=False)
        converted = convert_float_to_float16(model, keep_io_types=True)

        # The scales initializer should remain float32 (not converted to fp16)
        scales_init = self._get_initializer(converted, "scales_0")
        self.assertIsNotNone(scales_init, "scales_0 initializer not found")
        self.assertEqual(
            scales_init.data_type,
            TensorProto.FLOAT,
            "Resize scales initializer should stay float32",
        )

        # Roi (index 1) is NOT protected for opset 11+ — the ONNX spec allows fp16 roi.
        # The initializer may be converted to fp16 (it is not in always_float_inputs).
        roi_init = self._get_initializer(converted, "roi_0")
        self.assertIsNotNone(roi_init, "roi_0 initializer not found")
        self.assertIn(
            roi_init.data_type,
            (TensorProto.FLOAT, TensorProto.FLOAT16),
            "Opset 11+ Resize roi is not protected — may be fp32 or fp16",
        )

    def test_resize_opset10_scales_initializer_stays_fp32(self):
        """Resize opset 10 scales initializer (input index 1) should stay float32.

        Before the fix, ALWAYS_FLOAT_INPUTS only protected index 2, so opset 10
        Resize (where scales is at index 1) would incorrectly convert scales to fp16.
        """
        model = _make_resize_model_opset10()
        converted = convert_float_to_float16(model, keep_io_types=True)

        # The scales initializer should remain float32
        scales_init = self._get_initializer(converted, "scales_0")
        self.assertIsNotNone(scales_init, "scales_0 initializer not found")
        self.assertEqual(
            scales_init.data_type,
            TensorProto.FLOAT,
            "Opset 10 Resize scales initializer should stay float32 (index 1 protected)",
        )

    def test_resize_opset10_multiple_unnamed_unique_names(self):
        """Multiple unnamed opset 10 Resize nodes should produce uniquely named Cast nodes."""
        model = _make_resize_model_opset10(num_resize_nodes=3, use_empty_names=True)
        converted = convert_float_to_float16(model, keep_io_types=True)

        node_names = self._get_all_node_names(converted)
        cast_names = [n for n in node_names if n and "cast" in n.lower()]
        self.assertEqual(len(cast_names), len(set(cast_names)), f"Duplicate Cast node names found: {cast_names}")

    def test_blocked_node_cast_naming_unique(self):
        """Multiple unnamed blocked-op nodes should produce uniquely named Cast nodes."""
        model = _make_blocked_node_model(num_nodes=2, use_empty_names=True)
        converted = convert_float_to_float16(model, keep_io_types=True)

        node_names = self._get_all_node_names(converted)
        cast_names = [n for n in node_names if n and "cast" in n.lower()]
        self.assertEqual(len(cast_names), len(set(cast_names)), f"Duplicate Cast node names found: {cast_names}")

        output_names = self._get_all_output_names(converted)
        cast_outputs = [n for n in output_names if "cast" in n.lower()]
        self.assertEqual(
            len(cast_outputs), len(set(cast_outputs)), f"Duplicate Cast output names found: {cast_outputs}"
        )

    def test_mixed_float_bfloat16_cast_naming_unique(self):
        """A shared input for FLOAT and BFLOAT16 consumers should get distinct Cast names."""
        model = _make_mixed_float_bfloat16_model()
        converted = convert_float_to_float16(
            model, keep_io_types=False, use_bfloat16_as_blocked_nodes_dtype=True, disable_shape_infer=True
        )

        shared_casts = [n for n in converted.graph.node if n.op_type == "Cast" and n.input[0] == "shared"]
        cast_types = {next(a.i for a in n.attribute if a.name == "to") for n in shared_casts}
        self.assertEqual(cast_types, {TensorProto.FLOAT, TensorProto.BFLOAT16})
        self.assertEqual(len({n.name for n in shared_casts}), 2)
        self.assertEqual(len({n.output[0] for n in shared_casts}), 2)

    def test_resize_with_op_block_list(self):
        """When Resize is in op_block_list, Cast nodes should have unique names."""
        model = _make_resize_model_opset11(num_resize_nodes=2, use_empty_names=True)
        converted = convert_float_to_float16(model, keep_io_types=True, op_block_list=["Resize"])

        # All Cast node names should be unique
        node_names = self._get_all_node_names(converted)
        cast_names = [n for n in node_names if n and "cast" in n.lower()]
        self.assertEqual(len(cast_names), len(set(cast_names)), f"Duplicate Cast node names found: {cast_names}")

    def test_data_input_converted_to_fp16(self):
        """Resize data input (index 0) should be converted to float16."""
        model = _make_resize_model_opset11(num_resize_nodes=1, use_empty_names=False)
        converted = convert_float_to_float16(model, keep_io_types=False)

        # Graph input should be float16
        graph_input = converted.graph.input[0]
        self.assertEqual(graph_input.type.tensor_type.elem_type, TensorProto.FLOAT16)

    def test_force_fp16_initializers(self):
        """With force_fp16_initializers=True, scales should be converted to fp16."""
        model = _make_resize_model_opset11(num_resize_nodes=1, use_empty_names=False)
        converted = convert_float_to_float16(model, keep_io_types=True, force_fp16_initializers=True)

        # With force_fp16_initializers, even protected initializers get converted
        # but Cast nodes are inserted to feed them back as fp32
        scales_init = self._get_initializer(converted, "scales_0")
        self.assertIsNotNone(scales_init)
        self.assertEqual(
            scales_init.data_type,
            TensorProto.FLOAT16,
            "With force_fp16_initializers, scales should be converted to fp16",
        )

    def test_constant_of_shape_without_value_gets_fp16_value(self):
        """ConstantOfShape without value should get an explicit float16 zero."""
        model = _make_constant_of_shape_model()
        converted = convert_float_to_float16(model, keep_io_types=True)

        constant_of_shape = next(node for node in converted.graph.node if node.op_type == "ConstantOfShape")
        value = next((attribute.t for attribute in constant_of_shape.attribute if attribute.name == "value"), None)
        self.assertIsNotNone(value)
        self.assertEqual(value.data_type, TensorProto.FLOAT16)
        np.testing.assert_array_equal(numpy_helper.to_array(value), np.array([0], dtype=np.float16))


class TestFloat16ModelPathConversion(unittest.TestCase):
    def _save_model(self, directory, external_data=False):
        graph = helper.make_graph(
            [
                helper.make_node("Mul", ["input", "weight"], ["product"]),
                helper.make_node("Identity", ["product"], ["output"]),
            ],
            "model_path_test",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])],
            [numpy_helper.from_array(np.array([0.5, 2.0], dtype=np.float32), "weight")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=8)
        model_path = directory / "model with spaces.onnx"
        onnx.save_model(
            model,
            model_path,
            save_as_external_data=external_data,
            all_tensors_to_one_file=True,
            location=f"{directory.name}.weights.bin",
            size_threshold=0,
        )
        return model_path

    @parameterized.expand([(False,), (True,)])
    def test_model_path_matches_model_proto(self, external_data):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            model_path = self._save_model(directory, external_data)
            original_files = {path.name: path.read_bytes() for path in directory.iterdir()}
            for keep_io_types in (False, True, ["input"]):
                with self.subTest(keep_io_types=keep_io_types):
                    expected = convert_float_to_float16(onnx.load(model_path), keep_io_types=keep_io_types)
                    for _ in range(2):
                        converted = convert_float_to_float16(str(model_path), keep_io_types=keep_io_types)
                        self.assertEqual(converted, expected)
                        self.assertEqual({path.name: path.read_bytes() for path in directory.iterdir()}, original_files)

    @parameterized.expand([("infer_shapes_path",), ("onnx.load",)])
    def test_model_path_removes_temporary_file_on_error(self, operation):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            model_path = self._save_model(directory)
            original_files = {path.name: path.read_bytes() for path in directory.iterdir()}
            error = RuntimeError("model path conversion failed")
            with (
                patch(f"{convert_float_to_float16.__module__}.{operation}", side_effect=error),
                self.assertRaises(RuntimeError) as raised,
            ):
                convert_float_to_float16(str(model_path))
            self.assertIs(raised.exception, error)
            self.assertEqual({path.name: path.read_bytes() for path in directory.iterdir()}, original_files)


if __name__ == "__main__":
    unittest.main()
