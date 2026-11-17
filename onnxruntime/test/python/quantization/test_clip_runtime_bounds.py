#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
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


class TestClipRuntimeBounds(unittest.TestCase):
    def _check(self, source, bounds, quant_format, symmetric, following=False, calibrate_narrow=False):
        dynamic = source in ("runtime", "overridable", "computed")
        data = np.array([-2, -1, 1, 2], dtype=np.float32).reshape(1, 1, 2, 2)
        inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, data.shape)]
        initializers = [numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), "weight")]
        nodes = [helper.make_node("Conv", ["x", "weight"], ["before"], name="before_conv")]
        bound_names = []
        defaults = {"min": -0.5, "max": 0.5} if calibrate_narrow or not dynamic else {"min": -2.0, "max": 2.0}
        for name in ("min", "max"):
            if name not in bounds:
                bound_names.append("")
                continue
            bound_names.append(name)
            value = np.array(defaults[name], dtype=np.float32)
            if source in ("runtime", "overridable", "computed"):
                inputs.append(helper.make_tensor_value_info(name + "_input", TensorProto.FLOAT, []))
                bound_names[-1] = name + "_input"
            if source in ("initializer", "overridable"):
                initializers.append(numpy_helper.from_array(value, bound_names[-1]))
            elif source == "constant":
                nodes.append(helper.make_node("Constant", [], [name], value=numpy_helper.from_array(value)))
            elif source == "computed":
                initializers.append(numpy_helper.from_array(np.array(0, dtype=np.float32), name + "_zero"))
                nodes.append(helper.make_node("Add", [name + "_input", name + "_zero"], [name]))
                bound_names[-1] = name
        clip_inputs = ["before", *bound_names]
        attrs = {}
        if source == "attributes":
            clip_inputs = ["before"]
            attrs = {name: defaults[name] for name in bounds}
        nodes.append(helper.make_node("Clip", clip_inputs, ["clipped"], name="clip", **attrs))
        output_name = "clipped"
        if following:
            initializers.append(numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), "after_weight"))
            nodes.append(helper.make_node("Conv", [output_name, "after_weight"], ["y"], name="after_conv"))
            output_name = "y"
        graph = helper.make_graph(
            nodes,
            "clip_bounds",
            inputs,
            [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, data.shape)],
            initializers,
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 10 if source == "attributes" else 13)], ir_version=7
        )

        def feed(lower, upper):
            values = {"x": data}
            if dynamic:
                for name, value in (("min", lower), ("max", upper)):
                    if name in bounds:
                        values[name + "_input"] = np.array(value, dtype=np.float32)
            return values

        calibration_feed = feed(defaults["min"], defaults["max"])
        options = {"ActivationSymmetric": symmetric}
        if calibrate_narrow:
            options["OpTypesToExcludeOutputQuantization"] = ["Clip"]
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "quantized.onnx"
            quantize_static(
                model,
                output_path,
                TestDataFeeds([calibration_feed]),
                quant_format=quant_format,
                activation_type=QuantType.QInt8 if symmetric else QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                op_types_to_quantize=["Conv", "Clip"],
                extra_options=options,
            )
            quantized = onnx.load(output_path)
        onnx.checker.check_model(quantized, full_check=True)
        self.assertEqual(sum(node.op_type == "Clip" for node in quantized.graph.node), int(dynamic or symmetric))
        if dynamic:
            self.assertEqual(
                {value.name for value in quantized.graph.input}, {value.name for value in model.graph.input}
            )
        if quant_format == QuantFormat.QOperator:
            self.assertEqual(sum(node.op_type == "QLinearConv" for node in quantized.graph.node), 1 + int(following))
        else:
            self.assertEqual(sum(node.op_type == "Conv" for node in quantized.graph.node), 1 + int(following))
            self.assertTrue(any(node.op_type == "DequantizeLinear" for node in quantized.graph.node))

        for optimization in (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = optimization
            session_options.intra_op_num_threads = 1
            original = ort.InferenceSession(
                model.SerializeToString(), session_options, providers=["CPUExecutionProvider"]
            )
            actual = ort.InferenceSession(
                quantized.SerializeToString(), session_options, providers=["CPUExecutionProvider"]
            )
            for runtime_bounds in ((-0.5, 0.5), (-1.5, 1.5), (-0.25, 1.25)):
                lower, upper = runtime_bounds if dynamic else (defaults["min"], defaults["max"])
                feeds = feed(lower, upper)
                expected = np.clip(data, lower if "min" in bounds else -np.inf, upper if "max" in bounds else np.inf)
                np.testing.assert_allclose(original.run(None, feeds)[0], expected, rtol=0, atol=1e-6)
                np.testing.assert_allclose(actual.run(None, feeds)[0], expected, rtol=0, atol=0.025)
            if source == "overridable":
                expected = np.clip(
                    data,
                    defaults["min"] if "min" in bounds else -np.inf,
                    defaults["max"] if "max" in bounds else np.inf,
                )
                np.testing.assert_allclose(actual.run(None, {"x": data})[0], expected, rtol=0, atol=0.025)

    def test_runtime_clip_bounds(self):
        for args in itertools.product(
            ("runtime", "overridable", "computed"),
            (("min",), ("max",), ("min", "max")),
            (QuantFormat.QOperator, QuantFormat.QDQ),
            (False, True),
            (False, True),
        ):
            with self.subTest(args=args):
                self._check(*args)

    def test_constant_clip_bounds(self):
        for args in itertools.product(
            ("initializer", "constant", "attributes"),
            (("min",), ("max",), ("min", "max")),
            (QuantFormat.QOperator, QuantFormat.QDQ),
            (False, True),
        ):
            with self.subTest(args=args):
                self._check(*args)

    def test_omitted_clip_bounds(self):
        for quant_format, symmetric in itertools.product((QuantFormat.QOperator, QuantFormat.QDQ), (False, True)):
            with self.subTest(quant_format=quant_format, symmetric=symmetric):
                self._check("initializer", (), quant_format, symmetric)

    def test_runtime_bounds_keep_input_range(self):
        for source, quant_format, symmetric in itertools.product(
            ("runtime", "overridable", "computed"), (QuantFormat.QOperator, QuantFormat.QDQ), (False, True)
        ):
            with self.subTest(source=source, quant_format=quant_format, symmetric=symmetric):
                self._check(source, ("min", "max"), quant_format, symmetric, calibrate_narrow=True)


if __name__ == "__main__":
    unittest.main()
