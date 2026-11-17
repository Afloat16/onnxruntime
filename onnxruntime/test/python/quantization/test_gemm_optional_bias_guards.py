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
import test_gemm_optional_bias as optional
from onnx import TensorProto, helper, numpy_helper
from op_test_utils import TestDataFeeds

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static


class TestGemmOptionalBiasGuards(unittest.TestCase):
    @staticmethod
    def _quantize(model, feeds, per_channel=True, float8=False, extra_options=None):
        onnx.checker.check_model(model, full_check=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "quantized.onnx"
            quantize_static(
                model,
                output,
                TestDataFeeds([feeds]),
                quant_format=QuantFormat.QOperator,
                activation_type=QuantType.QFLOAT8E4M3FN if float8 else QuantType.QUInt8,
                weight_type=QuantType.QFLOAT8E4M3FN if float8 else QuantType.QInt8,
                per_channel=per_channel,
                calibrate_method=CalibrationMethod.Distribution if float8 else CalibrationMethod.MinMax,
                extra_options=extra_options,
            )
            result = onnx.load(output)
        onnx.checker.check_model(result, full_check=True)
        return result

    @staticmethod
    def _run(model, feeds, level):
        options = SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = level
        return InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"]).run(None, feeds)

    def _check_results(self, model, quantized, feeds):
        reference = self._run(model, feeds, GraphOptimizationLevel.ORT_DISABLE_ALL)
        for level in (GraphOptimizationLevel.ORT_DISABLE_ALL, GraphOptimizationLevel.ORT_ENABLE_ALL):
            actual = self._run(quantized, feeds, level)
            for expected, observed in zip(reference, actual, strict=True):
                self.assertEqual(expected.dtype, observed.dtype)
                np.testing.assert_allclose(observed, expected, rtol=0.04, atol=0.06)

    def test_negative_transpose(self):
        feeds = {"input": np.array([[1.0, 0.5, -0.25], [-0.5, 0.25, 1.0]], np.float32)}
        for trans_b, opset, per_channel in itertools.product((-1, -2), (13, 21), (False, True)):
            with self.subTest(trans_b=trans_b, opset=opset, per_channel=per_channel):
                omitted = optional.TestGemmOptionalBias._make_model(opset, trans_b)
                explicit = copy.deepcopy(omitted)
                explicit.graph.node[0].input.append("")
                first = self._quantize(omitted, feeds, per_channel)
                second = self._quantize(explicit, feeds, per_channel)
                self.assertEqual(first.SerializeToString(), second.SerializeToString())
                self._check_results(explicit, second, feeds)

    def test_unsupported_qgemm_types_keep_float_fallback(self):
        for kind, per_channel, explicit in itertools.product(("float16", "float8"), (False, True), (False, True)):
            if kind == "float8" and per_channel:
                continue
            with self.subTest(kind=kind, per_channel=per_channel, explicit=explicit):
                model = optional.TestGemmOptionalBias._make_model(21, 0)
                if explicit:
                    model.graph.node[0].input.append("")
                feeds = {"input": np.array([[1.0, 0.5, -0.25], [-0.5, 0.25, 1.0]], np.float32)}
                if kind == "float16":
                    for initializer in model.graph.initializer:
                        initializer.CopyFrom(
                            numpy_helper.from_array(
                                numpy_helper.to_array(initializer).astype(np.float16), initializer.name
                            )
                        )
                    for value in (*model.graph.input, *model.graph.output):
                        value.type.tensor_type.elem_type = TensorProto.FLOAT16
                    feeds["input"] = feeds["input"].astype(np.float16)
                quantized = self._quantize(model, feeds, per_channel, float8=kind == "float8")
                self.assertEqual(sum(node.op_type == "Gemm" for node in quantized.graph.node), 1)
                self.assertFalse(any(node.op_type == "QGemm" for node in quantized.graph.node))
                self._check_results(model, quantized, feeds)

    def test_effective_override_types_keep_float_fallback(self):
        feeds = {"input": np.array([[1.0, 0.5, -0.25], [-0.5, 0.25, 1.0]], np.float32)}
        for tensor, quant_type, explicit in itertools.product(
            ("input", "weight", "output"), (QuantType.QInt16, QuantType.QUInt16), (False, True)
        ):
            with self.subTest(tensor=tensor, quant_type=quant_type, explicit=explicit):
                model = optional.TestGemmOptionalBias._make_model(21, 0)
                if explicit:
                    model.graph.node[0].input.append("")
                quantized = self._quantize(
                    model,
                    feeds,
                    per_channel=False,
                    extra_options={"TensorQuantOverrides": {tensor: [{"quant_type": quant_type}]}},
                )
                self.assertFalse(any(node.op_type == "QGemm" for node in quantized.graph.node))
                self._check_results(model, quantized, feeds)

    def test_effective_signedness_combinations(self):
        feeds = {"input": np.array([[1.0, 0.5, -0.25], [-0.5, 0.25, 1.0]], np.float32)}
        for input_type, weight_type, output_type, explicit in itertools.product(
            (QuantType.QInt8, QuantType.QUInt8),
            (QuantType.QInt8, QuantType.QUInt8),
            (QuantType.QInt8, QuantType.QUInt8),
            (False, True),
        ):
            with self.subTest(
                input_type=input_type, weight_type=weight_type, output_type=output_type, explicit=explicit
            ):
                model = optional.TestGemmOptionalBias._make_model(21, 0)
                if explicit:
                    model.graph.node[0].input.append("")
                overrides = {
                    "input": [{"quant_type": input_type}],
                    "weight": [{"quant_type": weight_type}],
                    "output": [{"quant_type": output_type}],
                }
                quantized = self._quantize(
                    model, feeds, per_channel=False, extra_options={"TensorQuantOverrides": overrides}
                )
                self._check_results(model, quantized, feeds)
                supported = input_type == output_type and not (
                    input_type == QuantType.QInt8 and weight_type == QuantType.QUInt8
                )
                self.assertEqual(sum(node.op_type == "QGemm" for node in quantized.graph.node), int(supported))

    def _runtime_model(self, trans_a, trans_b, explicit, known_k):
        k = 3 if known_k else "K"
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "Gemm",
                        ["input", "weight", ""] if explicit else ["input", "weight"],
                        ["output"],
                        transA=trans_a,
                        transB=trans_b,
                    )
                ],
                "runtime_inner_dimension",
                [
                    helper.make_tensor_value_info("input", TensorProto.FLOAT, [k, 2] if trans_a else [2, k]),
                    helper.make_tensor_value_info("weight", TensorProto.FLOAT, [4, k] if trans_b else [k, 4]),
                ],
                [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])],
            ),
            opset_imports=[helper.make_opsetid("", 21)],
            ir_version=10,
        )
        a = np.array([[1.0, 0.5, -0.25], [-0.5, 0.25, 1.0]], np.float32)
        b = np.arange(12, dtype=np.float32).reshape(3, 4) / 10
        feeds = {"input": a.T if trans_a else a, "weight": b.T if trans_b else b}
        return model, feeds

    def test_runtime_zero_inner_dimension(self):
        for trans_a, trans_b, explicit in itertools.product((0, 1), (0, 1), (False, True)):
            with self.subTest(trans_a=trans_a, trans_b=trans_b, explicit=explicit):
                model, feeds = self._runtime_model(trans_a, trans_b, explicit, known_k=False)
                quantized = self._quantize(model, feeds)
                self.assertFalse(any(node.op_type == "QGemm" for node in quantized.graph.node))
                self._check_results(model, quantized, feeds)
                empty = {
                    "input": np.empty((0, 2) if trans_a else (2, 0), np.float32),
                    "weight": np.empty((4, 0) if trans_b else (0, 4), np.float32),
                }
                self._check_results(model, quantized, empty)
                np.testing.assert_array_equal(
                    self._run(quantized, empty, GraphOptimizationLevel.ORT_DISABLE_ALL)[0], np.zeros((2, 4), np.float32)
                )

    def test_known_positive_runtime_inner_dimension(self):
        for trans_a, trans_b, explicit in itertools.product((0, 1), (0, 1), (False, True)):
            with self.subTest(trans_a=trans_a, trans_b=trans_b, explicit=explicit):
                model, feeds = self._runtime_model(trans_a, trans_b, explicit, known_k=True)
                quantized = self._quantize(model, feeds)
                self.assertEqual(sum(node.op_type == "QGemm" for node in quantized.graph.node), 1)
                self._check_results(model, quantized, feeds)

    def test_overridable_initializer_does_not_fix_runtime_k(self):
        for trans_a, trans_b, explicit in itertools.product((0, 1), (0, 1), (False, True)):
            with self.subTest(trans_a=trans_a, trans_b=trans_b, explicit=explicit):
                model, feeds = self._runtime_model(trans_a, trans_b, explicit, known_k=False)
                model.graph.initializer.append(numpy_helper.from_array(feeds["weight"], "weight"))
                quantized = self._quantize(model, feeds)
                self.assertFalse(any(node.op_type == "QGemm" for node in quantized.graph.node))
                self._check_results(model, quantized, feeds)
                empty = {
                    "input": np.empty((0, 2) if trans_a else (2, 0), np.float32),
                    "weight": np.empty((4, 0) if trans_b else (0, 4), np.float32),
                }
                self._check_results(model, quantized, empty)

    def test_shared_weight_with_different_axes(self):
        for rows, reverse, opset, explicit in itertools.product((2, 3), (False, True), (13, 21), (False, True)):
            with self.subTest(rows=rows, reverse=reverse, opset=opset, explicit=explicit):
                weight = np.arange(1, rows * 3 + 1, dtype=np.float32).reshape(rows, 3)
                nodes = [
                    helper.make_node("Gemm", ["input", "weight"], ["output"], name="first", transB=0),
                    helper.make_node("Gemm", ["input2", "weight"], ["output2"], name="second", transB=1),
                ]
                if explicit:
                    for node in nodes:
                        node.input.append("")
                if reverse:
                    nodes.reverse()
                model = helper.make_model(
                    helper.make_graph(
                        nodes,
                        "shared_weight_axes",
                        [
                            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, rows]),
                            helper.make_tensor_value_info("input2", TensorProto.FLOAT, [1, 3]),
                        ],
                        [
                            helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3]),
                            helper.make_tensor_value_info("output2", TensorProto.FLOAT, [1, rows]),
                        ],
                        [numpy_helper.from_array(weight, "weight")],
                    ),
                    opset_imports=[helper.make_opsetid("", opset)],
                    ir_version=10,
                )
                feeds = {
                    "input": np.linspace(-0.25, 0.5, rows, dtype=np.float32).reshape(1, rows),
                    "input2": np.array([[1.0, -0.25, 0.5]], np.float32),
                }
                quantized = self._quantize(model, feeds)
                self.assertEqual(sum(node.op_type == "QGemm" for node in quantized.graph.node), 1)
                self.assertEqual(sum(node.op_type == "Gemm" for node in quantized.graph.node), 1)
                self._check_results(model, quantized, feeds)


if __name__ == "__main__":
    unittest.main()
