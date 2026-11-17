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
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.calibrate import TensorsData
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from onnxruntime.quantization.quant_utils import QuantizationMode


class TestMatMulMissingCalibration(unittest.TestCase):
    @staticmethod
    def _value(name):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, [2, 2])

    @staticmethod
    def _quantize(model, ranges, qtype, per_channel, subgraph=False):
        tensors = TensorsData(
            CalibrationMethod.MinMax,
            {name: (np.array(-4, dtype=np.float32), np.array(4, dtype=np.float32)) for name in ranges},
        )
        return ONNXQuantizer(
            model,
            per_channel,
            False,
            QuantizationMode.QLinearOps,
            True,
            qtype,
            qtype,
            tensors,
            [],
            [],
            ["MatMul"],
            {"EnableSubgraph": subgraph},
        ).quantize_model()

    def _check_outputs(self, model, quantized, feeds, expected):
        onnx.checker.check_model(quantized, full_check=True)
        for optimization in (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
            options = ort.SessionOptions()
            options.graph_optimization_level = optimization
            options.intra_op_num_threads = 1
            original = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
            actual = ort.InferenceSession(quantized.SerializeToString(), options, providers=["CPUExecutionProvider"])
            for feed, answer in zip(feeds, expected, strict=True):
                np.testing.assert_allclose(original.run(None, feed)[0], answer, atol=1e-6, rtol=1e-6)
                np.testing.assert_allclose(actual.run(None, feed)[0], answer, atol=0.06, rtol=0.02)

    def test_partial_calibration_ranges(self):
        lhs = np.array([[0.25, -0.5], [1, -1.5]], dtype=np.float32)
        rhs = np.array([[1, 0.5], [-0.5, 0.75]], dtype=np.float32)
        for constant_rhs, before, qtype, per_channel in itertools.product(
            (False, True), (False, True), (QuantType.QInt8, QuantType.QUInt8), (False, True)
        ):
            for missing in itertools.product((False, True), repeat=3):
                if before and missing[0]:
                    continue
                with self.subTest(
                    constant_rhs=constant_rhs, before=before, qtype=qtype, per_channel=per_channel, missing=missing
                ):
                    inputs = [self._value("x" if before else "a")]
                    initializers = []
                    nodes = []
                    if before:
                        initializers.append(numpy_helper.from_array(np.eye(2, dtype=np.float32), "identity"))
                        nodes.append(helper.make_node("MatMul", ["x", "identity"], ["a"], name="before"))
                    if constant_rhs:
                        initializers.append(numpy_helper.from_array(rhs, "b"))
                    else:
                        inputs.append(self._value("b"))
                    nodes.append(helper.make_node("MatMul", ["a", "b"], ["y"], name="target"))
                    graph = helper.make_graph(nodes, "partial_ranges", inputs, [self._value("y")], initializers)
                    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=7)
                    ranges = {name for name, absent in zip(("a", "b", "y"), missing, strict=True) if not absent}
                    if before:
                        ranges.add("x")
                    quantized = self._quantize(model, ranges, qtype, per_channel)
                    quantized_target = not missing[0] and not missing[2] and (constant_rhs or not missing[1])
                    self.assertEqual(
                        sum(node.op_type == "QLinearMatMul" for node in quantized.graph.node),
                        int(before) + int(quantized_target),
                    )
                    feeds, answers = [], []
                    for index in range(2):
                        a = lhs if index == 0 else -lhs
                        b = rhs if constant_rhs or index == 0 else rhs.T.copy()
                        feed = {"x" if before else "a": a}
                        if not constant_rhs:
                            feed["b"] = b
                        feeds.append(feed)
                        answers.append(a @ b)
                    self._check_outputs(model, quantized, feeds, answers)

    def test_contrib_outputs_without_value_info(self):
        lhs = np.array([[0.25, -0.5], [1, -1.5]], dtype=np.float32)
        rhs = np.array([[1, 0.5], [-0.5, 0.75]], dtype=np.float32)
        for source, annotated, qtype, per_channel in itertools.product(
            ("a", "b", "both"), (False, True), (QuantType.QInt8, QuantType.QUInt8), (False, True)
        ):
            with self.subTest(source=source, annotated=annotated, qtype=qtype, per_channel=per_channel):
                nodes, inputs, value_info = [], [], []
                for name in ("a", "b"):
                    if source in (name, "both"):
                        inputs.append(self._value("raw_" + name))
                        nodes.append(
                            helper.make_node("FusedMatMul", ["raw_" + name, "identity"], [name], domain="com.microsoft")
                        )
                        if annotated:
                            value_info.append(self._value(name))
                    else:
                        inputs.append(self._value(name))
                nodes.append(helper.make_node("MatMul", ["a", "b"], ["y"], name="target"))
                graph = helper.make_graph(
                    nodes,
                    "contrib_ranges",
                    inputs,
                    [self._value("y")],
                    [numpy_helper.from_array(np.eye(2, dtype=np.float32), "identity")],
                    value_info=value_info,
                )
                model = helper.make_model(
                    graph,
                    opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("com.microsoft", 1)],
                    ir_version=7,
                )
                feeds, answers = [], []
                for index in range(2):
                    a, b = (lhs, rhs) if index == 0 else (-lhs, rhs.T.copy())
                    feeds.append(
                        {
                            ("raw_" + name if source in (name, "both") else name): value
                            for name, value in (("a", a), ("b", b))
                        }
                    )
                    answers.append(a @ b)
                with tempfile.TemporaryDirectory() as directory:
                    output_path = Path(directory) / "quantized.onnx"
                    quantize_static(
                        model,
                        output_path,
                        TestDataFeeds(feeds),
                        quant_format=QuantFormat.QOperator,
                        activation_type=qtype,
                        weight_type=qtype,
                        per_channel=per_channel,
                        op_types_to_quantize=["MatMul"],
                    )
                    quantized = onnx.load(output_path)
                self.assertEqual(sum(node.op_type == "QLinearMatMul" for node in quantized.graph.node), int(annotated))
                self._check_outputs(model, quantized, feeds, answers)

    def test_missing_output_does_not_poison_shared_inputs(self):
        lhs = np.array([[0.25, -0.5], [1, -1.5]], dtype=np.float32)
        rhs = np.array([[1, 0.5], [-0.5, 0.75]], dtype=np.float32)
        for constant_rhs, qtype, per_channel in itertools.product(
            (False, True), (QuantType.QInt8, QuantType.QUInt8), (False, True)
        ):
            with self.subTest(constant_rhs=constant_rhs, qtype=qtype, per_channel=per_channel):
                graph = helper.make_graph(
                    [
                        helper.make_node("MatMul", ["a", "b"], ["first"], name="fallback"),
                        helper.make_node("MatMul", ["a", "b"], ["second"], name="quantized"),
                        helper.make_node("Add", ["first", "second"], ["y"]),
                    ],
                    "shared_inputs",
                    [self._value("a")] + ([] if constant_rhs else [self._value("b")]),
                    [self._value("y")],
                    [numpy_helper.from_array(rhs, "b")] if constant_rhs else [],
                )
                model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=7)
                quantized = self._quantize(model, {"a", "b", "second", "y"}, qtype, per_channel)
                self.assertEqual(sum(node.op_type == "QLinearMatMul" for node in quantized.graph.node), 1)
                feeds = [{"a": value, **({} if constant_rhs else {"b": rhs})} for value in (lhs, -lhs)]
                self._check_outputs(model, quantized, feeds, [2 * lhs @ rhs, -2 * lhs @ rhs])

    def test_captured_inputs_and_initializers(self):
        lhs = np.array([[0.25, -0.5], [1, -1.5]], dtype=np.float32)
        rhs = np.array([[1, 0.5], [-0.5, 0.75]], dtype=np.float32)
        for missing, qtype, per_channel in itertools.product(
            ("none", "input", "output"), (QuantType.QInt8, QuantType.QUInt8), (False, True)
        ):
            with self.subTest(missing=missing, qtype=qtype, per_channel=per_channel):
                branches = [
                    helper.make_graph([helper.make_node("MatMul", ["a", "b"], [name])], name, [], [self._value(name)])
                    for name in ("then_y", "else_y")
                ]
                graph = helper.make_graph(
                    [helper.make_node("If", ["condition"], ["y"], then_branch=branches[0], else_branch=branches[1])],
                    "captured_ranges",
                    [self._value("a"), helper.make_tensor_value_info("condition", TensorProto.BOOL, [])],
                    [self._value("y")],
                    [numpy_helper.from_array(rhs, "b")],
                )
                model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=7)
                ranges = {"y"}
                if missing != "input":
                    ranges.add("a")
                if missing != "output":
                    ranges.update(("then_y", "else_y"))
                quantized = self._quantize(model, ranges, qtype, per_channel, subgraph=True)
                if_node = next(node for node in quantized.graph.node if node.op_type == "If")
                self.assertEqual(
                    sum(node.op_type == "QLinearMatMul" for attr in if_node.attribute for node in attr.g.node),
                    2 if missing == "none" else 0,
                )
                feeds = [{"a": lhs, "condition": np.array(value)} for value in (True, False)]
                self._check_outputs(model, quantized, feeds, [lhs @ rhs] * 2)

    def test_initializer_inputs_do_not_require_ranges(self):
        lhs = np.array([[0.25, -0.5], [1, -1.5]], dtype=np.float32)
        rhs = np.array([[1, 0.5], [-0.5, 0.75]], dtype=np.float32)
        for qtype, per_channel in itertools.product((QuantType.QInt8, QuantType.QUInt8), (False, True)):
            with self.subTest(qtype=qtype, per_channel=per_channel):
                graph = helper.make_graph(
                    [helper.make_node("MatMul", ["a", "b"], ["y"])],
                    "constant_inputs",
                    [],
                    [self._value("y")],
                    [numpy_helper.from_array(lhs, "a"), numpy_helper.from_array(rhs, "b")],
                )
                model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=7)
                quantized = self._quantize(model, {"y"}, qtype, per_channel)
                self.assertEqual(sum(node.op_type == "QLinearMatMul" for node in quantized.graph.node), 1)
                self._check_outputs(model, quantized, [{}, {}], [lhs @ rhs] * 2)


if __name__ == "__main__":
    unittest.main()
