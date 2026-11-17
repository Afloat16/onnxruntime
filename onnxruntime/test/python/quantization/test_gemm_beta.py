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
from onnx import TensorProto, helper, numpy_helper
from op_test_utils import TestDataFeeds

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.base_quantizer import BaseQuantizer


class TestGemmBeta(unittest.TestCase):
    @staticmethod
    def _model(beta, dtype=np.float32, trans_b=False, magnitude=0.5, bias_magnitude=1.0):
        weights = np.array([[magnitude, magnitude / 2], [-magnitude / 2, magnitude]], dtype=dtype)
        bias = np.array([bias_magnitude, -2 * bias_magnitude], dtype=dtype)
        elem_type = TensorProto.FLOAT if dtype == np.float32 else TensorProto.FLOAT16
        return helper.make_model(
            helper.make_graph(
                [helper.make_node("Gemm", ["input", "weight", "bias"], ["output"], beta=beta, transB=int(trans_b))],
                "gemm_beta",
                [helper.make_tensor_value_info("input", elem_type, [2, 2])],
                [helper.make_tensor_value_info("output", elem_type, [2, 2])],
                [
                    numpy_helper.from_array(weights.T if trans_b else weights, "weight"),
                    numpy_helper.from_array(bias, "bias"),
                ],
            ),
            opset_imports=[helper.make_opsetid("", 21)],
            ir_version=10,
        )

    @staticmethod
    def _run(model, feeds, level):
        options = SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = level
        return InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"]).run(None, feeds)

    @staticmethod
    def _quantized_bias(model):
        tensors = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        gemm = next(node for node in model.graph.node if node.op_type in {"Gemm", "QGemm"})
        if gemm.op_type == "QGemm":
            return tensors[gemm.input[6]], np.atleast_1d(tensors[gemm.input[1]] * tensors[gemm.input[4]])
        producers = {output: node for node in model.graph.node for output in node.output}
        dequant = producers[gemm.input[2]]
        if dequant.op_type == "Cast":
            dequant = producers[dequant.input[0]]
        return tensors[dequant.input[0]], tensors[dequant.input[1]]

    def _check_quantized(self, model, feeds, quant_format, per_channel, extra_options=None):
        onnx.checker.check_model(model, full_check=True)
        expected = self._run(model, feeds, GraphOptimizationLevel.ORT_DISABLE_ALL)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quantized.onnx"
            quantize_static(
                model,
                path,
                TestDataFeeds([feeds]),
                quant_format=quant_format,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=per_channel,
                extra_options=extra_options,
            )
            quantized = onnx.load(path)
        onnx.checker.check_model(quantized, full_check=True)
        for level in (GraphOptimizationLevel.ORT_DISABLE_ALL, GraphOptimizationLevel.ORT_ENABLE_ALL):
            actual = self._run(quantized, feeds, level)
            for actual_output, expected_output in zip(actual, expected, strict=True):
                np.testing.assert_allclose(actual_output, expected_output, rtol=0.02, atol=0.03)
        return quantized

    def test_nonunit_beta(self):
        feeds = {"input": np.eye(2, dtype=np.float32)}
        for beta, quant_format, per_channel, trans_b in itertools.product(
            (2.0, 0.5, -2.0, -0.5, 0.0, 1.0),
            (QuantFormat.QDQ, QuantFormat.QOperator),
            (False, True),
            (False, True),
        ):
            with self.subTest(beta=beta, format=quant_format, per_channel=per_channel, trans_b=trans_b):
                model = self._model(beta, trans_b=trans_b)
                quantized = self._check_quantized(model, feeds, quant_format, per_channel)
                tensors = {init.name: numpy_helper.to_array(init) for init in quantized.graph.initializer}
                quantized_bias, bias_scale = self._quantized_bias(quantized)
                self.assertTrue(np.all(bias_scale > 0))
                np.testing.assert_array_equal(
                    bias_scale, np.atleast_1d(tensors["input_scale"] * tensors["weight_scale"])
                )
                np.testing.assert_allclose(
                    quantized_bias.astype(np.float64) * bias_scale,
                    np.array([1.0, -2.0]) * beta,
                    rtol=1e-5,
                    atol=1e-5,
                )

    def test_scaled_bias_overflow_adjustment(self):
        feeds = {"input": np.eye(2, dtype=np.float32)}
        for beta, quant_format, per_channel in itertools.product(
            (4.0, -4.0), (QuantFormat.QDQ, QuantFormat.QOperator), (False, True)
        ):
            with self.subTest(beta=beta, format=quant_format, per_channel=per_channel):
                model = self._model(beta, magnitude=0.0001, bias_magnitude=10.0)
                quantized = self._check_quantized(model, feeds, quant_format, per_channel)
                effective_bias = np.array([10.0, -20.0]) * beta
                quantized_bias, bias_scale = self._quantized_bias(quantized)
                bias_scale = bias_scale.astype(np.float64)
                self.assertTrue(np.all(bias_scale * np.iinfo(np.int32).max >= np.abs(effective_bias)))
                np.testing.assert_allclose(quantized_bias.astype(np.float64) * bias_scale, effective_bias, rtol=1e-6)

    def test_nonunit_alpha_and_beta(self):
        feeds = {"input": np.eye(2, dtype=np.float32)}
        for alpha, beta, quant_format, per_channel in itertools.product(
            (0.0, 0.5, 2.0, -1.0),
            (2.0, -0.5, 0.0),
            (QuantFormat.QDQ, QuantFormat.QOperator),
            (False, True),
        ):
            with self.subTest(alpha=alpha, beta=beta, format=quant_format, per_channel=per_channel):
                model = self._model(beta)
                model.graph.node[0].attribute.append(helper.make_attribute("alpha", alpha))
                quantized = self._check_quantized(model, feeds, quant_format, per_channel)
                if quant_format == QuantFormat.QOperator and beta != 0.0:
                    # QGemm applies alpha to its bias accumulator as well as A @ B.
                    self.assertTrue(any(node.op_type == "Gemm" for node in quantized.graph.node))

    def test_float16_scaled_bias(self):
        feeds = {"input": np.eye(2, dtype=np.float16)}
        for beta, magnitude, bias_magnitude, per_channel in itertools.product(
            (2.0, -2.0, 0.5), (0.005, 2**-11), (1.0, 150.0), (False, True)
        ):
            with self.subTest(beta=beta, magnitude=magnitude, bias=bias_magnitude, per_channel=per_channel):
                model = self._model(beta, np.float16, magnitude=magnitude, bias_magnitude=bias_magnitude)
                quantized = self._check_quantized(model, feeds, QuantFormat.QDQ, per_channel)
                effective_bias = np.array([bias_magnitude, -2 * bias_magnitude]) * beta
                quantized_bias, bias_scale = self._quantized_bias(quantized)
                bias_scale = bias_scale.astype(np.float64)
                self.assertTrue(np.all(bias_scale > 0))
                np.testing.assert_allclose(quantized_bias.astype(np.float64) * bias_scale, effective_bias, rtol=1e-5)

    def test_shared_bias_outputs(self):
        feeds = {"input": np.eye(2, dtype=np.float32)}
        for beta, quant_format, per_channel in itertools.product(
            (2.0, -2.0, 0.0), (QuantFormat.QDQ, QuantFormat.QOperator), (False, True)
        ):
            with self.subTest(beta=beta, format=quant_format, per_channel=per_channel):
                model = self._model(beta)
                model.graph.node.append(helper.make_node("Identity", ["bias"], ["original_bias"]))
                model.graph.output.extend(
                    [
                        helper.make_tensor_value_info("bias", TensorProto.FLOAT, [2]),
                        helper.make_tensor_value_info("original_bias", TensorProto.FLOAT, [2]),
                    ]
                )
                quantized = self._check_quantized(model, feeds, quant_format, per_channel)
                source_bias = next(init for init in quantized.graph.initializer if init.name == "bias")
                np.testing.assert_array_equal(numpy_helper.to_array(source_bias), np.array([1.0, -2.0], np.float32))

    def test_shared_bias_between_gemms(self):
        feeds = {"input": np.eye(2, dtype=np.float32)}
        for betas, quant_format, per_channel in itertools.product(
            ((2.0, 1.0), (1.0, 2.0), (2.0, -0.5), (-0.5, 2.0), (0.0, 2.0), (2.0, 0.0)),
            (QuantFormat.QDQ, QuantFormat.QOperator),
            (False, True),
        ):
            with self.subTest(betas=betas, format=quant_format, per_channel=per_channel):
                model = self._model(betas[0])
                model.graph.node[0].name = "first"
                model.graph.node.append(
                    helper.make_node(
                        "Gemm", ["input", "weight", "bias"], ["second_output"], name="second", beta=betas[1]
                    )
                )
                model.graph.output.append(helper.make_tensor_value_info("second_output", TensorProto.FLOAT, [2, 2]))
                self._check_quantized(model, feeds, quant_format, per_channel)

    def test_outer_scope_initializers_keep_subgraph_fallback(self):
        for beta, captured, per_channel in itertools.product(
            (2.0, -2.0, 0.0), (("bias",), ("weight",), ("weight", "bias")), (False, True)
        ):
            with self.subTest(beta=beta, captured=captured, per_channel=per_channel):
                model = self._model(beta)
                source_initializers = list(model.graph.initializer)
                branches = {}
                for branch_name in ("then_branch", "else_branch"):
                    node = copy.deepcopy(model.graph.node[0])
                    node.output[0] = branch_name + "_output"
                    local_initializers = []
                    for index, source in enumerate(source_initializers, 1):
                        if source.name not in captured:
                            initializer = copy.deepcopy(source)
                            initializer.name = branch_name + "_" + source.name
                            node.input[index] = initializer.name
                            local_initializers.append(initializer)
                    branches[branch_name] = helper.make_graph(
                        [node],
                        branch_name,
                        [],
                        [helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, [2, 2])],
                        local_initializers,
                    )
                model.graph.ClearField("node")
                model.graph.node.extend(
                    [
                        helper.make_node("MatMul", ["input", "identity"], ["control"]),
                        helper.make_node("If", ["condition"], ["output"], **branches),
                    ]
                )
                model.graph.ClearField("initializer")
                model.graph.initializer.extend(init for init in source_initializers if init.name in captured)
                model.graph.initializer.append(numpy_helper.from_array(np.eye(2, dtype=np.float32), "identity"))
                model.graph.input.append(helper.make_tensor_value_info("condition", TensorProto.BOOL, []))
                model.graph.output.append(helper.make_tensor_value_info("control", TensorProto.FLOAT, [2, 2]))
                feeds = {"input": np.eye(2, dtype=np.float32), "condition": np.array(True)}
                quantized = self._check_quantized(
                    model, feeds, QuantFormat.QOperator, per_channel, {"EnableSubgraph": True}
                )
                feeds["condition"] = np.array(False)
                expected = self._run(model, feeds, GraphOptimizationLevel.ORT_DISABLE_ALL)
                for level in (GraphOptimizationLevel.ORT_DISABLE_ALL, GraphOptimizationLevel.ORT_ENABLE_ALL):
                    for actual, reference in zip(self._run(quantized, feeds, level), expected, strict=True):
                        np.testing.assert_allclose(actual, reference, rtol=0.02, atol=0.03)

    def test_shared_weight_scale_adjustment(self):
        feeds = {"input": np.eye(2, dtype=np.float32)}
        for betas, quant_format, per_channel in itertools.product(
            ((1.0, 4.0), (4.0, 1.0), (0.5, 4.0), (4.0, 0.5), (-0.5, 4.0), (4.0, -0.5), (0.0, 4.0), (4.0, 0.0)),
            (QuantFormat.QDQ, QuantFormat.QOperator),
            (False, True),
        ):
            with self.subTest(betas=betas, format=quant_format, per_channel=per_channel):
                model = self._model(betas[0], magnitude=0.0001)
                model.graph.initializer[1].CopyFrom(numpy_helper.from_array(np.array([10, 20], np.float32), "bias"))
                model.graph.node[0].name = "first"
                model.graph.node.append(
                    helper.make_node(
                        "Gemm", ["input", "weight", "bias"], ["second_output"], name="second", beta=betas[1]
                    )
                )
                model.graph.output.append(helper.make_tensor_value_info("second_output", TensorProto.FLOAT, [2, 2]))
                self._check_quantized(model, feeds, quant_format, per_channel)

    def test_private_weight_preserves_overrides(self):
        feeds = {"input": np.eye(2, dtype=np.float32)}
        for beta, per_channel, trans_b in itertools.product((2.0, -0.5), (False, True), (False, True)):
            with self.subTest(beta=beta, per_channel=per_channel, trans_b=trans_b):
                model = self._model(beta, trans_b=trans_b)
                model.graph.node.append(helper.make_node("Identity", ["weight"], ["original_weight"]))
                model.graph.output.append(helper.make_tensor_value_info("original_weight", TensorProto.FLOAT, [2, 2]))
                overrides = {"weight": [{"scale": np.float32(0.125), "zero_point": np.int8(0)}]}
                if per_channel:
                    overrides["weight"][0]["axis"] = 0 if trans_b else 1
                    overrides["weight"].append({"scale": np.float32(0.0625), "zero_point": np.int8(0)})
                original_overrides = copy.deepcopy(overrides)
                quantized = self._check_quantized(
                    model, feeds, QuantFormat.QOperator, per_channel, {"TensorQuantOverrides": overrides}
                )
                gemm = next(node for node in quantized.graph.node if node.op_type == "QGemm")
                scale = next(init for init in quantized.graph.initializer if init.name == gemm.input[4])
                np.testing.assert_array_equal(
                    numpy_helper.to_array(scale), np.array([0.125, 0.0625] if per_channel else 0.125, np.float32)
                )
                self.assertEqual(overrides, original_overrides)

    def test_private_names_avoid_existing_tensors(self):
        suffixes = [
            ("bias", suffix)
            for suffix in ("", "_quantized", "_quantized_scale", "_quantized_zero_point", "_DequantizeLinear_Output")
        ] + [("weight", suffix) for suffix in ("", "_quantized", "_scale", "_zero_point")]

        def branch(name, output, depth):
            if depth == 1:
                nodes = [
                    helper.make_node(
                        "Constant", [], [output], value=numpy_helper.from_array(np.array([9, 8], np.float32))
                    )
                ]
            else:
                nodes = [
                    helper.make_node(
                        "If",
                        ["condition"],
                        [name + "_output"],
                        then_branch=branch(name + "_then", output, depth - 1),
                        else_branch=branch(name + "_else", output, depth - 1),
                    )
                ]
                output = name + "_output"
            return helper.make_graph(nodes, name, [], [helper.make_tensor_value_info(output, TensorProto.FLOAT, [2])])

        for (source, suffix), location, quant_format in itertools.product(
            suffixes, ("input", "initializer", "node", "if", "nested_if"), (QuantFormat.QDQ, QuantFormat.QOperator)
        ):
            if source == "weight" and quant_format == QuantFormat.QDQ:
                continue
            with self.subTest(source=source, suffix=suffix, location=location, format=quant_format):
                model = self._model(2.0)
                if source == "weight":
                    model.graph.node.append(helper.make_node("Identity", ["weight"], ["original_weight"]))
                    model.graph.output.append(
                        helper.make_tensor_value_info("original_weight", TensorProto.FLOAT, [2, 2])
                    )
                feeds = {"input": np.eye(2, dtype=np.float32)}
                occupied = source + "_gemm_" + source + suffix
                output = occupied
                if location == "input":
                    model.graph.input.append(helper.make_tensor_value_info(occupied, TensorProto.FLOAT, [2]))
                    feeds[occupied] = np.array([9, 8], np.float32)
                elif location == "initializer":
                    model.graph.initializer.append(numpy_helper.from_array(np.array([9, 8], np.float32), occupied))
                elif location == "node":
                    model.graph.node.append(helper.make_node("Identity", ["bias"], [occupied]))
                else:
                    depth = 1 if location == "if" else 2
                    model.graph.input.append(helper.make_tensor_value_info("condition", TensorProto.BOOL, []))
                    feeds["condition"] = np.array(True)
                    output = "branch_output"
                    model.graph.node.append(
                        helper.make_node(
                            "If",
                            ["condition"],
                            [output],
                            then_branch=branch("then", occupied, depth),
                            else_branch=branch("else", occupied, depth),
                        )
                    )
                model.graph.output.append(helper.make_tensor_value_info(output, TensorProto.FLOAT, [2]))
                self._check_quantized(model, feeds, quant_format, False)

    def test_bias_overrides_keep_beta(self):
        feeds = {"input": np.eye(2, dtype=np.float32)}
        overrides = {
            "bias": [{"quant_type": QuantType.QUInt8, "scale": np.float32(0.125), "zero_point": np.uint8(128)}]
        }
        for beta, per_channel in itertools.product((2.0, -2.0, 0.5, 0.0, 1.0), (False, True)):
            with self.subTest(beta=beta, per_channel=per_channel):
                quantized = self._check_quantized(
                    self._model(beta),
                    feeds,
                    QuantFormat.QDQ,
                    per_channel,
                    {"TensorQuantOverrides": overrides},
                )
                gemm = next(node for node in quantized.graph.node if node.op_type == "Gemm")
                actual_beta = next(attr.f for attr in gemm.attribute if attr.name == "beta")
                self.assertEqual(actual_beta, beta)

    def test_float8_bias_values(self):
        for beta, dtype in itertools.product((2.0, -2.0, 0.5, 0.0, 1.0), (np.float16, np.float32)):
            with self.subTest(beta=beta, dtype=dtype):
                quantizer = BaseQuantizer(
                    self._model(beta, dtype),
                    per_channel=False,
                    reduce_range=False,
                    weight_qType=QuantType.QFLOAT8E4M3FN,
                    activation_qType=QuantType.QFLOAT8E4M3FN,
                    tensors_range=None,
                    nodes_to_quantize=[],
                    nodes_to_exclude=[],
                    op_types_to_quantize=["Gemm"],
                )
                names = quantizer.quantize_bias_static_impl("bias", np.array(0.5), np.array(0.25), beta)
                self.assertEqual(names[4], "Cast")
                quantized = numpy_helper.to_array(quantizer.model.get_initializer(names[0]))
                self.assertEqual(quantized.dtype, np.float32)
                np.testing.assert_array_equal(quantized, np.array([1.0, -2.0], dtype=np.float32) * beta)
                np.testing.assert_array_equal(
                    numpy_helper.to_array(quantizer.model.get_initializer("bias")), np.array([1.0, -2.0], dtype=dtype)
                )


if __name__ == "__main__":
    unittest.main()
