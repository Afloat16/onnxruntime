# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

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


class TestNormalizationPerChannelAxis(unittest.TestCase):
    def _check(self, op_type, qtype, per_channel, source="initializer", override="default", scale_rank=1):
        gamma = np.array([-0.01, 0.1, -10], np.float32)
        x = np.array([[0, 1, 4]], np.float32)
        attributes = {}
        if scale_rank == 2:
            gamma = np.stack([gamma, 2 * gamma])
            x = np.concatenate([x, x + 1], axis=0)
            attributes["axis"] = 0
        elif op_type != "LayerNormalization":
            x = np.tile(x, (1, 3, 1))
        beta = np.array([0.001, 0.01, 1], np.float32)
        inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, x.shape)]
        initializers = []
        if source == "initializer":
            initializers.append(numpy_helper.from_array(gamma, "gamma"))
        else:
            inputs.append(helper.make_tensor_value_info("gamma", TensorProto.FLOAT, gamma.shape))
        node_inputs = ["input", "gamma"]
        if scale_rank == 1:
            initializers.append(numpy_helper.from_array(beta, "beta"))
            node_inputs.append("beta")
        if op_type == "BatchNormalization":
            initializers.extend(
                [
                    numpy_helper.from_array(np.zeros(3, np.float32), "mean"),
                    numpy_helper.from_array(np.ones(3, np.float32), "variance"),
                ]
            )
            node_inputs.extend(["mean", "variance"])
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node(op_type, node_inputs, ["output"], name="normalization", **attributes)],
                "normalization_per_channel",
                inputs,
                [helper.make_tensor_value_info("output", TensorProto.FLOAT, x.shape)],
                initializers,
            ),
            opset_imports=[helper.make_opsetid("", 21)],
            ir_version=7,
        )
        feeds = [
            {"input": data, **({"gamma": gamma * factor} if source == "runtime" else {})}
            for data, factor in ((x, 1), (np.flip(x).copy() - 2, 0.75))
        ]
        extra_options = {"OpTypesToExcludeOutputQuantization": [op_type]}
        if override in ("op_axis", "tensor_priority"):
            extra_options["QDQOpTypePerChannelSupportToAxis"] = {op_type: 1 if override == "tensor_priority" else -1}
        if override in ("tensor_axis", "tensor_priority"):
            extra_options["TensorQuantOverrides"] = {"gamma": [{"axis": -1 if override == "tensor_priority" else 0}]}
        elif override == "per_tensor":
            extra_options["TensorQuantOverrides"] = {"gamma": [{"quant_type": qtype}]}
        expected_per_channel = source == "initializer" and (
            override in ("tensor_axis", "tensor_priority") or (per_channel and override != "per_tensor")
        )
        expected_axis = 0 if scale_rank == 1 else 1
        onnx.checker.check_model(model, full_check=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quantized.onnx"
            quantize_static(
                model,
                path,
                TestDataFeeds(feeds),
                quant_format=QuantFormat.QDQ,
                per_channel=per_channel,
                activation_type=QuantType.QUInt8,
                weight_type=qtype,
                extra_options=extra_options,
            )
            quantized = onnx.load(path)
            onnx.checker.check_model(quantized, full_check=True)
            producers = {output: node for node in quantized.graph.node for output in node.output}
            normalization = next(node for node in quantized.graph.node if node.op_type == op_type)
            gamma_dq = producers[normalization.input[1]]
            self.assertEqual(gamma_dq.op_type, "DequantizeLinear")
            arrays = {tensor.name: numpy_helper.to_array(tensor) for tensor in quantized.graph.initializer}
            gamma_scale = arrays[gamma_dq.input[1]]
            self.assertEqual(gamma_scale.shape, (gamma.shape[expected_axis],) if expected_per_channel else ())
            if expected_per_channel:
                self.assertEqual(
                    next(attribute.i for attribute in gamma_dq.attribute if attribute.name == "axis"), expected_axis
                )
                if scale_rank == 1:
                    restored = (
                        arrays[gamma_dq.input[0]].astype(np.int32) - arrays[gamma_dq.input[2]].astype(np.int32)
                    ) * gamma_scale
                    np.testing.assert_array_less(
                        np.abs(restored - gamma),
                        gamma_scale / 2 + np.finfo(np.float32).eps * np.maximum(1, np.abs(gamma)),
                    )

            for optimization in (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
                options = ort.SessionOptions()
                options.graph_optimization_level = optimization
                options.intra_op_num_threads = options.inter_op_num_threads = 1
                options.log_severity_level = 3
                original = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
                actual = ort.InferenceSession(
                    quantized.SerializeToString(), options, providers=["CPUExecutionProvider"]
                )
                for feed in feeds:
                    values = feed["input"].astype(np.float64)
                    current_gamma = feed.get("gamma", gamma)
                    if op_type == "LayerNormalization":
                        axes = tuple(range(attributes.get("axis", values.ndim - 1), values.ndim))
                        expected = (values - values.mean(axis=axes, keepdims=True)) / np.sqrt(
                            values.var(axis=axes, keepdims=True) + 1e-5
                        )
                        expected = expected * current_gamma + (beta if scale_rank == 1 else 0)
                    elif op_type == "InstanceNormalization":
                        expected = (values - values.mean(axis=2, keepdims=True)) / np.sqrt(
                            values.var(axis=2, keepdims=True) + 1e-5
                        )
                        expected = expected * current_gamma.reshape(1, 3, 1) + beta.reshape(1, 3, 1)
                    else:
                        expected = values / np.sqrt(1 + 1e-5) * current_gamma.reshape(1, 3, 1) + beta.reshape(1, 3, 1)
                    original_result = original.run(None, feed)[0]
                    result = actual.run(None, feed)[0]
                    np.testing.assert_allclose(original_result, expected, rtol=1e-6, atol=1e-6)
                    np.testing.assert_allclose(result, expected, rtol=0.03, atol=0.2)
                    if expected_per_channel and scale_rank == 1:
                        small_channels = (slice(None), slice(0, 2))
                        np.testing.assert_allclose(result[small_channels], expected[small_channels], rtol=0, atol=0.003)

    def test_rank_one_defaults(self):
        for op_type, qtype, source, per_channel in itertools.product(
            ("InstanceNormalization", "LayerNormalization", "BatchNormalization"),
            (QuantType.QInt8, QuantType.QUInt8),
            ("initializer", "runtime"),
            (False, True),
        ):
            with self.subTest(op_type=op_type, qtype=qtype, source=source, per_channel=per_channel):
                self._check(op_type, qtype, per_channel, source)

    def test_explicit_overrides(self):
        for op_type, qtype, per_channel, override in itertools.product(
            ("InstanceNormalization", "LayerNormalization", "BatchNormalization"),
            (QuantType.QInt8, QuantType.QUInt8),
            (False, True),
            ("op_axis", "tensor_axis", "per_tensor", "tensor_priority"),
        ):
            with self.subTest(op_type=op_type, qtype=qtype, per_channel=per_channel, override=override):
                self._check(op_type, qtype, per_channel, override=override)

    def test_higher_rank_layer_normalization(self):
        for qtype, per_channel, override in itertools.product(
            (QuantType.QInt8, QuantType.QUInt8), (False, True), ("default", "per_tensor")
        ):
            with self.subTest(qtype=qtype, per_channel=per_channel, override=override):
                self._check("LayerNormalization", qtype, per_channel, override=override, scale_rank=2)


if __name__ == "__main__":
    unittest.main()
