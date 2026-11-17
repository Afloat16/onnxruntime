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
from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic, quantize_static


class TestEmbedLayerNormEligibility(unittest.TestCase):
    def _check(self, position_source, mask_index_type, quantization, segments, following=False, mask_output="present"):
        rng = np.random.default_rng(42)
        arrays = {
            "word": rng.uniform(-1, 1, (16, 4)).astype(np.float32),
            "position": rng.uniform(-1, 1, (8, 4)).astype(np.float32),
            "gamma": np.array([1, 0.9, 1.1, 0.8], dtype=np.float32),
            "beta": np.array([-0.2, 0.1, 0.15, -0.1], dtype=np.float32),
        }
        if segments:
            arrays["segment"] = rng.uniform(-1, 1, (2, 4)).astype(np.float32)
        inputs = [helper.make_tensor_value_info(name, TensorProto.INT32, [2, 3]) for name in ("ids", "mask")]
        node_inputs = [
            "ids",
            "segments" if segments else "",
            "word",
            "position",
            "segment" if segments else "",
            "gamma",
            "beta",
            "mask",
        ]
        if segments:
            inputs.append(helper.make_tensor_value_info("segments", TensorProto.INT32, [2, 3]))
        if position_source == "runtime":
            node_inputs.append("positions")
            inputs.append(helper.make_tensor_value_info("positions", TensorProto.INT32, [2, 3]))
        elif position_source == "initializer":
            node_inputs.append("positions")
            arrays["positions"] = np.array([[2, 0, 4]], dtype=np.int32)
        elif position_source == "empty":
            node_inputs.append("")
        node_outputs = ["embedded"]
        if mask_output != "omitted":
            node_outputs.append("mask_index" if mask_output == "present" else "")
        attrs = {"epsilon": 1e-4}
        if mask_index_type is not None:
            attrs["mask_index_type"] = mask_index_type
        nodes = [
            helper.make_node(
                "EmbedLayerNormalization", node_inputs, node_outputs, domain="com.microsoft", name="embedding", **attrs
            )
        ]
        output_name = "embedded"
        if following:
            arrays["after_weight"] = np.eye(4, dtype=np.float32)
            nodes.append(helper.make_node("MatMul", [output_name, "after_weight"], ["y"], name="after"))
            output_name = "y"
        outputs = [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [2, 3, 4])]
        if mask_output == "present":
            outputs.append(helper.make_tensor_value_info("mask_index", TensorProto.INT32, [2]))
        graph = helper.make_graph(
            nodes,
            "embedding_eligibility",
            inputs,
            outputs,
            [numpy_helper.from_array(value, name) for name, value in arrays.items()],
            value_info=[helper.make_tensor_value_info("embedded", TensorProto.FLOAT, [2, 3, 4])] if following else [],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 14), helper.make_opsetid("com.microsoft", 1)], ir_version=7
        )
        feeds = []
        for index in range(2):
            feed = {
                "ids": np.array([[1, 2, 3], [3, 4, 5]], dtype=np.int32) + index,
                "mask": np.array([[1, 1, 0], [1, 0, 0]], dtype=np.int32),
            }
            if segments:
                feed["segments"] = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int32)
            if position_source == "runtime":
                feed["positions"] = np.array([[3, 2, 1], [5, 4, 3]], dtype=np.int32) + index
            feeds.append(feed)
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "quantized.onnx"
            if quantization == "dynamic":
                quantize_dynamic(model, output_path)
            else:
                qtype = QuantType.QInt8 if quantization == "signed" else QuantType.QUInt8
                quantize_static(
                    model,
                    output_path,
                    TestDataFeeds(feeds),
                    quant_format=QuantFormat.QOperator,
                    activation_type=qtype,
                    weight_type=qtype,
                )
            quantized = onnx.load(output_path)
        onnx.checker.check_model(quantized, full_check=True)
        for optimization in (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
            options = ort.SessionOptions()
            options.graph_optimization_level = optimization
            options.intra_op_num_threads = 1
            original = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
            actual = ort.InferenceSession(quantized.SerializeToString(), options, providers=["CPUExecutionProvider"])
            for feed in feeds:
                positions = feed.get("positions", arrays.get("positions", np.arange(3)))
                embedding_sum = arrays["word"][feed["ids"]] + arrays["position"][positions]
                if segments:
                    embedding_sum += arrays["segment"][feed["segments"]]
                centered = embedding_sum - embedding_sum.mean(axis=-1, keepdims=True)
                expected = centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + 1e-4)
                expected = expected * arrays["gamma"] + arrays["beta"]
                original_outputs = original.run(None, feed)
                actual_outputs = actual.run(None, feed)
                np.testing.assert_allclose(original_outputs[0], expected, atol=1e-5, rtol=1e-5)
                np.testing.assert_allclose(actual_outputs[0], expected, atol=0.04, rtol=0.03)
                if mask_output == "present":
                    expected_mask = feed["mask"].sum(axis=1, dtype=np.int32)
                    np.testing.assert_array_equal(original_outputs[1], expected_mask)
                    np.testing.assert_array_equal(actual_outputs[1], expected_mask)
        supported = position_source in ("omitted", "empty") and mask_output == "present"
        self.assertEqual(
            sum(node.op_type == "QEmbedLayerNormalization" for node in quantized.graph.node), int(supported)
        )
        self.assertEqual(
            sum(node.op_type == "EmbedLayerNormalization" for node in quantized.graph.node), int(not supported)
        )
        if supported:
            embedding_node = next(node for node in quantized.graph.node if node.op_type == "QEmbedLayerNormalization")
            self.assertEqual([attribute.name for attribute in embedding_node.attribute], ["epsilon"])
        else:
            for name in ("word", "position", "gamma", "beta"):
                value = next(value for value in quantized.graph.initializer if value.name == name)
                np.testing.assert_array_equal(numpy_helper.to_array(value), arrays[name])
        if following:
            self.assertIn(
                "MatMulInteger" if quantization == "dynamic" else "QLinearMatMul",
                [node.op_type for node in quantized.graph.node],
            )

    def test_positions_and_mask_attribute(self):
        for args in itertools.product(
            ("omitted", "empty", "runtime", "initializer"),
            (None, 0, 1),
            ("dynamic", "unsigned", "signed"),
            (False, True),
        ):
            with self.subTest(args=args):
                self._check(*args)

    def test_following_matmul_remains_quantized(self):
        for position_source, quantization, segments in itertools.product(
            ("omitted", "empty", "runtime", "initializer"), ("dynamic", "unsigned", "signed"), (False, True)
        ):
            with self.subTest(position_source=position_source, quantization=quantization, segments=segments):
                self._check(position_source, None, quantization, segments, following=True)

    def test_missing_mask_output_remains_float(self):
        for quantization, segments, mask_output in itertools.product(
            ("dynamic", "unsigned", "signed"), (False, True), ("omitted", "empty")
        ):
            with self.subTest(quantization=quantization, segments=segments, mask_output=mask_output):
                self._check("omitted", 0, quantization, segments, mask_output=mask_output)


if __name__ == "__main__":
    unittest.main()
