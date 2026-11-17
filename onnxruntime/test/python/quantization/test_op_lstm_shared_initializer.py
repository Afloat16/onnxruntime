# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import copy
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic


class TestLSTMSharedInitializer(unittest.TestCase):
    def test_shared_lstm_weights(self):
        for sharing, per_channel, direction in itertools.product(
            ("W", "R", "both", "same"), (False, True), ("forward", "bidirectional")
        ):
            with self.subTest(sharing=sharing, per_channel=per_channel, direction=direction):
                self.check_shared_weights(sharing, "lstm", per_channel, direction)

    def test_float_consumers_of_lstm_weights(self):
        for consumer, per_channel, direction in itertools.product(
            ("identity", "output"), (False, True), ("forward", "bidirectional")
        ):
            with self.subTest(consumer=consumer, per_channel=per_channel, direction=direction):
                self.check_shared_weights("both", consumer, per_channel, direction)

    def test_matmul_consumers_of_lstm_weights(self):
        for consumer, per_channel, direction in itertools.product(
            ("matmul_before", "matmul_after"), (False, True), ("forward", "bidirectional")
        ):
            with self.subTest(consumer=consumer, per_channel=per_channel, direction=direction):
                self.check_shared_weights("both", consumer, per_channel, direction)

    def test_single_lstm_control(self):
        for per_channel, direction in itertools.product((False, True), ("forward", "bidirectional")):
            with self.subTest(per_channel=per_channel, direction=direction):
                self.check_shared_weights("both", "single", per_channel, direction)

    def test_lstm_quantization_overrides_use_original_names(self):
        for per_channel, direction in itertools.product((False, True), ("forward", "bidirectional")):
            with self.subTest(per_channel=per_channel, direction=direction):
                self.check_shared_weights("both", "lstm", per_channel, direction, use_overrides=True)

    def test_lstm_weight_names_respect_existing_graph_values(self):
        for per_channel, direction in itertools.product((False, True), ("forward", "bidirectional")):
            with self.subTest(per_channel=per_channel, direction=direction):
                self.check_shared_weights("both", "single", per_channel, direction, collisions=True)

    def test_subgraph_weights_keep_separate_scopes(self):
        for per_channel in (False, True):
            with self.subTest(per_channel=per_channel), tempfile.TemporaryDirectory() as directory:
                models = []
                branches = {}
                for branch in ("then_branch", "else_branch"):
                    model, feeds, _ = self.make_model("both", "single", "forward", False)
                    if branch == "else_branch":
                        for value in model.graph.initializer:
                            value.CopyFrom(numpy_helper.from_array(-numpy_helper.to_array(value), value.name))
                    models.append(model)
                    branches[branch] = copy.deepcopy(model.graph)
                    branches[branch].ClearField("input")
                model = helper.make_model(
                    helper.make_graph(
                        [helper.make_node("If", ["condition"], ["Y"], **branches)],
                        "scoped-lstm-weights",
                        [
                            helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 1, 2]),
                            helper.make_tensor_value_info("condition", TensorProto.BOOL, []),
                        ],
                        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 1, 1, 3])],
                    ),
                    # Keep this scope regression independent of subgraph opset propagation.
                    opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("com.microsoft", 1)],
                    ir_version=7,
                )
                models.append(model)
                sessions = []
                session_options = ort.SessionOptions()
                session_options.intra_op_num_threads = 1
                session_options.inter_op_num_threads = 1
                for index, model in enumerate(models):
                    onnx.checker.check_model(model, full_check=True)
                    path = Path(directory) / f"quantized{index}.onnx"
                    quantize_dynamic(
                        model,
                        path,
                        per_channel=per_channel,
                        weight_type=QuantType.QInt8,
                        extra_options={"EnableSubgraph": True},
                    )
                    onnx.checker.check_model(onnx.load(path), full_check=True)
                    sessions.append(
                        ort.InferenceSession(
                            str(path), sess_options=session_options, providers=["CPUExecutionProvider"]
                        )
                    )
                expected = [session.run(None, feeds)[0] for session in sessions[:2]]
                self.assertFalse(np.array_equal(*expected))
                for condition, reference in zip((True, False), expected, strict=True):
                    actual = sessions[2].run(None, {**feeds, "condition": np.array(condition)})[0]
                    np.testing.assert_array_equal(actual, reference)

    def make_model(self, sharing, consumer, direction, independent, collisions=False):
        rng = np.random.default_rng(42)
        directions = 2 if direction == "bidirectional" else 1
        # A single column keeps batched MatMul's quantization parameters scalar,
        # which is supported by the CPU MatMulInteger kernel in both modes.
        matmul_consumer = consumer in ("matmul_before", "matmul_after")
        hidden_size = 1 if matmul_consumer else 3
        input_size = hidden_size if sharing == "same" or matmul_consumer else 2
        weights = {
            "W": rng.normal(0, 0.2, (directions, 4 * hidden_size, input_size)).astype(np.float32),
            "R": rng.normal(0, 0.2, (directions, 4 * hidden_size, hidden_size)).astype(np.float32),
        }
        if sharing == "same":
            weights["R"] = weights["W"]
        initializers = {}

        def weight_name(kind, use):
            name = "shared" if sharing == "same" else kind
            if independent or (use == "lstm1" and sharing not in (kind, "both", "same")):
                name += "_" + use + "_" + kind
            if name not in initializers:
                initializers[name] = numpy_helper.from_array(weights[kind], name)
            return name

        inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 1, input_size])]
        feeds = {"X": rng.normal(0, 0.5, (4, 1, input_size)).astype(np.float32)}
        nodes = []
        outputs = []
        for index in range(2 if consumer == "lstm" else 1):
            use = f"lstm{index}"
            nodes.append(
                helper.make_node(
                    "LSTM",
                    ["X", weight_name("W", use), weight_name("R", use)],
                    [f"Y{index}"],
                    name=use,
                    hidden_size=hidden_size,
                    direction=direction,
                )
            )
            outputs.append(
                helper.make_tensor_value_info(f"Y{index}", TensorProto.FLOAT, [4, directions, 1, hidden_size])
            )

        if consumer in ("identity", "output"):
            for kind in ("W", "R"):
                name = weight_name(kind, "float")
                if consumer == "identity":
                    nodes.append(helper.make_node("Identity", [name], [kind + "_original"]))
                    name = kind + "_original"
                outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, weights[kind].shape))
        elif consumer in ("matmul_before", "matmul_after"):
            inputs.append(helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 1, 4 * hidden_size]))
            feeds["Z"] = rng.normal(0, 0.5, (1, 1, 4 * hidden_size)).astype(np.float32)
            matmuls = []
            for kind in ("W", "R"):
                matmuls.append(helper.make_node("MatMul", ["Z", weight_name(kind, "matmul")], [kind + "_product"]))
                outputs.append(
                    helper.make_tensor_value_info(
                        kind + "_product", TensorProto.FLOAT, [directions, 1, weights[kind].shape[-1]]
                    )
                )
            nodes = matmuls + nodes if consumer == "matmul_before" else nodes + matmuls

        if collisions:
            for index, name in enumerate(("W_lstm_quantized", "W_lstm_0_scale")):
                initializers[name] = numpy_helper.from_array(np.array(index + 5, dtype=np.float32), name)
                nodes.append(helper.make_node("Identity", [name], [f"kept{index}"]))
                outputs.append(helper.make_tensor_value_info(f"kept{index}", TensorProto.FLOAT, []))
            branches = {}
            for branch, value in (("then_branch", 7), ("else_branch", 9)):
                branches[branch] = helper.make_graph(
                    [
                        helper.make_node(
                            "Constant",
                            [],
                            ["W_lstm_1_zero_point"],
                            value=numpy_helper.from_array(np.array(value, dtype=np.float32)),
                        )
                    ],
                    branch,
                    [],
                    [helper.make_tensor_value_info("W_lstm_1_zero_point", TensorProto.FLOAT, [])],
                )
            inputs.append(helper.make_tensor_value_info("condition", TensorProto.BOOL, []))
            feeds["condition"] = np.array(True)
            nodes.append(helper.make_node("If", ["condition"], ["kept_branch"], **branches))
            outputs.append(helper.make_tensor_value_info("kept_branch", TensorProto.FLOAT, []))

        model = helper.make_model(
            helper.make_graph(nodes, "shared-lstm-weights", inputs, outputs, list(initializers.values())),
            opset_imports=[helper.make_opsetid("", 13)],
            ir_version=7,
        )
        onnx.checker.check_model(model, full_check=True)
        return model, feeds, weights

    def check_shared_weights(self, sharing, consumer, per_channel, direction, use_overrides=False, collisions=False):
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        results = []
        for independent in (True, False):
            model, feeds, weights = self.make_model(sharing, consumer, direction, independent, collisions)
            original_weight_names = {
                name for node in model.graph.node if node.op_type == "LSTM" for name in node.input[1:3]
            }
            options = {}
            if use_overrides:
                options["TensorQuantOverrides"] = {
                    value.name: [
                        {
                            "axis": 0,
                            "scale": np.float32(0.03125 if value.name.startswith("W") else 0.015625),
                            "zero_point": np.int8(0),
                        }
                    ]
                    for value in model.graph.initializer
                }
            original_options = copy.deepcopy(options)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "quantized.onnx"
                quantize_dynamic(
                    model,
                    path,
                    per_channel=per_channel,
                    weight_type=QuantType.QInt8,
                    op_types_to_quantize=["LSTM", "MatMul"],
                    extra_options=options,
                )
                quantized = onnx.load(path)
                onnx.checker.check_model(quantized, full_check=True)
                if consumer in ("lstm", "single"):
                    self.assertTrue(
                        original_weight_names.isdisjoint(value.name for value in quantized.graph.initializer)
                    )
                self.assertEqual(options.get("TensorQuantOverrides"), original_options.get("TensorQuantOverrides"))
                lstms = [node for node in quantized.graph.node if node.op_type == "DynamicQuantizeLSTM"]
                self.assertEqual(
                    len(lstms),
                    2 if consumer == "lstm" else 1,
                )
                if not independent and consumer == "lstm":
                    for index, kind in ((1, "W"), (2, "R")):
                        if sharing in (kind, "both", "same"):
                            self.assertEqual(lstms[0].input[index], lstms[1].input[index])
                    if sharing == "same":
                        self.assertEqual(lstms[0].input[1], lstms[0].input[2])
                if use_overrides:
                    inits = {value.name: numpy_helper.to_array(value) for value in quantized.graph.initializer}
                    for node in lstms:
                        for index, scale in ((8, 0.03125), (10, 0.015625)):
                            np.testing.assert_array_equal(inits[node.input[index]], scale)
                results.append(
                    ort.InferenceSession(
                        str(path), sess_options=session_options, providers=["CPUExecutionProvider"]
                    ).run(None, feeds)
                )
            if not independent and consumer in ("identity", "output"):
                inits = {value.name: numpy_helper.to_array(value) for value in quantized.graph.initializer}
                for index, kind in enumerate(("W", "R"), 1):
                    np.testing.assert_array_equal(results[-1][index], weights[kind])
                    self.assertIn(kind, inits)
                    np.testing.assert_array_equal(inits[kind], weights[kind])
        # Duplicate initializers provide an independent, alias-free quantization
        # reference, including MatMul's ordinary (unpacked) weight layout.
        for expected, actual in zip(*results, strict=True):
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
