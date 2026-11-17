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


class TestCleanInitializers(unittest.TestCase):
    @staticmethod
    def _value_info(name):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1])

    @staticmethod
    def _weight(name, value):
        tensor = numpy_helper.from_array(np.array([value], dtype=np.float32), name)
        tensor.doc_string = f"Metadata for {name}"
        return tensor

    def _run(self, model, feeds):
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = onnxruntime.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
        return session.run(None, feeds)

    def test_cleanup_preserves_order_and_initializer_outputs(self):
        retained_patterns = (
            (),
            tuple(range(16)),
            tuple(range(8)),
            tuple(range(8, 16)),
            tuple(range(4, 12)),
            tuple(range(0, 16, 2)),
            (0, 3, 4, 7, 8, 9, 15),
        )
        for ir_version in (3, 10):
            for retained in retained_patterns:
                with self.subTest(ir_version=ir_version, retained=retained):
                    weights = [self._weight(f"w{index}", index) for index in range(16)]
                    weights.insert(5, self._weight("constant_output", 42))
                    used_names = {f"w{index}" for index in retained} | {"constant_output"}
                    inputs = [self._value_info("input")]
                    if ir_version < 4:
                        inputs.extend(self._value_info(weight.name) for weight in reversed(weights))
                    node = helper.make_node("Sum", ["input", *[f"w{index}" for index in retained]], ["output"])
                    model = helper.make_model(
                        helper.make_graph(
                            [node],
                            "cleanup_ranges",
                            inputs,
                            [self._value_info("output"), self._value_info("constant_output")],
                            weights,
                        ),
                        opset_imports=[helper.make_opsetid("", 13)],
                        ir_version=ir_version,
                    )
                    onnx.checker.check_model(model, full_check=True)
                    expected_weights = [weight.SerializeToString() for weight in weights if weight.name in used_names]
                    expected_inputs = [
                        value.SerializeToString()
                        for value in inputs
                        if value.name == "input" or value.name in used_names
                    ]
                    original = self._run(model, {"input": np.array([-3], dtype=np.float32)})

                    _, requesting = ONNXModel(model).clean_initializers()

                    self.assertEqual(requesting, set())
                    self.assertEqual([value.SerializeToString() for value in model.graph.initializer], expected_weights)
                    self.assertEqual([value.SerializeToString() for value in model.graph.input], expected_inputs)
                    onnx.checker.check_model(model, full_check=True)
                    actual = self._run(model, {"input": np.array([-3], dtype=np.float32)})
                    for expected, value in zip(original, actual, strict=True):
                        np.testing.assert_array_equal(value, expected)

    def _branch(self, prefix):
        return helper.make_graph(
            [
                helper.make_node("Add", ["input", "captured_weight"], [f"{prefix}_sum"]),
                helper.make_node("Add", [f"{prefix}_sum", f"{prefix}_used"], [f"{prefix}_output"]),
            ],
            prefix,
            [],
            [self._value_info(f"{prefix}_output")],
            [self._weight(f"{prefix}_unused", -10), self._weight(f"{prefix}_used", 2)],
        )

    def test_cleanup_retains_nested_captures(self):
        inner = helper.make_node(
            "If",
            ["condition"],
            ["inner_output"],
            then_branch=self._branch("inner_then"),
            else_branch=self._branch("inner_else"),
        )
        nested = helper.make_graph(
            [inner], "nested", [], [self._value_info("inner_output")], [self._weight("nested_unused", 7)]
        )
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "If", ["condition"], ["output"], then_branch=nested, else_branch=self._branch("else")
                    )
                ],
                "nested_cleanup",
                [self._value_info("input"), helper.make_tensor_value_info("condition", TensorProto.BOOL, [])],
                [self._value_info("output")],
                [self._weight("unused_before", 5), self._weight("captured_weight", 3), self._weight("unused_after", 9)],
            ),
            opset_imports=[helper.make_opsetid("", 13)],
            ir_version=10,
        )
        original = model.SerializeToString()
        _, requesting = ONNXModel(model).clean_initializers()
        self.assertEqual(requesting, set())
        self.assertEqual([value.name for value in model.graph.initializer], ["captured_weight"])

        def check_graph(graph):
            self.assertFalse(any("unused" in value.name for value in graph.initializer))
            for node in graph.node:
                for attribute in node.attribute:
                    if attribute.type == onnx.AttributeProto.GRAPH:
                        check_graph(attribute.g)

        check_graph(model.graph)
        onnx.checker.check_model(model, full_check=True)
        for condition in (False, True):
            feeds = {"input": np.array([-1], dtype=np.float32), "condition": np.array(condition)}
            expected = self._run(onnx.load_from_string(original), feeds)
            actual = self._run(model, feeds)
            np.testing.assert_array_equal(actual[0], expected[0])

    def test_cleanup_retains_captures_from_graph_list_attributes(self):
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "GraphGroup", [], ["output"], graphs=[self._branch("first"), self._branch("second")]
                    )
                ],
                "graph_list_cleanup",
                [self._value_info("input")],
                [self._value_info("output")],
                [self._weight("captured_weight", 3), self._weight("unused", 1)],
            )
        )
        _, requesting = ONNXModel(model).clean_initializers()
        self.assertEqual(requesting, set())
        self.assertEqual([value.name for value in model.graph.initializer], ["captured_weight"])
        graphs = model.graph.node[0].attribute[0].graphs
        self.assertEqual(
            [[value.name for value in graph.initializer] for graph in graphs], [["first_used"], ["second_used"]]
        )


if __name__ == "__main__":
    unittest.main()
