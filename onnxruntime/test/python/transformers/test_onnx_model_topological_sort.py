# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto, helper

source_dir = Path(__file__).resolve().parents[3] / "python" / "tools" / "transformers"
if source_dir.is_dir():
    sys.path.append(str(source_dir))
    from onnx_model import OnnxModel
else:
    from onnxruntime.transformers.onnx_model import OnnxModel


class TestTopologicalSort(unittest.TestCase):
    @staticmethod
    def _graph(nodes, output="d", initializers=()):
        return helper.make_graph(
            nodes,
            "topological_sort",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info(output, TensorProto.FLOAT, [1])],
            list(initializers),
        )

    def test_preserves_forward_scan_order(self):
        for deterministic, expected in ((False, ["a", "z", "b", "c", "d"]), (True, ["a", "b", "c", "d", "z"])):
            with self.subTest(deterministic=deterministic):
                graph = self._graph(
                    [
                        helper.make_node("Identity", [source], [name], name=name)
                        for name, source in (("d", "c"), ("b", "a"), ("a", "x"), ("c", "b"), ("z", "x"))
                    ]
                )
                OnnxModel.graph_topological_sort(graph, deterministic)
                self.assertEqual([node.name for node in graph.node], expected)
                onnx.checker.check_graph(graph)

    def test_repeated_inputs_and_optional_inputs(self):
        graph = self._graph(
            [
                helper.make_node("Add", ["clipped", "clipped"], ["d"], name="add"),
                helper.make_node("Clip", ["bias", "", ""], ["clipped"], name="clip"),
            ],
            initializers=[helper.make_tensor("bias", TensorProto.FLOAT, [1], [2.0])],
        )
        graph.input.append(helper.make_tensor_value_info("bias", TensorProto.FLOAT, [1]))
        OnnxModel.graph_topological_sort(graph)
        self.assertEqual([node.name for node in graph.node], ["clip", "add"])
        onnx.checker.check_graph(graph)

    def test_input_free_nodes_and_empty_graph(self):
        graph = self._graph(
            [
                helper.make_node("Add", ["constant", "x"], ["d"], name="add"),
                helper.make_node(
                    "Constant",
                    [],
                    ["constant"],
                    name="constant",
                    value=helper.make_tensor("value", TensorProto.FLOAT, [1], [1.0]),
                ),
            ]
        )
        OnnxModel.graph_topological_sort(graph)
        self.assertEqual([node.name for node in graph.node], ["constant", "add"])
        onnx.checker.check_graph(graph)
        empty = self._graph([], output="x")
        OnnxModel.graph_topological_sort(empty)
        onnx.checker.check_graph(empty)

    def test_reverse_ordered_chain(self):
        nodes = [
            helper.make_node("Identity", ["x" if index == 0 else f"v{index - 1}"], [f"v{index}"], name=f"n{index}")
            for index in range(256)
        ]
        graph = self._graph(reversed(nodes), output="v255")
        OnnxModel.graph_topological_sort(graph)
        self.assertEqual(list(graph.node), nodes)
        onnx.checker.check_graph(graph)

    def test_invalid_graph_is_not_modified(self):
        for inputs in (("missing", "a"), ("b", "a")):
            with self.subTest(inputs=inputs):
                graph = self._graph(
                    [
                        helper.make_node("Identity", [inputs[0]], ["a"], name="first"),
                        helper.make_node("Identity", [inputs[1]], ["b"], name="second"),
                    ],
                    output="b",
                )
                before = graph.SerializeToString()
                with self.assertRaisesRegex(RuntimeError, "Graph is not a DAG"):
                    OnnxModel.graph_topological_sort(graph)
                self.assertEqual(graph.SerializeToString(), before)


if __name__ == "__main__":
    unittest.main()
