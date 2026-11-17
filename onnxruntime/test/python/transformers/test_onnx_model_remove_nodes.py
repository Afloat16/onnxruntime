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


class TestRemoveNodes(unittest.TestCase):
    @staticmethod
    def _model(nodes):
        return OnnxModel(
            helper.make_model(
                helper.make_graph(
                    nodes,
                    "remove_nodes",
                    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
                ),
                opset_imports=[helper.make_opsetid("", 13)],
            )
        )

    @staticmethod
    def _node(name, op_type="Identity"):
        return helper.make_node(op_type, ["x"], [name], name=name)

    def test_removes_copied_nodes_in_arbitrary_order(self):
        nodes = [self._node(name) for name in ("y", "a", "b", "c", "d")]
        model = self._model(nodes)
        model.remove_nodes([nodes[4], nodes[1], nodes[3]])
        self.assertEqual([node.output[0] for node in model.graph().node], ["y", "b"])
        onnx.checker.check_model(model.model)

    def test_preserves_surviving_node_references(self):
        model = self._model([self._node(name) for name in ("a", "y", "b")])
        survivor = model.graph().node[1]
        model.remove_nodes([model.graph().node[0], model.graph().node[2]])
        survivor.name = "still_in_the_graph"
        self.assertEqual(model.graph().node[0].name, "still_in_the_graph")
        onnx.checker.check_model(model.model)

    def test_duplicate_requests_and_outputs_use_node_equality(self):
        original = self._node("same")
        different = self._node("same", "Neg")
        model = self._model([original, different, original, self._node("y")])
        model.remove_nodes([original, original])
        self.assertEqual([node.op_type for node in model.graph().node], ["Neg", "Identity"])
        model.remove_nodes([different])
        onnx.checker.check_model(model.model)

    def test_warns_for_each_unmatched_request(self):
        node = self._node("removed")
        missing = self._node("missing")
        model = self._model([node, self._node("y")])
        with self.assertLogs(OnnxModel.__module__, level="WARNING") as captured:
            model.remove_nodes([node, missing, node])
        self.assertEqual(len(captured.records), 2)
        self.assertEqual(captured.records[0].args, (missing,))
        self.assertEqual(captured.records[1].args, (node,))
        self.assertEqual([value.output[0] for value in model.graph().node], ["y"])

    def test_empty_requests_and_iterator(self):
        nodes = [self._node(name) for name in ("a", "y", "b")]
        model = self._model(nodes)
        before = model.model.SerializeToString()
        model.remove_nodes([])
        self.assertEqual(model.model.SerializeToString(), before)
        model.remove_nodes(iter([nodes[2], nodes[0]]))
        self.assertEqual([value.output[0] for value in model.graph().node], ["y"])

    def test_subgraph_removals_keep_request_order(self):
        branch = helper.make_graph(
            [self._node("branch_output"), self._node("dead")],
            "branch",
            [],
            [helper.make_tensor_value_info("branch_output", TensorProto.FLOAT, [1])],
        )
        node = helper.make_node("If", ["condition"], ["y"], then_branch=branch, else_branch=branch)
        model = self._model([node])
        model.graph().input.append(helper.make_tensor_value_info("condition", TensorProto.BOOL, []))
        parent_copy = onnx.NodeProto()
        parent_copy.CopyFrom(model.graph().node[0])
        # Removing a child changes the parent proto's value. A previously made
        # copy of that parent must no longer match it in a subsequent request.
        with self.assertLogs(OnnxModel.__module__, level="WARNING") as captured:
            model.remove_nodes([self._node("dead"), parent_copy])
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(len(model.graph().node), 1)
        self.assertEqual([len(graph.node) for graph in model.graphs()[1:]], [1, 2])
        onnx.checker.check_model(model.model)


if __name__ == "__main__":
    unittest.main()
