# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

source_dir = Path(__file__).resolve().parents[3] / "python" / "tools" / "transformers"
if source_dir.is_dir():
    sys.path.append(str(source_dir))
    from onnx_model import OnnxModel
else:
    from onnxruntime.transformers.onnx_model import OnnxModel


class TestOutputLiveness(unittest.TestCase):
    @staticmethod
    def _value(name):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1])

    @staticmethod
    def _constant(name, value):
        return helper.make_node(
            "Constant", [], [name], name=name, value=helper.make_tensor("value", TensorProto.FLOAT, [1], [value])
        )

    @staticmethod
    def _model(nodes, inputs, outputs):
        return helper.make_model(
            helper.make_graph(nodes, "output_liveness", inputs, outputs),
            opset_imports=[helper.make_opsetid("", 13)],
            ir_version=10,
        )

    def _check_outputs(self, model, feeds, cleanup, **kwargs):
        onnx.checker.check_model(model)
        expected = ReferenceEvaluator(model).run(None, feeds)
        optimized = OnnxModel(model)
        getattr(optimized, cleanup)(**kwargs)
        onnx.checker.check_model(model)
        actual = ReferenceEvaluator(model).run(None, feeds)
        self.assertEqual(len(expected), len(actual))
        for before, after in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(after, before)
        return optimized

    def test_keeps_constant_graph_outputs(self):
        for cleanup in ("remove_unused_constant", "update_graph", "prune_graph"):
            with self.subTest(cleanup=cleanup):
                model = self._model(
                    [self._constant("answer", 42), self._constant("unused", -1)], [], [self._value("answer")]
                )
                self._check_outputs(model, {}, cleanup)
                self.assertEqual([node.output[0] for node in model.graph.node], ["answer"])

    def test_keeps_passthrough_graph_inputs(self):
        for cleanup in ("update_graph", "prune_graph"):
            with self.subTest(cleanup=cleanup):
                model = self._model([], [self._value("answer"), self._value("unused")], [self._value("answer")])
                self._check_outputs(
                    model,
                    {"answer": np.array([42], dtype=np.float32)},
                    cleanup,
                    allow_remove_graph_inputs=True,
                )
                self.assertEqual([value.name for value in model.graph.input], ["answer"])

    def test_selected_outputs_can_remove_other_passthrough_inputs(self):
        for output in ("first", "second"):
            with self.subTest(output=output):
                model = self._model(
                    [], [self._value("first"), self._value("second")], [self._value("first"), self._value("second")]
                )
                optimized = OnnxModel(model)
                optimized.prune_graph(outputs=[output])
                onnx.checker.check_model(model)
                self.assertEqual([value.name for value in model.graph.input], [output])
                self.assertEqual([value.name for value in model.graph.output], [output])
                actual = ReferenceEvaluator(model).run(None, {output: np.array([42], dtype=np.float32)})
                np.testing.assert_array_equal(actual[0], np.array([42], dtype=np.float32))

    def test_keeps_constant_subgraph_outputs(self):
        for cleanup in ("remove_unused_constant", "update_graph", "prune_graph"):
            for condition in (False, True):
                with self.subTest(cleanup=cleanup, condition=condition):
                    branches = {}
                    for name, value in (("then_branch", 42), ("else_branch", -1)):
                        branches[name] = helper.make_graph(
                            [self._constant(name, value), self._constant(name + "_unused", 0)],
                            name,
                            [],
                            [self._value(name)],
                        )
                    model = self._model(
                        [helper.make_node("If", ["condition"], ["answer"], **branches)],
                        [helper.make_tensor_value_info("condition", TensorProto.BOOL, [])],
                        [self._value("answer")],
                    )
                    optimized = self._check_outputs(model, {"condition": np.array(condition)}, cleanup)
                    for graph in optimized.graphs()[1:]:
                        self.assertEqual([node.output[0] for node in graph.node], [graph.output[0].name])

    def test_keeps_node_inputs_and_removes_unreferenced_values(self):
        for cleanup in ("update_graph", "prune_graph"):
            with self.subTest(cleanup=cleanup):
                model = self._model(
                    [
                        self._constant("bias", 1),
                        self._constant("unused_constant", -1),
                        helper.make_node("Add", ["input", "bias"], ["answer"]),
                    ],
                    [self._value("input"), self._value("unused_input")],
                    [self._value("answer")],
                )
                self._check_outputs(
                    model,
                    {"input": np.array([41], dtype=np.float32)},
                    cleanup,
                    allow_remove_graph_inputs=True,
                )
                self.assertEqual([value.name for value in model.graph.input], ["input"])
                self.assertEqual([node.op_type for node in model.graph.node], ["Constant", "Add"])


if __name__ == "__main__":
    unittest.main()
