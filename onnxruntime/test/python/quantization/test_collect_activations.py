# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import gc
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import onnxruntime
from onnxruntime.quantization import qdq_loss_debug


class TestCollectActivations(unittest.TestCase):
    @staticmethod
    def _model(path, augmented=True, reverse_outputs=False):
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("Relu", ["input"], ["activation"]),
                    helper.make_node("ConstantOfShape", ["shape"], ["ordinary_output"]),
                ],
                "activation_collection",
                [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None])],
                [
                    helper.make_tensor_value_info("ordinary_output", TensorProto.FLOAT, [4, 4]),
                    helper.make_tensor_value_info("activation", TensorProto.FLOAT, [None]),
                ],
                [numpy_helper.from_array(np.array([4, 4], np.int64), "shape")],
            ),
            opset_imports=[helper.make_opsetid("", 13)],
            ir_version=10,
        )
        onnx.save(model, path)
        if augmented:
            qdq_loss_debug.modify_model_output_intermediate_tensors(path, path, op_types_for_saving=["Relu"])
        if reverse_outputs:
            model = onnx.load(path)
            outputs = list(reversed(model.graph.output))
            model.graph.ClearField("output")
            model.graph.output.extend(outputs)
            onnx.save(model, path)

    @staticmethod
    def _options():
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        return options

    def _collect_with_retention_tracking(self, path, inputs):
        discarded = []
        live_counts = []
        real_session = onnxruntime.InferenceSession

        class TrackingSession:
            def __init__(self, *args, **kwargs):
                self.session = real_session(*args, **kwargs)

            def get_outputs(self):
                return self.session.get_outputs()

            def run(self, *args, **kwargs):
                outputs = self.session.run(*args, **kwargs)
                for info, output in zip(self.session.get_outputs(), outputs, strict=True):
                    if info.name in {"ordinary_output", "activation"}:
                        discarded.append(weakref.ref(output))
                return outputs

        def reader():
            for values in inputs:
                gc.collect()
                live_counts.append(sum(reference() is not None for reference in discarded))
                yield {"input": values}
            gc.collect()
            live_counts.append(sum(reference() is not None for reference in discarded))

        with patch.object(qdq_loss_debug.onnxruntime, "InferenceSession", TrackingSession):
            result = qdq_loss_debug.collect_activations(str(path), reader(), self._options())
        return result, live_counts

    def test_discarded_outputs_do_not_accumulate_across_batches(self):
        for reverse_outputs in (False, True):
            with self.subTest(reverse_outputs=reverse_outputs), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.onnx"
                self._model(path, reverse_outputs=reverse_outputs)
                inputs = [np.arange(size, dtype=np.float32) - size / 2 for size in (3, 8, 1, 0, 5, 2)]
                result, live_counts = self._collect_with_retention_tracking(path, inputs)
                self.assertLessEqual(max(live_counts), 1)
                self.assertEqual(set(result), {"input", "activation"})
                for index, values in enumerate(inputs):
                    np.testing.assert_array_equal(result["input"][index], values)
                    np.testing.assert_array_equal(result["activation"][index], np.maximum(values, 0))
                self.assertEqual(len(result["input"]), len(inputs))
                self.assertEqual(len(result["activation"]), len(inputs))

    def test_empty_reader_still_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            self._model(path)
            with self.assertRaisesRegex(RuntimeError, "No data is collected"):
                qdq_loss_debug.collect_activations(str(path), iter(()), self._options())

    def test_nonempty_reader_without_saved_outputs_returns_empty_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            self._model(path, augmented=False)
            reader = iter([{"input": np.array([-1, 2], np.float32)} for _ in range(3)])
            self.assertEqual(qdq_loss_debug.collect_activations(str(path), reader, self._options()), {})


if __name__ == "__main__":
    unittest.main()
