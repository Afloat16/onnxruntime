# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import onnxruntime
from onnxruntime.quantization.calibrate import CalibrationDataReader, MinMaxCalibrater


class _Reader(CalibrationDataReader):
    def __init__(self, inputs):
        self.inputs = iter([inputs])

    def get_next(self):
        return next(self.inputs, None)


class TestCalibrationAugmentation(unittest.TestCase):
    @staticmethod
    def _model(opset):
        shape = [2, 3, 2]
        inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name in ("a", "b", "unused")]
        outputs = [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name in ("out", "leaf", "b", "unused")
        ]
        nodes = [
            helper.make_node("Add", ["a", "a"], ["doubled"], name="double"),
            helper.make_node("Clip", ["b", "", ""], ["clipped"], name="clip"),
            helper.make_node("Add", ["doubled", "clipped"], ["sum"], name="sum"),
            helper.make_node("Add", ["sum", "bias"], ["out"], name="output"),
            helper.make_node("Neg", ["a"], ["leaf"], name="leaf"),
        ]
        for node in nodes:
            node.doc_string = f"Metadata for {node.name}"
        graph = helper.make_graph(
            nodes,
            "calibration-branches",
            inputs,
            outputs,
            initializer=[numpy_helper.from_array(np.array(0.25, dtype=np.float32), "bias")],
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])

    def test_ranges_and_first_consumer_placement(self):
        a = np.linspace(-4, 4, 12, dtype=np.float32).reshape(2, 3, 2)
        b = np.linspace(-2, 3, 12, dtype=np.float32).reshape(2, 3, 2)
        values = {"a": a, "b": b, "doubled": a + a, "clipped": b, "leaf": -a}
        values["sum"] = values["doubled"] + b
        values["out"] = values["sum"] + np.float32(0.25)
        inputs = {"a": a, "b": b, "unused": a - b}

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.onnx"
            augmented_path = Path(directory) / "augmented.onnx"
            for opset in (13, 18):
                for per_channel in (False, True):
                    for op_types in (None, ["Clip"]):
                        with self.subTest(opset=opset, per_channel=per_channel, op_types=op_types):
                            onnx.save(self._model(opset), model_path)
                            calibrater = MinMaxCalibrater(
                                model_path,
                                op_types_to_calibrate=op_types,
                                augmented_model_path=str(augmented_path),
                                per_channel=per_channel,
                            )
                            original_nodes = list(calibrater.model.graph.node)
                            original_bytes = [node.SerializeToString() for node in original_nodes]
                            original_outputs = [output.SerializeToString() for output in calibrater.model.graph.output]
                            tensors, _ = calibrater.select_tensors_to_calibrate(calibrater.model)
                            calibrater.augment_graph()
                            model = onnx.load(augmented_path)
                            onnx.checker.check_model(model, full_check=True)

                            original_names = {node.name for node in original_nodes}
                            self.assertEqual(
                                [node.SerializeToString() for node in model.graph.node if node.name in original_names],
                                original_bytes,
                            )
                            self.assertEqual(
                                [output.SerializeToString() for output in model.graph.output[: len(original_outputs)]],
                                original_outputs,
                            )
                            self.assertEqual(len(model.graph.node), len(original_nodes) + 4 * len(tensors))
                            self.assertNotIn("bias", tensors)
                            self.assertNotIn("unused", tensors)
                            positions = {node.name: index for index, node in enumerate(model.graph.node)}
                            for name in tensors:
                                first_consumer = next((node for node in original_nodes if name in node.input), None)
                                minimum = positions[name + "_ReduceMin"]
                                maximum = positions[name + "_ReduceMax"]
                                self.assertEqual(positions[name + "_ReduceMax_Reshape"], maximum + 1)
                                self.assertEqual(positions[name + "_ReduceMin_Reshape"], minimum + 1)
                                self.assertLess(maximum, minimum)
                                if first_consumer is None:
                                    self.assertGreater(maximum, max(positions[node.name] for node in original_nodes))
                                else:
                                    self.assertLess(minimum + 1, positions[first_consumer.name])

                            options = onnxruntime.SessionOptions()
                            options.intra_op_num_threads = 1
                            options.inter_op_num_threads = 1
                            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
                            calibrater.infer_session = onnxruntime.InferenceSession(
                                str(augmented_path), options, providers=["CPUExecutionProvider"]
                            )
                            calibrater.collect_data(_Reader(inputs))
                            ranges = calibrater.compute_data()
                            self.assertEqual(set(ranges), tensors)
                            axes = (0, 2) if per_channel else None
                            for name in tensors:
                                minimum, maximum = ranges[name].range_value
                                np.testing.assert_array_equal(minimum, np.min(values[name], axis=axes).reshape(-1))
                                np.testing.assert_array_equal(maximum, np.max(values[name], axis=axes).reshape(-1))

    def test_no_selected_tensors_preserves_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.onnx"
            onnx.save(self._model(18), model_path)
            calibrater = MinMaxCalibrater(
                model_path,
                op_types_to_calibrate=["Conv"],
                augmented_model_path=str(Path(directory) / "augmented.onnx"),
            )
            before = [node.SerializeToString() for node in calibrater.model.graph.node]
            calibrater.augment_graph()
            self.assertEqual([node.SerializeToString() for node in calibrater.model.graph.node], before)
            self.assertEqual([output.name for output in calibrater.model.graph.output], ["out", "leaf", "b", "unused"])


if __name__ == "__main__":
    unittest.main()
