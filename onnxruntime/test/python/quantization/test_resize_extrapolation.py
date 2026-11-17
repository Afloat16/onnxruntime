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
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static


class TestResizeExtrapolation(unittest.TestCase):
    def _quantize_resize(
        self, quant_format, source, mode, coordinate_mode, extrapolation, runtime_roi, downstream_conv=False
    ):
        values = np.array([-1, 0, 1, 2], dtype=np.float32).reshape(1, 1, 2, 2)
        roi = np.array([0, 0, -0.5, -0.5, 1, 1, 1.5, 1.5], dtype=np.float32)
        inputs = []
        initializers = [numpy_helper.from_array(np.array([1, 1, 3, 3], dtype=np.int64), "sizes")]
        nodes = []
        if source == "constant":
            initializers.append(numpy_helper.from_array(values, "x"))
            inputs.append(helper.make_tensor_value_info("unused", TensorProto.FLOAT, [1]))
            feeds = {"unused": np.zeros(1, dtype=np.float32)}
        else:
            inputs.append(helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2]))
            feeds = {"x": values}
        if source == "conv":
            initializers.append(numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), "w"))
            nodes.append(helper.make_node("Conv", ["x", "w"], ["conv_output"], name="conv"))
        if runtime_roi:
            inputs.append(helper.make_tensor_value_info("roi", TensorProto.FLOAT, [8]))
            feeds["roi"] = roi
        else:
            initializers.append(numpy_helper.from_array(roi, "roi"))
        attrs = {"mode": mode}
        if coordinate_mode is not None:
            attrs["coordinate_transformation_mode"] = coordinate_mode
        if extrapolation is not None:
            attrs["extrapolation_value"] = extrapolation
        nodes.append(
            helper.make_node(
                "Resize",
                ["conv_output" if source == "conv" else "x", "roi", "", "sizes"],
                ["resized" if downstream_conv else "y"],
                name="resize",
                **attrs,
            )
        )
        if downstream_conv:
            initializers.append(numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), "output_w"))
            nodes.append(helper.make_node("Conv", ["resized", "output_w"], ["y"], name="output_conv"))
        model = helper.make_model(
            helper.make_graph(
                nodes,
                "resize_extrapolation",
                inputs,
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3, 3])],
                initializers,
            ),
            opset_imports=[helper.make_opsetid("", 13)],
            ir_version=7,
        )
        onnx.checker.check_model(model, full_check=True)
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "quantized.onnx"
            quantize_static(
                model,
                output_path,
                TestDataFeeds([feeds]),
                quant_format=quant_format,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QUInt8,
                extra_options={"ForceQuantizeNoInputCheck": source != "conv"},
            )
            quantized = onnx.load(output_path)
        onnx.checker.check_model(quantized, full_check=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        expected = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"]).run(
            None, feeds
        )[0]
        for optimization in (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
            with self.subTest(optimization=optimization):
                options.graph_optimization_level = optimization
                actual = ort.InferenceSession(
                    quantized.SerializeToString(), options, providers=["CPUExecutionProvider"]
                ).run(None, feeds)[0]
                np.testing.assert_allclose(actual, expected, rtol=0, atol=0.05 if downstream_conv else 0.02)
        if source == "conv":
            self.assertTrue(
                any(n.op_type == "QLinearConv" for n in quantized.graph.node)
                if quant_format == QuantFormat.QOperator
                else any(n.op_type == "QuantizeLinear" for n in quantized.graph.node)
            )
        return model, quantized, expected

    def test_crop_and_resize_extrapolation(self):
        for quant_format, source, mode, extrapolation, runtime_roi in itertools.product(
            (QuantFormat.QOperator, QuantFormat.QDQ),
            ("conv", "constant", "runtime"),
            ("nearest", "linear"),
            (None, 10.0, -10.0),
            (False, True),
        ):
            with self.subTest(
                quant_format=quant_format,
                source=source,
                mode=mode,
                extrapolation=extrapolation,
                runtime_roi=runtime_roi,
            ):
                original, quantized, expected = self._quantize_resize(
                    quant_format, source, mode, "tf_crop_and_resize", extrapolation, runtime_roi
                )
                resize = next(n for n in quantized.graph.node if n.name == "resize")
                self.assertEqual(list(resize.input[1:]), ["roi", "", "sizes"])
                self.assertEqual(resize.output[0], "y")
                self.assertEqual(list(resize.attribute), list(original.graph.node[-1].attribute))
                fill = 0.0 if extrapolation is None else extrapolation
                np.testing.assert_array_equal(expected[0, 0, (0, 2), :], np.full((2, 3), fill))

    def test_other_coordinate_modes_remain_quantized(self):
        for quant_format, source, mode, coordinate_mode in itertools.product(
            (QuantFormat.QOperator, QuantFormat.QDQ),
            ("conv", "constant"),
            ("nearest", "linear"),
            (None, "asymmetric"),
        ):
            with self.subTest(quant_format=quant_format, source=source, mode=mode, coordinate_mode=coordinate_mode):
                _, quantized, _ = self._quantize_resize(quant_format, source, mode, coordinate_mode, 10.0, False)
                resize = next(n for n in quantized.graph.node if n.name == "resize")
                self.assertNotEqual(resize.output[0], "y")
                output_dq = next(n for n in quantized.graph.node if n.output[0] == "y")
                self.assertEqual(output_dq.op_type, "DequantizeLinear")

    def test_crop_and_resize_before_quantized_consumer(self):
        for quant_format, mode, extrapolation in itertools.product(
            (QuantFormat.QOperator, QuantFormat.QDQ), ("nearest", "linear"), (None, 10.0, -10.0)
        ):
            with self.subTest(quant_format=quant_format, mode=mode, extrapolation=extrapolation):
                _, quantized, _ = self._quantize_resize(
                    quant_format, "conv", mode, "tf_crop_and_resize", extrapolation, True, downstream_conv=True
                )
                output_conv = next(n for n in quantized.graph.node if n.name.startswith("output_conv"))
                if quant_format == QuantFormat.QOperator:
                    self.assertEqual(output_conv.op_type, "QLinearConv")
                else:
                    self.assertEqual(output_conv.op_type, "Conv")
                    producer = next(n for n in quantized.graph.node if output_conv.input[0] in n.output)
                    self.assertEqual(producer.op_type, "DequantizeLinear")


if __name__ == "__main__":
    unittest.main()
