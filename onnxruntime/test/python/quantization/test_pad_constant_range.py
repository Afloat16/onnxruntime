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


class TestPadConstantRange(unittest.TestCase):
    def _check(
        self,
        qtype,
        source,
        fill,
        *,
        quant_format=QuantFormat.QOperator,
        opset=13,
        mode="constant",
        following_conv=False,
        runtime_values=None,
    ):
        use_axes = opset >= 18
        x = np.array([-1, 0, 1, 2], np.float32).reshape(1, 1, 2, 2)
        feed = {"x": x}
        graph_inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])]
        initializers = [numpy_helper.from_array(np.ones((1, 1, 1, 1), np.float32), "weight")]
        nodes = [helper.make_node("Conv", ["x", "weight"], ["conv_out"], name="conv")]
        pads = [1, 1] if use_axes else [0, 0, 1, 1, 0, 0, 1, 1]
        initializers.append(numpy_helper.from_array(np.array(pads, np.int64), "pads"))
        pad_inputs = ["conv_out", "pads"]
        if source not in ("omitted", "empty"):
            pad_inputs.append("fill")
            value = numpy_helper.from_array(np.array(fill, np.float32), "fill")
            if source in ("initializer", "overridable"):
                initializers.append(value)
            elif source == "constant":
                nodes.append(helper.make_node("Constant", [], ["fill"], value=value))
            if source in ("runtime", "overridable"):
                graph_inputs.append(helper.make_tensor_value_info("fill", TensorProto.FLOAT, []))
                feed["fill"] = np.array(fill, np.float32)
        elif source == "empty" or use_axes:
            pad_inputs.append("")
        if use_axes:
            pad_inputs.append("axes")
            initializers.append(numpy_helper.from_array(np.array([-1], np.int64), "axes"))
        nodes.append(helper.make_node("Pad", pad_inputs, ["pad_out"], name="pad", mode=mode))
        if following_conv:
            nodes.append(helper.make_node("Conv", ["pad_out", "weight"], ["out"], name="after_pad"))
        output_shape = [1, 1, 2 if use_axes else 4, 4]
        model = helper.make_model(
            helper.make_graph(
                nodes,
                "pad_constant_range",
                graph_inputs,
                [
                    helper.make_tensor_value_info(
                        "out" if following_conv else "pad_out", TensorProto.FLOAT, output_shape
                    )
                ],
                initializers,
            ),
            opset_imports=[helper.make_opsetid("", opset)],
            ir_version=7,
        )
        onnx.checker.check_model(model, full_check=True)
        with tempfile.TemporaryDirectory() as directory:
            quant_path = Path(directory) / "quantized.onnx"
            quantize_static(
                model,
                quant_path,
                TestDataFeeds([feed]),
                quant_format=quant_format,
                activation_type=qtype,
                weight_type=QuantType.QInt8,
                extra_options={"ActivationSymmetric": False},
            )
            quantized = onnx.load(quant_path)
            onnx.checker.check_model(quantized, full_check=True)
            feeds = [feed]
            if runtime_values is not None:
                feeds = [{"x": x}, *({"x": x, "fill": np.array(v, np.float32)} for v in runtime_values)]
            optimizations = [ort.GraphOptimizationLevel.ORT_DISABLE_ALL]
            # The existing native layout optimizer mishandles Pad axes before QLinearConv.
            if not (use_axes and following_conv):
                optimizations.append(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
            for optimization in optimizations:
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                options.graph_optimization_level = optimization
                options.log_severity_level = 3
                original = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
                actual = ort.InferenceSession(
                    quantized.SerializeToString(), options, providers=["CPUExecutionProvider"]
                )
                for current_feed in feeds:
                    current_fill = float(current_feed.get("fill", fill))
                    widths = ((0, 0), (0, 0), (0, 0) if use_axes else (1, 1), (1, 1))
                    kwargs = {"constant_values": current_fill} if mode == "constant" else {}
                    expected = np.pad(x, widths, mode=mode, **kwargs)
                    np.testing.assert_array_equal(original.run(None, current_feed)[0], expected)
                    np.testing.assert_allclose(actual.run(None, current_feed)[0], expected, rtol=0, atol=0.06)

            if quant_format == QuantFormat.QOperator:
                inferred = onnx.shape_inference.infer_shapes(quantized)
                types = {
                    v.name: v.type.tensor_type.elem_type
                    for v in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
                }
                pad = next(n for n in quantized.graph.node if n.op_type == "Pad")
                keep_float = source in ("runtime", "overridable") or abs(fill) > 2
                expected_type = TensorProto.FLOAT if keep_float else qtype.tensor_type
                self.assertEqual(types[pad.input[0]], expected_type)
                self.assertEqual(
                    sum(n.op_type == "QLinearConv" for n in quantized.graph.node), 2 if following_conv else 1
                )

    def test_constant_and_runtime_fills(self):
        for qtype, source, fill in itertools.product(
            (QuantType.QInt8, QuantType.QUInt8), ("initializer", "constant", "runtime"), (-10, -1, 0, 1, 10)
        ):
            with self.subTest(qtype=qtype, source=source, fill=fill):
                self._check(qtype, source, fill)

    def test_overridable_fills(self):
        for qtype, fill in itertools.product((QuantType.QInt8, QuantType.QUInt8), (0, 1)):
            with self.subTest(qtype=qtype, fill=fill):
                self._check(qtype, "overridable", fill, runtime_values=(-10, 0, 1, 10))

    def test_axes_and_following_quantized_conv(self):
        for qtype, opset, fill, following in itertools.product(
            (QuantType.QInt8, QuantType.QUInt8), (11, 18, 21), (-10, 10), (False, True)
        ):
            with self.subTest(qtype=qtype, opset=opset, fill=fill, following=following):
                self._check(qtype, "initializer", fill, opset=opset, following_conv=following)

    def test_qdq_controls(self):
        for qtype, source, fill in itertools.product(
            (QuantType.QInt8, QuantType.QUInt8), ("initializer", "constant", "runtime"), (-10, 10)
        ):
            with self.subTest(qtype=qtype, source=source, fill=fill):
                self._check(qtype, source, fill, quant_format=QuantFormat.QDQ)

    def test_other_modes(self):
        for qtype, mode in itertools.product((QuantType.QInt8, QuantType.QUInt8), ("edge", "reflect", "wrap")):
            with self.subTest(qtype=qtype, mode=mode):
                self._check(qtype, "omitted", 0, mode=mode, opset=21)

    def test_omitted_and_empty_fills(self):
        for qtype, source, opset in itertools.product(
            (QuantType.QInt8, QuantType.QUInt8), ("omitted", "empty"), (11, 18)
        ):
            with self.subTest(qtype=qtype, source=source, opset=opset):
                self._check(qtype, source, 0, opset=opset)


if __name__ == "__main__":
    unittest.main()
