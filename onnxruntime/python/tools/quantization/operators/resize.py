from .direct_q8 import Direct8BitOp, QDQDirect8BitOp


class QResize(Direct8BitOp):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert node.op_type == "Resize"

        # Crop-and-resize can produce extrapolation values outside the input range.
        # Keep it in float rather than interpreting the float attribute as an integer.
        if self.quantizer.opset_version < 11 or any(
            attr.name == "coordinate_transformation_mode" and attr.s == b"tf_crop_and_resize" for attr in node.attribute
        ):
            super(Direct8BitOp, self).quantize()
            return

        # Direct 8bits op
        return super().quantize()


class QDQResize(QDQDirect8BitOp):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert node.op_type == "Resize"

        # Reusing input quantization parameters can clip extrapolation values.
        if self.quantizer.opset_version < 11 or any(
            attr.name == "coordinate_transformation_mode" and attr.s == b"tf_crop_and_resize" for attr in node.attribute
        ):
            return

        # Direct 8bits op
        return super().quantize()
