from ..quant_utils import FLOAT8_TYPES
from .direct_q8 import Direct8BitOp, QDQDirect8BitOp


class QResize(Direct8BitOp):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert node.op_type == "Resize"

        # Resize requires opset 11 and does not support FP8 tensors.
        if (
            self.quantizer.opset_version < 11
            or self.quantizer.activation_qType in FLOAT8_TYPES
            or self._get_input_quantized_type() in FLOAT8_TYPES
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

        # if version is less than 11, just keep this node
        if self.quantizer.opset_version < 11:
            return

        # Direct 8bits op
        return super().quantize()
