from onnx import TensorProto

from ..quant_utils import TENSOR_NAME_QUANT_SUFFIX, QuantizedValue, QuantizedValueType
from .base_operator import QuantOperatorBase
from .qdq_base_operator import QDQOperatorBase


# For operators that support 8bits operations directly, and output could
# reuse input[0]'s type, zeropoint, scale; For example,Transpose, Reshape, etc.
class Direct8BitOp(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def _get_input_quantized_type(self):
        input_name = self.node.input[0]
        overrides = self.quantizer.tensor_quant_overrides.get(input_name, [{}])[0]
        qtype = overrides["quant_type"].tensor_type if "quant_type" in overrides else self.quantizer.activation_qType
        quantized_value = self.quantizer.find_quantized_value(input_name)
        owner = self.quantizer
        while quantized_value is not None and owner is not None:
            zero_point = owner.model.get_initializer(quantized_value.zp_name)
            if zero_point is not None:
                return zero_point.data_type
            owner = owner.parent
        return qtype

    def quantize(self):
        node = self.node

        if not self.quantizer.force_quantize_no_input_check:
            # Keep backward compatibility
            # Quantize when input[0] is quantized already. Otherwise keep it.
            quantized_input_value = self.quantizer.find_quantized_value(node.input[0])
            if quantized_input_value is None:
                self.quantizer.new_nodes += [node]
                return

            quantized_output_value = QuantizedValue(
                node.output[0],
                node.output[0] + TENSOR_NAME_QUANT_SUFFIX,
                quantized_input_value.scale_name,
                quantized_input_value.zp_name,
                quantized_input_value.value_type,
            )
            self.quantizer.quantized_value_map[node.output[0]] = quantized_output_value

            node.input[0] = quantized_input_value.q_name
            node.output[0] = quantized_output_value.q_name
            self.quantizer.new_nodes += [node]

        else:
            # Force quantize those ops if possible, use exclude node list if this is not you want
            if not self.quantizer.is_float_tensor(node.input[0]):
                super().quantize()
                return
            # DynamicQuantizeLinear only accepts FLOAT. Static FLOAT16 input
            # quantization requires QuantizeLinear/DequantizeLinear opset 19.
            if (
                self.quantizer.find_quantized_value(node.input[0]) is None
                and not self.quantizer.find_initializer_in_path(node.input[0])
                and self.quantizer.get_tensor_type(node.input[0]) != TensorProto.FLOAT
                and (not self.quantizer.static or self.quantizer.opset_version < 19)
            ):
                return super().quantize()

            (
                quantized_input_names,
                zero_point_names,
                scale_names,
                nodes,
            ) = self.quantizer.quantize_activation(node, [0])
            if quantized_input_names is None:
                return super().quantize()

            # Create an entry for output quantized value
            quantized_output_value = QuantizedValue(
                node.output[0],
                node.output[0] + TENSOR_NAME_QUANT_SUFFIX,
                scale_names[0],
                zero_point_names[0],
                QuantizedValueType.Input,
            )
            self.quantizer.quantized_value_map[node.output[0]] = quantized_output_value

            node.input[0] = quantized_input_names[0]
            node.output[0] = quantized_output_value.q_name
            nodes.append(node)

            self.quantizer.new_nodes += nodes


class QDQDirect8BitOp(QDQOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        if self.quantizer.force_quantize_no_input_check:
            self.quantizer.quantize_activation_tensor(self.node.input[0])
            if not self.disable_qdq_for_node_output:
                self.quantizer.quantize_output_same_as_input(self.node.output[0], self.node.input[0], self.node.name)
        elif self.quantizer.is_tensor_quantized(self.node.input[0]) and not self.disable_qdq_for_node_output:
            self.quantizer.quantize_output_same_as_input(self.node.output[0], self.node.input[0], self.node.name)
