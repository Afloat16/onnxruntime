import numpy
import onnx
from onnx import onnx_pb as onnx_proto

from ..quant_utils import TENSOR_NAME_QUANT_SUFFIX, QuantType, attribute_to_kwarg, ms_domain  # noqa: F401
from .base_operator import QuantOperatorBase

"""
    Quantize LSTM
"""


class LSTMQuant(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def _weight_name_prefix(self, weight_name):
        root_quantizer = self.quantizer
        while root_quantizer.parent is not None:
            root_quantizer = root_quantizer.parent
        if root_quantizer.lstm_weight_names is None:
            names = set()
            graphs = [root_quantizer.model.graph()]
            while graphs:
                graph = graphs.pop()
                names.update(tensor.name for tensor in graph.initializer)
                names.update(value.name for value in (*graph.input, *graph.output, *graph.value_info))
                for node in graph.node:
                    names.update(node.input)
                    names.update(node.output)
                    for attr in node.attribute:
                        if attr.type == onnx.AttributeProto.GRAPH:
                            graphs.append(attr.g)
                        elif attr.type == onnx.AttributeProto.GRAPHS:
                            graphs.extend(attr.graphs)
            root_quantizer.lstm_weight_names = names
        names = root_quantizer.lstm_weight_names
        prefix = weight_name + "_lstm"
        candidate = prefix
        suffix = 0
        tensor_suffixes = ("", TENSOR_NAME_QUANT_SUFFIX, "_zero_point", "_scale")
        while any(
            candidate + tensor_suffix in names
            or self.quantizer.find_initializer_in_path(candidate + tensor_suffix)
            or self.quantizer.contains_tensor(candidate + tensor_suffix)
            for tensor_suffix in tensor_suffixes
        ):
            candidate = f"{prefix}_{suffix}"
            suffix += 1
        names.update(candidate + tensor_suffix for tensor_suffix in tensor_suffixes)
        return candidate

    def _quantize_weight(self, weight):
        if weight.name in self.quantizer.lstm_weight_cache:
            return self.quantizer.lstm_weight_cache[weight.name]

        num_dir, gate_size, input_size = weight.dims
        # Reshape only the quantizer's NumPy view. Keep the original initializer
        # and the ordinary quantized-value cache available to other consumers.
        quantized = self.quantizer.quantize_weight_per_channel_impl(
            weight.name,
            onnx_proto.TensorProto.INT8,
            0,
            weight_shape=(num_dir * gate_size, input_size) if self.quantizer.is_per_channel() else None,
            quantized_name_prefix=self._weight_name_prefix(weight.name),
        )
        quantized_weight = self.quantizer.model.get_initializer(quantized[0])
        quantized_array = onnx.numpy_helper.to_array(quantized_weight).reshape(num_dir, gate_size, input_size)
        quantized_weight.CopyFrom(
            onnx.numpy_helper.from_array(numpy.transpose(quantized_array, (0, 2, 1)), quantized[0])
        )
        if self.quantizer.is_per_channel():
            for name in quantized[1:]:
                self.quantizer.model.get_initializer(name).dims[:] = [num_dir, gate_size]
        self.quantizer.lstm_weight_cache[weight.name] = quantized
        return quantized

    def quantize(self):
        """
        parameter node: LSTM node.
        parameter new_nodes_list: List of new nodes created before processing this node.
        return: a list of nodes in topological order that represents quantized Attention node.
        """
        node = self.node
        assert node.op_type == "LSTM"

        if not self.quantizer.is_valid_quantize_weight(node.input[1]) or not self.quantizer.is_valid_quantize_weight(
            node.input[2]
        ):
            super().quantize()
            return

        model = self.quantizer.model
        W = model.get_initializer(node.input[1])  # noqa: N806
        R = model.get_initializer(node.input[2])  # noqa: N806

        if len(W.dims) != 3 or len(R.dims) != 3:
            super().quantize()
            return

        quant_input_weight_tuple = self._quantize_weight(W)
        quant_recurrent_weight_tuple = self._quantize_weight(R)

        inputs = []
        input_len = len(node.input)
        inputs.extend([node.input[0]])
        inputs.extend([quant_input_weight_tuple[0], quant_recurrent_weight_tuple[0]])
        inputs.extend([node.input[3] if input_len > 3 else ""])
        inputs.extend([node.input[4] if input_len > 4 else ""])
        inputs.extend([node.input[5] if input_len > 5 else ""])
        inputs.extend([node.input[6] if input_len > 6 else ""])
        inputs.extend([node.input[7] if input_len > 7 else ""])
        inputs.extend(
            [
                quant_input_weight_tuple[2],
                quant_input_weight_tuple[1],
                quant_recurrent_weight_tuple[2],
                quant_recurrent_weight_tuple[1],
            ]
        )

        kwargs = {}
        for attribute in node.attribute:
            if attribute.name == "layout":
                continue
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain

        quant_lstm_name = "" if not node.name else node.name + "_quant"
        quant_lstm_node = onnx.helper.make_node("DynamicQuantizeLSTM", inputs, node.output, quant_lstm_name, **kwargs)
        self.quantizer.new_nodes.append(quant_lstm_node)

        dequantize_node = self.quantizer._dequantize_value(node.input[0])
        if dequantize_node is not None:
            self.quantizer.new_nodes.append(dequantize_node)
