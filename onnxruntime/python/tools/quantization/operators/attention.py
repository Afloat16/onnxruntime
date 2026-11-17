import onnx
from onnx import onnx_pb as onnx_proto  # noqa: F401

from ..quant_utils import attribute_to_kwarg, ms_domain
from .base_operator import QuantOperatorBase

"""
    Quantize Attention
"""


class AttentionQuant(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def should_quantize(self):
        return self.quantizer.should_quantize_node(self.node)

    def quantize(self):
        """
        parameter node: Attention node.
        parameter new_nodes_list: List of new nodes created before processing this node.
        return: a list of nodes in topological order that represents quantized Attention node.
        """
        node = self.node
        assert node.op_type == "Attention"

        # QAttention represents the contrib operator, requires a projection bias,
        # and has no inputs for attention_bias or past_sequence_length.
        if node.domain != ms_domain or len(node.input) < 3 or not node.input[2] or any(node.input[5:]):
            return super().quantize()

        # These Attention attributes are not part of the QAttention schema.
        for attr in node.attribute:
            if attr.name in ("qkv_hidden_sizes", "rotary_embedding_dim"):
                return super().quantize()

        (
            quantized_input_names,
            zero_point_names,
            scale_names,
            nodes,
        ) = self.quantizer.quantize_activation(node, [0])

        (
            quantized_input_names_weight,
            zero_point_names_weight,
            scale_names_weight,
            nodes_weight,
        ) = self.quantizer.quantize_weight(node, [1], reduce_range=True, op_level_per_channel=True)
        quantized_input_names.extend(quantized_input_names_weight)
        zero_point_names.extend(zero_point_names_weight)
        scale_names.extend(scale_names_weight)
        nodes.extend(nodes_weight)

        if quantized_input_names is None:
            return super().quantize()

        qattention_name = "" if not node.name else node.name + "_quant"

        inputs = []
        inputs.extend(quantized_input_names)
        inputs.extend([node.input[2]])
        inputs.extend(scale_names)
        inputs.extend([node.input[3] if len(node.input) > 3 else ""])
        inputs.extend(zero_point_names)
        inputs.extend([node.input[4] if len(node.input) > 4 else ""])

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain
        qattention_node = onnx.helper.make_node("QAttention", inputs, node.output, qattention_name, **kwargs)
        nodes.append(qattention_node)

        self.quantizer.new_nodes += nodes
