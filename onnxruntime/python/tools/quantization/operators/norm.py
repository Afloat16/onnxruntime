# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from .qdq_base_operator import QDQOperatorBase


class QDQNormalization(QDQOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert node.op_type in {"InstanceNormalization", "LayerNormalization", "BatchNormalization"}

        # Input
        self.quantizer.quantize_activation_tensor(node.input[0])

        # Scale
        scale_initializer = self.quantizer.model.get_initializer(node.input[1])
        scale_is_initializer = scale_initializer is not None
        default_axis = 0 if scale_is_initializer and len(scale_initializer.dims) == 1 else 1
        scale_is_per_channel, scale_channel_axis = self.quantizer.is_tensor_per_channel(
            node.input[1], default_axis=default_axis, op_type=node.op_type
        )

        if scale_is_per_channel:
            self.quantizer.quantize_weight_tensor_per_channel(node.input[1], axis=scale_channel_axis)
        elif scale_is_initializer:
            self.quantizer.quantize_weight_tensor(node.input[1])
        else:
            self.quantizer.quantize_activation_tensor(node.input[1])

        # Bias
        if len(node.input) > 2 and node.input[2]:
            self.quantizer.quantize_bias_tensor(node.name, node.input[2], node.input[0], node.input[1])

        # Output
        if not self.disable_qdq_for_node_output:
            for output_name in node.output:
                self.quantizer.quantize_activation_tensor(output_name)
