"""Init module with compatibility patches for HuggingFace Transformers 5.x."""

try:
    import torch
    import transformers
    import transformers.pytorch_utils
    from transformers import PretrainedConfig, PreTrainedModel

    # Patch 1: find_pruneable_heads_and_indices for transformers >= 4.45 / 5.0
    if not hasattr(transformers.pytorch_utils, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - 1
                mask[head] = 0
            mask = mask.view(-1).contiguous()
            index = torch.arange(len(mask))[mask == 0]
            return heads, index

        transformers.pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    # Patch 2: Default attributes on PretrainedConfig for jina-bert models in transformers 5.x
    for attr, val in [("is_decoder", False), ("add_cross_attention", False), ("is_encoder_decoder", False)]:
        if not hasattr(PretrainedConfig, attr):
            setattr(PretrainedConfig, attr, val)

    # Patch 3: get_extended_attention_mask on PreTrainedModel for jina-bert models in transformers 5.x
    if not hasattr(PreTrainedModel, "get_extended_attention_mask"):
        def get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
            if attention_mask.dim() == 3:
                extended_attention_mask = attention_mask[:, None, :, :]
            elif attention_mask.dim() == 2:
                extended_attention_mask = attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask
            if dtype is None:
                dtype = getattr(self, "dtype", torch.float32)
            extended_attention_mask = extended_attention_mask.to(dtype=dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            return extended_attention_mask

        PreTrainedModel.get_extended_attention_mask = get_extended_attention_mask

    # Patch 4: get_head_mask on PreTrainedModel for jina-bert models in transformers 5.x
    if not hasattr(PreTrainedModel, "get_head_mask"):
        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                if is_attention_chunked:
                    head_mask = head_mask.unsqueeze(-1)
            else:
                head_mask = [None] * num_hidden_layers
            return head_mask

        PreTrainedModel.get_head_mask = get_head_mask

except Exception as e:
    pass
