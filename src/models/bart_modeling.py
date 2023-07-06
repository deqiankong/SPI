# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch BART model. Taken from https://github.com/huggingface/transformers/blob/v4.20.0/src/transformers/models/bart/modeling_bart.py"""
import time
import copy
import math
import random
import warnings
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import (
    BartLearnedPositionalEmbedding,
    BartPretrainedModel,
    # BartEncoder,
    BartLearnedPositionalEmbedding, 
    BartEncoderLayer,
    BartDecoder,
    BartClassificationHead,
    BartModel,
    BartForConditionalGeneration,
)

from src.models.latent_utils import LatentModel, AverageSelfAttention, GELU, convert_mask
from src.models.modeling_outputs import (
    Seq2SeqModelOutputWithLatent,
    Seq2SeqLMOutputWithLatent,
    BaseEncoderOutputWithLatent
)

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/bart-base"
_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 768]

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "valhalla/bart-large-sst2"
_SEQ_CLASS_EXPECTED_LOSS = 0.0
_SEQ_CLASS_EXPECTED_OUTPUT = "'POSITIVE'"

# QuestionAsnwering docstring
_CHECKPOINT_FOR_QA = "valhalla/bart-large-finetuned-squadv1"
_QA_EXPECTED_LOSS = 0.59
_QA_EXPECTED_OUTPUT = "' nice puppet'"

BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-large",
    # see all BART models at https://huggingface.co/models?filter=bart
]


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(float("-inf")))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class BartAttentionWithLV(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper with latent variable in self-attention"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        latent_context: Optional[torch.Tensor] = None,  # ours
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)


        # ours: concate z with key and value if is_self_attn
        if not is_cross_attention and latent_context is not None:
            latent_k, latent_v = latent_context
            latent_k = self._shape(latent_k, 1, bsz).view(*proj_shape)
            latent_v = self._shape(latent_v, 1, bsz).view(*proj_shape)

            key_states   = torch.cat([latent_k, key_states], dim=1)
            value_states = torch.cat([latent_v, value_states], dim=1)
        

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            attn_src_len = src_len - 1 if (not is_cross_attention and latent_context is not None) else src_len
            if attention_mask.size() != (bsz, 1, tgt_len, attn_src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, attn_src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_cross_attention and latent_context is not None:
                attn_weights[:,:,:,1:] = attn_weights[:,:,:,1:] + attention_mask
            else:
                attn_weights += attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value



class BartDecoderLayerWithLV(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttentionWithLV(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttentionWithLV(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        latent_context: Optional[torch.Tensor] = None,  # ours
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            latent_context=latent_context, # ours
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs



        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )



class BartDecoderWithLV(BartPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BartDecoderLayerWithLV`]
    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([BartDecoderLayerWithLV(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(inputs_embeds.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        latent_context: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.
                Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.
                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of
                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                    latent_context,   # ours
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    latent_context=latent_context,   # ours
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class BartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BartEncoderLayer`].
    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.
                Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.
                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape_orig = input_ids.size()
            input_ids = input_ids.view(-1, input_shape_orig[-1])
            attention_mask = attention_mask.view(-1, input_shape_orig[-1])
            # Ours update input size
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if len(input_shape_orig) == 3:
            hidden_states = hidden_states.view(input_shape_orig[0], input_shape_orig[1], input_shape_orig[2], -1)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


@add_start_docstrings(
    "The bare BART Model with EBM module outputting raw hidden-states without any specific head on top."
)
class BartModelWithLV(BartPretrainedModel, LatentModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        if hasattr(config, "attend_latent_w_self") and self.config.attend_latent_w_self:
            self.decoder = BartDecoderWithLV(config, self.shared)
        else:
            self.decoder = BartDecoder(config, self.shared)

        # parameters for latent variables
        self.average_self_attn = AverageSelfAttention(config.d_model)

        self.latent_model  = nn.Linear(config.d_model, 2 * config.latent_size, bias=False)
        self.latent_size   = config.latent_size
        self.sample_latent = config.sample_latent
        self.kn_selector   = config.kn_selector
        self.target_kl     = config.target_kl
        self.attend_latent = config.attend_latent
        self.attend_latent_w_self = config.attend_latent_w_self if hasattr(config, "attend_latent_w_self") else False
        self.fuse_z        = config.fuse_z
        self.use_feature   = config.use_feature
        self.use_z_for_cls = config.use_z_for_cls

        if self.kn_selector == "linear":
            # knowledge prediction
            self.classification_head = BartClassificationHead(
                config.d_model,
                config.d_model,
                1,  # config.num_labels,
                config.classifier_dropout,
            )
        else:
            raise NotImplementedError
        
        if self.attend_latent_w_self:
            self.self_attn_transform = nn.Linear(config.d_model, 2 * config.d_model, bias=False)

        if self.fuse_z == "concate" or self.fuse_z == "residual":
            self.proj_hz = nn.Linear(config.d_model+config.latent_size, config.d_model, bias=False)
        elif self.fuse_z is None:
            # Nothing will happen
            pass
        else:
            raise NotImplementedError
            
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    
    def transform_z_to_latent_context(self, z):
        # Modified from 
        # https://github.com/lemuria-wchen/DialogVED/blob/23177dfeecff831c1417fc661e75f8947fffba5d/src/model.py#L1118
        # Transform to latent key and value
        # latent_context: [bsz, 1, latent_dim]
        latent_context = self.self_attn_transform(z).chunk(2, -1)
        # chunk -> Convert latent context from [bsz, 1, 2 * hdim] to 2 * [bsz, 1, hdim]
        return latent_context

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        knowledge_mask: Optional[torch.Tensor] = None,
        ctx_outputs: Optional[torch.FloatTensor] = None,
        ctx_attention_mask: Optional[torch.Tensor] = None,
        decoder_shapes: Optional[tuple] = None,
        decoder_cls_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Seq2SeqModelOutputWithLatent]:

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hdim = encoder_outputs[0].shape[-1]
        # Ours: Get sentence representation for Kn selection & Posterior inference
        # Get the average representation over the whole sequence
        if self.use_feature == "kn":
            sentence_states, _ = self.average_self_attn(encoder_outputs[0], attention_mask=knowledge_mask)
        elif self.use_feature == "dial":
            sentence_states, _ = self.average_self_attn(encoder_outputs[0],
                                                        attention_mask=attention_mask - knowledge_mask)
        else:
            sentence_states, _ = self.average_self_attn(encoder_outputs[0], attention_mask=attention_mask)

        if self.use_z_for_cls:
            # Ours: Forward for Prior Model
            # Latent model forward here
            prior_latent_states = self.latent_model(sentence_states)
            prior_latent_states, prior_mean, prior_logvar = self.vae_latent_forward(prior_latent_states)
            z = prior_mean

        # Ours: Knowledge selection based on sentence represention of inputs (H, K)
        # 1. knowledge prediction
        # 2. mutual information
        # 3. inner product
        if self.kn_selector == "linear":
            classification_logits = self.classification_head(z) if self.use_z_for_cls else self.classification_head(sentence_states)
        elif self.kn_selector == "pmi":
            # after langevin, we get posterior z
            # use this posterior latent to choose between knowledge candidates
            pass
            raise NotImplementedError
        elif self.kn_selector == "mips":
            pass
            raise NotImplementedError
        else:
            raise NotImplementedError
        # Ours: add shape record here
        bsz, n_kns, seq_len = decoder_shapes
        # after we obtain the logits/score for each knowledge
        # we select the most possible knoweldge with argmax
        # assum the dimention of logits is (bsz * num_kn) x 1 -> bsz x num_kn x 1
        classification_logits = classification_logits.view(-1, n_kns)
        if decoder_cls_mask is not None:
            # apply cls mask to classification logits to manually set the elements as -inf 
            # where the corresponding z from all padding seqs
            # This case only happens when batch size > 1
            decoder_cls_mask = convert_mask(decoder_cls_mask, dtype=classification_logits.dtype)
            classification_logits += decoder_cls_mask
        kn_select_index = torch.argmax(classification_logits.unsqueeze(-1), dim=1).unsqueeze(-1)  # bsz x 1 x 1

        if not self.use_z_for_cls:
            # Ours: Forward for Prior Model
            # Latent model forward here
            prior_latent_states = self.latent_model(sentence_states)
            prior_latent_states, prior_mean, prior_logvar = self.vae_latent_forward(prior_latent_states)
            z = prior_mean

        # Select knowledge based on `kn_select_index`
        # Get the index of the selected knowledge
        enc_kn_select_index = kn_select_index.expand(-1, -1, seq_len)

        # If True -> Baseline: not to add knowledge into decoder
        # change attention mask to avoid decoder from attending to knowledge component
        # If False -> Ours: take the encoder output as the decoder inputs
        # select the dimension of the tensor corresponding to the selected knowledge
        if ctx_attention_mask is not None:
            # The original shape of attention mask: (bsz * 1) x seq_len -> bsz x seq_len
            attention_mask = ctx_attention_mask
            attention_mask = attention_mask.view(-1, seq_len)

        else:
            # The original shape of attention mask: (bsz * num_kn) x seq_len -> bsz x num_kn x seq_len
            attention_mask = attention_mask.view(-1, n_kns, seq_len)
            attention_mask = torch.gather(attention_mask, dim=1, index=enc_kn_select_index).squeeze(1)

        if ctx_outputs is not None:
            encoder_hidden_states = ctx_outputs
            # The original shape of hidden states: (bsz * 1) x seq_len x hdim -> bsz x seq_len x hdim 
            encoder_hidden_states = encoder_hidden_states.view(-1, seq_len, hdim)
        else:
            encoder_hidden_states = encoder_outputs[0]
            # The original shape of hidden states: (bsz * num_kn) x seq_len x hdim -> bsz x num_kn x seq_len x hdim 
            encoder_hidden_states = encoder_hidden_states.view(-1, n_kns, seq_len, hdim)
            encoder_hidden_states = torch.gather(encoder_hidden_states, dim=1,
                                                 index=enc_kn_select_index.unsqueeze(-1).expand(-1, -1, -1,
                                                                                                hdim)).squeeze(1)

        # Also select the corresponding z
        # The original shape of z: (bsz * num_kn) x hdim -> bsz x num_kn x hdim
        z = z.view(-1, n_kns, self.latent_size)
        z = torch.gather(z, dim=1, index=kn_select_index.expand(-1, -1, self.latent_size))


        # Concate z into the encoder hidden states
        # Here encoder_hidden_states.size -> bsz x seq_len x hdim
        # latent_z -> bsz x hdim
        # We are concatenating z into the second dim of the encoder hidden states
        # z = z.unsqueeze(1)
        if self.attend_latent:
            z_mask = torch.ones(attention_mask.shape[0], device=attention_mask.device,
                                dtype=attention_mask.dtype).unsqueeze(1)
            encoder_hidden_states = torch.cat((z, encoder_hidden_states), dim=1)
            attention_mask = torch.cat((z_mask, attention_mask), dim=1)
        if self.attend_latent_w_self:
            latent_context = self.transform_z_to_latent_context(z)

        if self.fuse_z is not None:
            z_expand = z.expand(-1, encoder_hidden_states.shape[1], -1)
            hz = torch.cat([encoder_hidden_states, z_expand], dim=-1)
            hz = self.proj_hz(hz)
            if self.fuse_z == "residual":
                encoder_hidden_states += hz
            else:
                encoder_hidden_states = hz
            
        # print("classification_logits", classification_logits.shape)
        # print("encoder_hidden_states", encoder_hidden_states.shape)
        # print("z", z.shape)
        # print("extended_encoder_hidden_states", extended_encoder_hidden_states.shape)
        # print("extended_attention_mask", extended_attention_mask.shape)
        # print("decoder_input_ids", decoder_input_ids.shape)
        # print("decoder_attention_mask", decoder_attention_mask.shape if decoder_attention_mask is not None else None)
        # print("decoder_head_mask", decoder_head_mask.shape if decoder_head_mask is not None else None)
        # print("cross_attn_head_mask", cross_attn_head_mask.shape if cross_attn_head_mask is not None else None)
        # input()

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_input_dict = {
            "input_ids": decoder_input_ids,
            "attention_mask": decoder_attention_mask,
            "encoder_hidden_states": encoder_hidden_states, # encoder_outputs[0],
            "encoder_attention_mask": attention_mask, # attention_mask,
            "head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "past_key_values": past_key_values,
            "inputs_embeds": decoder_inputs_embeds,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
        if self.attend_latent_w_self:
            decoder_input_dict.update({"latent_context": latent_context}) 
        decoder_outputs = self.decoder(**decoder_input_dict)            

        kl_loss = None
        if not return_dict:
            return (kl_loss, classification_logits,) + decoder_outputs + encoder_outputs

        return Seq2SeqModelOutputWithLatent(
            kl_loss=kl_loss,
            classification_logits=classification_logits,
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for summarization."
)
class BartForConditionalGenerationWithLV(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModelWithLV(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.cls_ratio = config.cls_ratio
        self.no_kn_decode = config.no_kn_decode if hasattr(config, "no_kn_decode") else False

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        classification_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        knowledge_mask: Optional[torch.Tensor] = None,
        decoder_knowledge_mask: Optional[torch.Tensor] = None,
        ctx_input_ids: Optional[torch.LongTensor] = None,
        ctx_attention_mask: Optional[torch.Tensor] = None,
        decoder_ctx_input_ids: Optional[torch.LongTensor] = None,
        decoder_ctx_attention_mask: Optional[torch.Tensor] = None,
        decoder_shapes: Optional[tuple] = None,
        vae_kl_weight: Optional[float] = None,
        decoder_cls_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Seq2SeqLMOutputWithLatent]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        bsz, n_kns, seq_len = decoder_shapes

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if ctx_input_ids is not None and (self.no_kn_decode or self.model.learn_prior):
            assert ctx_input_ids is not None, "ctx_input_ids is None :("

            ctx_outputs = self.model.encoder(
                input_ids=ctx_input_ids,
                attention_mask=ctx_attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            ctx_outputs = ctx_outputs[0]
        else:
            ctx_outputs = None

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            knowledge_mask=knowledge_mask,
            ctx_outputs=ctx_outputs,
            ctx_attention_mask=ctx_attention_mask,
            decoder_shapes=decoder_shapes,
            decoder_cls_mask=decoder_cls_mask,
        )
        lm_logits = self.lm_head(outputs[2]) + self.final_logits_bias
        classification_logits = outputs[1]

        total_loss = None
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            total_loss = masked_lm_loss
            masked_lm_loss = masked_lm_loss.item()

        cls_loss = None
        if classification_labels is not None:
            cls_loss_fct = CrossEntropyLoss()
            cls_loss = cls_loss_fct(classification_logits.view(bsz, -1), classification_labels.view(-1))
            total_loss = total_loss + self.cls_ratio * cls_loss if total_loss is not None else self.cls_ratio * cls_loss
            cls_loss = cls_loss.item()

        kl_loss = outputs[0]
        if kl_loss is not None:
            total_loss = total_loss + vae_kl_weight * kl_loss if total_loss is not None else vae_kl_weight * kl_loss
            kl_loss = kl_loss.item()

        if not return_dict:
            output = (lm_logits, classification_logits,) + outputs[3:]
            return ((total_loss, masked_lm_loss, cls_loss, kl_loss) + output) if total_loss is not None else output

        # print("total loss", total_loss.item(), "lm loss", masked_lm_loss.item(), "cls loss", cls_loss.item(), "kl loss", kl_loss.item())
        # input()
        return Seq2SeqLMOutputWithLatent(
            loss=total_loss,
            lm_loss=masked_lm_loss,
            cls_loss=cls_loss,
            kl_loss=kl_loss,
            logits=lm_logits,
            classification_logits=classification_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def get_ignore_keys(self):
        return ["loss", "lm_loss", "cls_loss", "kl_loss", \
                "past_key_values", "decoder_hidden_states", "decoder_attentions", "cross_attentions", \
                "encoder_last_hidden_state", "encoder_hidden_states", "encoder_attentions"]

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        knowledge_mask = kwargs["decoder_knowledge_mask"]
        ctx_input_ids = kwargs["decoder_ctx_input_ids"] if "decoder_ctx_input_ids" in kwargs else None
        ctx_attention_mask = kwargs["decoder_ctx_attention_mask"] if "decoder_ctx_attention_mask" in kwargs else None
        expand_size = attention_mask.shape[0] // knowledge_mask.shape[0]
        if expand_size > 1:
            # beam search: need to expand the knowledge mask
            expanded_return_idx = (
                torch.arange(knowledge_mask.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(
                    attention_mask.device)
            )
            knowledge_mask = knowledge_mask.index_select(0, expanded_return_idx)

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "decoder_shapes": kwargs["decoder_shapes"],
            "knowledge_mask": knowledge_mask,
            "ctx_input_ids": ctx_input_ids,
            "ctx_attention_mask": ctx_attention_mask,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


class BartModelWithLangevin(BartPretrainedModel, LatentModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        if hasattr(config, "attend_latent_w_self") and self.config.attend_latent_w_self:
            self.decoder = BartDecoderWithLV(config, self.shared)
        else:
            self.decoder = BartDecoder(config, self.shared)

        # parameters for latent variables
        self.average_self_attn = AverageSelfAttention(config.d_model)

        self.latent_model  = nn.Linear(config.d_model, 2 * config.latent_size, bias=False)
        self.kn_selector   = config.kn_selector
        self.target_kl     = config.target_kl
        self.attend_latent = config.attend_latent
        self.attend_latent_w_self = config.attend_latent_w_self
        self.use_feature   = config.use_feature
        self.use_z_for_cls = config.use_z_for_cls
        self.top_k_kn      = config.top_k_kn
        self.oracle        = config.oracle

        assert self.top_k_kn >= 1

        if self.kn_selector == "linear":
            # knowledge prediction
            self.classification_head = BartClassificationHead(
                config.d_model,
                config.d_model,
                1,  # config.num_labels,
                config.classifier_dropout,
            )
        else:
            raise NotImplementedError

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def encoder_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        knowledge_mask: Optional[torch.Tensor] = None,
        ctx_outputs: Optional[torch.FloatTensor] = None,
        ctx_attention_mask: Optional[torch.Tensor] = None,
        decoder_shapes: Optional[tuple] = None,
        decoder_cls_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Seq2SeqModelOutputWithLatent]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # Ours: add shape record here
        bsz, n_kns, seq_len = decoder_shapes
        hdim = encoder_outputs[0].shape[-1]
        encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.view(-1, seq_len, hdim)
        attention_mask = attention_mask.view(-1, seq_len)
        knowledge_mask = knowledge_mask.view(-1, seq_len)
        # Ours: Get sentence representation for Kn selection & Posterior inference
        # Get the average representation over the whole sequence
        if self.use_feature == "kn":
            sentence_states, _ = self.average_self_attn(encoder_outputs[0], attention_mask=knowledge_mask)
        elif self.use_feature == "dial":
            sentence_states, _ = self.average_self_attn(encoder_outputs[0],
                                                        attention_mask=attention_mask - knowledge_mask)
        else:
            sentence_states, _ = self.average_self_attn(encoder_outputs[0], attention_mask=attention_mask)
        

        # Ours: Knowledge selection based on sentence represention of inputs (H, K)
        # 1. knowledge prediction
        # 2. mutual information
        # 3. inner product
        if self.kn_selector == "linear":
            classification_logits =  self.classification_head(sentence_states)
        elif self.kn_selector == "pmi":
            # after langevin, we get posterior z
            # use this posterior latent to choose between knowledge candidates
            pass
            raise NotImplementedError
        elif self.kn_selector == "mips":
            pass
            raise NotImplementedError
        else:
            raise NotImplementedError
        # after we obtain the logits/score for each knowledge
        # we select the most possible knoweldge with argmax
        # assum the dimention of logits is (bsz * num_kn) x 1 -> bsz x num_kn x 1
        classification_logits = classification_logits.view(-1, n_kns)
        if decoder_cls_mask is not None:
            # apply cls mask to classification logits to manually set the elements as -inf 
            # where the corresponding z from all padding seqs
            # This case only happens when batch size > 1
            decoder_cls_mask = convert_mask(decoder_cls_mask, dtype=classification_logits.dtype)
            classification_logits += decoder_cls_mask
        # kn_select_index = torch.argmax(classification_logits.unsqueeze(-1), dim=1).unsqueeze(-1)  # bsz x 1 x 1
        top_k = min(self.top_k_kn, n_kns) if self.training else 1
        kn_select_index = torch.topk(classification_logits, top_k, dim=1).indices.unsqueeze(-1)  # bsz x top_k x 1
        if self.oracle:
            # TODO: new code for oracle
            # print("kn_select_index", kn_select_index, kn_select_index.shape)
            new_kn_select_index = torch.tensor([[[0]]], dtype=kn_select_index.dtype, device=kn_select_index.device)
            # print("new_kn_select_index", new_kn_select_index, new_kn_select_index.shape)
            # input()
            kn_select_index = new_kn_select_index
            # TODO: new code for oracle - UNTIL HERE

        # Ours: Forward for Prior Model
        # Latent model forward here
        prior_latent_states = self.latent_model(sentence_states)
        prior_latent_states, prior_mean, prior_logvar = self.vae_latent_forward(prior_latent_states)
        z = prior_mean

        if top_k > 1 and decoder_cls_mask is not None:
            decoder_cls_mask = torch.gather(decoder_cls_mask, dim=1, index=kn_select_index.squeeze(-1))
        else:
            decoder_cls_mask = None
        # Select knowledge based on `kn_select_index`
        # Get the index of the selected knowledge
        enc_kn_select_index = kn_select_index.expand(-1, -1, seq_len)

        # If True -> Baseline: not to add knowledge into decoder
        # change attention mask to avoid decoder from attending to knowledge component
        # If False -> Ours: take the encoder output as the decoder inputs
        # select the dimension of the tensor corresponding to the selected knowledge
        if ctx_attention_mask is not None:
            # The original shape of attention mask: (bsz * 1) x seq_len -> bsz x seq_len
            attention_mask = ctx_attention_mask
            attention_mask = attention_mask.view(-1, seq_len)

        else:
            # The original shape of attention mask: (bsz * num_kn) x seq_len -> bsz x num_kn x seq_len
            attention_mask = attention_mask.view(-1, n_kns, seq_len)
            attention_mask = torch.gather(attention_mask, dim=1, index=enc_kn_select_index).squeeze(1)

        if ctx_outputs is not None:
            encoder_hidden_states = ctx_outputs
            # The original shape of hidden states: (bsz * 1) x seq_len x hdim -> bsz x seq_len x hdim 
            encoder_hidden_states = encoder_hidden_states.view(-1, seq_len, hdim)
        else:
            encoder_hidden_states = encoder_outputs[0]
            # The original shape of hidden states: (bsz * num_kn) x seq_len x hdim -> bsz x num_kn x seq_len x hdim 
            encoder_hidden_states = encoder_hidden_states.view(-1, n_kns, seq_len, hdim)
            encoder_hidden_states = torch.gather(encoder_hidden_states, dim=1,
                                                 index=enc_kn_select_index.unsqueeze(-1).expand(-1, -1, -1,
                                                                                                hdim)).squeeze(1)

        # Also select the corresponding z
        # The original shape of z: (bsz * num_kn) x hdim -> bsz x num_kn x hdim
        z = z.view(-1, n_kns, hdim)
        z = torch.gather(z, dim=1, index=kn_select_index.expand(-1, -1, hdim))

        kl_loss = None
        if not return_dict:
            return (kl_loss, classification_logits, z, attention_mask, encoder_hidden_states,) + encoder_outputs[1:]
        return BaseEncoderOutputWithLatent(
            kl_loss=kl_loss,
            classification_logits=classification_logits,
            z=z,
            attention_mask=attention_mask,
            classification_idx=kn_select_index,
            classification_mask=decoder_cls_mask,
            last_hidden_state=encoder_hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def decoder_forward(
        self,
        input_ids: torch.LongTensor = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutputWithLatent]:

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Ours
        # Get z, encoder attention mask, and encoder hidden states
        z = encoder_outputs[2]
        attention_mask = encoder_outputs[3]
        encoder_hidden_states = encoder_outputs[6]

        # Concate z into the encoder hidden states
        # Here encoder_hidden_states.size -> bsz x seq_len x hdim
        # latent_z -> bsz x hdim
        # We are concatenating z into the second dim of the encoder hidden states
        # z = z.unsqueeze(1)
        if self.attend_latent:
            z_mask = torch.ones(attention_mask.shape[0], device=attention_mask.device,
                                dtype=attention_mask.dtype).unsqueeze(1)
            encoder_hidden_states = torch.cat((z, encoder_hidden_states), dim=1)
            attention_mask = torch.cat((z_mask, attention_mask), dim=1)

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,  # encoder_outputs[0],
            encoder_attention_mask=attention_mask,  # attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        kl_loss = encoder_outputs[0]
        classification_logits = encoder_outputs[1]
        if not return_dict:
            return (kl_loss, classification_logits,) + decoder_outputs + encoder_outputs[6:]

        return Seq2SeqModelOutputWithLatent(
            kl_loss=kl_loss,
            classification_logits=classification_logits,
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class BartForConditionalGenerationWithLangvegin(BartForConditionalGenerationWithLV, LatentModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModelWithLangevin(config)

        self.sample_latent  = config.sample_latent
        self.g_l_steps      = config.g_l_steps
        self.g_l_step_size  = config.g_l_step_size
        self.verbose        = config.verbose
        self.remove_noise   = config.remove_noise
        self.add_z_mse      = config.add_z_mse
        self.gen_with_noise = config.gen_with_noise
        self.pseudo_confidence = config.pseudo_confidence
        self.pseudo_label_only = config.pseudo_label_only
        self.random_choice  = config.random_choice
        self.categorical_prior = config.categorical_prior

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        classification_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        knowledge_mask: Optional[torch.Tensor] = None,
        decoder_knowledge_mask: Optional[torch.Tensor] = None,
        ctx_input_ids: Optional[torch.LongTensor] = None,
        ctx_attention_mask: Optional[torch.Tensor] = None,
        decoder_ctx_input_ids: Optional[torch.LongTensor] = None,
        decoder_ctx_attention_mask: Optional[torch.Tensor] = None,
        decoder_shapes: Optional[tuple] = None,
        vae_kl_weight: Optional[float] = 1.0,
        decoder_cls_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Seq2SeqLMOutputWithLatent]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        bsz, n_kns, seq_len = decoder_shapes

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if ctx_input_ids is not None and (self.no_kn_decode or self.model.learn_prior):
            assert ctx_input_ids is not None, "ctx_input_ids is None :("

            ctx_outputs = self.model.encoder(
                input_ids=ctx_input_ids,
                attention_mask=ctx_attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            ctx_outputs = ctx_outputs[0]
        else:
            ctx_outputs = None

        encoder_outputs = self.model.encoder_forward(
            input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            knowledge_mask=knowledge_mask,
            ctx_outputs=ctx_outputs,
            ctx_attention_mask=ctx_attention_mask,
            decoder_shapes=decoder_shapes,
            decoder_cls_mask=decoder_cls_mask,
        )


        def top_k_selection(encoder_outputs, labels, random=False):
            z = encoder_outputs[2].unsqueeze(2)
            attention_mask = encoder_outputs[3]
            encoder_hidden_states = encoder_outputs[6]
            # print("TopK: z                      |", z.shape)
            # print("TopK: attention_mask         |", attention_mask.shape)
            # print("TopK: encoder_hidden_states  |", encoder_hidden_states.shape)

            bsz, top_k, hdim = z.shape[0], z.shape[1], z.shape[3]
            expanded_decoder_input_ids = decoder_input_ids.unsqueeze(1).expand(-1, top_k, -1)

            encoder_outputs.z = z.view(bsz*top_k, -1, hdim)
            encoder_outputs.attention_mask = attention_mask.view(bsz*top_k, -1)
            encoder_outputs.last_hidden_state = encoder_hidden_states.view(bsz*top_k, -1, hdim)
            expanded_decoder_input_ids = expanded_decoder_input_ids.reshape(bsz*top_k, -1)
            ce_labels = labels.unsqueeze(1).expand(-1, top_k, -1).reshape(bsz*top_k, -1)

            # print("TopK: decoder_input_ids      |", decoder_input_ids.shape)
            # print("TopK: expanded_input_ids     |", expanded_decoder_input_ids.shape)
            # print("TopK: decoder_attention_mask |", decoder_attention_mask)
            # print("TopK: decoder_head_mask      |", decoder_head_mask)
            # print("TopK: decoder_inputs_embeds  |", decoder_inputs_embeds)

            self.model.eval()
            outputs = self.model.decoder_forward(
                input_ids,
                decoder_input_ids=expanded_decoder_input_ids,
                encoder_outputs=encoder_outputs,  # put all the encoder outputs to the decoder
                decoder_attention_mask=decoder_attention_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            self.model.train()
            lm_logits = self.lm_head(outputs[2]) + self.final_logits_bias

            # print("TopK: logits               |", lm_logits.shape)
            # print("TopK: ce_labels            |", ce_labels.shape)

            # Compute loss with topk candidates
            loss_fct = CrossEntropyLoss(reduction='none')
            masked_lm_loss = loss_fct(lm_logits.transpose(1,2), ce_labels).detach()
            masked_lm_loss = torch.mean(masked_lm_loss, dim=1, keepdim=True).view(bsz, top_k)
            if encoder_outputs.classification_mask is not None:
                # print("TopK: lm_loss before       |", masked_lm_loss)
                masked_lm_loss -= encoder_outputs.classification_mask
                # print("TopK: cls_mask             |", encoder_outputs.classification_mask)
            
            neg_likelihood = masked_lm_loss.detach()
            if self.categorical_prior:
                classification_logits = encoder_outputs.classification_logits.detach()
                neg_likelihood -= classification_logits
            if random:
                probs = nn.functional.softmax(neg_likelihood * -1, dim=1) 
                # print("neg_likelihood |", neg_likelihood, neg_likelihood.shape)
                # print("probs          |", probs, probs.shape)
                min_ce_index = probs.multinomial(num_samples=1, replacement=True).unsqueeze(-1)
            else:
                min_ce_index = torch.argmin(neg_likelihood, dim=1, keepdim=True).unsqueeze(-1)

            classification_idx  = encoder_outputs.classification_idx
            # print("TopK: lm_loss              |", masked_lm_loss)
            # print("TopK: lm_loss              |", masked_lm_loss.shape)
            # print("TopK: min_ce_index         |", min_ce_index, min_ce_index.shape)
            # print("TopK: argmin               |", torch.argmin(neg_likelihood, dim=1, keepdim=True).unsqueeze(-1))
            # print("TopK: classification_idx   |", classification_idx, classification_idx.shape)

            # Select z based on ce loss
            cls_min_ce_index = min_ce_index.expand(-1, -1, attention_mask.shape[-1])
            attention_mask = torch.gather(attention_mask, dim=1, index=cls_min_ce_index).squeeze(1)
            encoder_hidden_states = torch.gather(encoder_hidden_states, dim=1,
                                                 index=cls_min_ce_index.unsqueeze(-1).expand(-1, -1, -1,
                                                                                             hdim)).squeeze(1)
            pseudo_labels = torch.gather(classification_idx, dim=1, index=min_ce_index).squeeze(1)
            z = torch.gather(z, dim=1, index=min_ce_index.unsqueeze(-1).expand(-1, -1, -1, hdim)).squeeze(1)

            # print("TopK: z                     |", z.shape)
            # print("TopK: attention_mask        |", attention_mask.shape)
            # print("TopK: encoder_hidden_states |", encoder_hidden_states.shape)
            # print("TopK: pseudo_labels         |", pseudo_labels, pseudo_labels.shape)
            # print("-"*80)
            # input()

            encoder_outputs.z                 = z
            encoder_outputs.attention_mask    = attention_mask
            encoder_outputs.last_hidden_state = encoder_hidden_states
            return encoder_outputs, pseudo_labels


        # def random_selection(encoder_outputs):
        #     classification_logits = encoder_outputs[1]
        #     z = encoder_outputs[2].unsqueeze(2)
        #     attention_mask = encoder_outputs[3]
        #     encoder_hidden_states = encoder_outputs[6]

        #     bsz, top_k, hdim = z.shape[0], z.shape[1], z.shape[3]
        #     classification_idx  = encoder_outputs.classification_idx
        #     # print("classification_idx     |", classification_idx.shape)
        #     # print("classification_logits  |", classification_logits.shape)
        #     probs = torch.gather(classification_logits, dim=1, index=classification_idx.squeeze(-1)).detach()
        #     probs = nn.functional.softmax(probs, dim=1)
        #     min_ce_index = probs.multinomial(num_samples=1, replacement=True).unsqueeze(-1)

        #     # print("TopK: probs                |", probs)
        #     # print("TopK: min_ce_index         |", min_ce_index, min_ce_index.shape)
        #     # print("TopK: classification_idx   |", classification_idx, classification_idx.shape)

        #     # Select z based on ce loss
        #     cls_min_ce_index = min_ce_index.expand(-1, -1, attention_mask.shape[-1])
        #     attention_mask = torch.gather(attention_mask, dim=1, index=cls_min_ce_index).squeeze(1)
        #     encoder_hidden_states = torch.gather(encoder_hidden_states, dim=1,
        #                                          index=cls_min_ce_index.unsqueeze(-1).expand(-1, -1, -1,
        #                                                                                      hdim)).squeeze(1)
        #     pseudo_labels = torch.gather(classification_idx, dim=1, index=min_ce_index).squeeze(1)
        #     z = torch.gather(z, dim=1, index=min_ce_index.unsqueeze(-1).expand(-1, -1, -1, hdim)).squeeze(1)

        #     # print("TopK: z                     |", z.shape)
        #     # print("TopK: attention_mask        |", attention_mask.shape)
        #     # print("TopK: encoder_hidden_states |", encoder_hidden_states.shape)
        #     # print("TopK: pseudo_labels         |", pseudo_labels, pseudo_labels.shape)
        #     # print("-"*80)
        #     # input()

        #     encoder_outputs.z                 = z
        #     encoder_outputs.attention_mask    = attention_mask
        #     encoder_outputs.last_hidden_state = encoder_hidden_states
        #     return encoder_outputs, pseudo_labels


        # g_l_steps: total steps of langevin sampling   (self.g_l_steps)
        # g_l_step_size: step size in langevin dynamics (self.g_l_step_size)
        # z^{i+1} = z^{i} + s^2/2 * grad{log p(z) + log p(x|z)} + s * N(0,1)
        def sample_langevin_post_z(z):
            z = z.clone().detach()
            z_0 = z.clone().detach()
            z.requires_grad = True
            z_grad_g_grad_norm = 0
            for i in range(self.g_l_steps):
                encoder_outputs.__setattr__('z', z)
                # print('before', decoder_inputs_embeds)
                self.model.eval()
                outputs = self.model.decoder_forward(
                    input_ids,
                    decoder_input_ids=decoder_input_ids,
                    encoder_outputs=encoder_outputs,  # put all the encoder outputs to the decoder
                    decoder_attention_mask=decoder_attention_mask,
                    decoder_head_mask=decoder_head_mask,
                    cross_attn_head_mask=cross_attn_head_mask,
                    past_key_values=past_key_values,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                self.model.train()
                lm_logits = self.lm_head(outputs[2]) + self.final_logits_bias
                # print(outputs[2])
                # print(outputs[1])
                # print(outputs[0])
                loss_fct = CrossEntropyLoss(reduction='sum')
                masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
                z_grad_g = torch.autograd.grad(masked_lm_loss, z)[0]
                # print('after', z)
                # input()
                # print('loss', masked_lm_loss)
                # print('z_grad_shape', z_grad_g.shape)

                '''
                if self.sample_latent:
                    # prior from N(0, 1)
                    z_grad_e = z.data * 2 * vae_kl_weight
                else:
                    # prior initialzed from bart encoder
                    z_grad_e = (z.data - z_0) * 2 * vae_kl_weight
                '''
                
                z_grad_e = (z.data - z_0) * 2 * vae_kl_weight

                z.data = z.data - 0.5 * self.g_l_step_size * self.g_l_step_size * (z_grad_g + z_grad_e)
                #z.data = z.data
                # TODO not adding noise here
                if not self.remove_noise:
                    z.data += self.g_l_step_size * torch.randn_like(z).data

                z_grad_g_grad_norm = z_grad_g.view(bsz, -1).norm(dim=1).mean()
                #z_grad_g_grad_norm = 0 
                if self.verbose:
                    print('Langevin posterior {:3d}/{:3d}: CE={:8.3f} Norm={:8.3f}'.format(i + 1, self.g_l_steps,
                                                                                           masked_lm_loss.item(),
                                                                                           z_grad_g_grad_norm))
                    if i == self.g_l_steps - 1: print("--------------------\n")
            return z.detach(), z_grad_g_grad_norm


        # Ours: second step knowledge selection
        # Get top-k candidates and select k^* which has the lowest CE loss with response
        if labels is not None and self.training and self.model.top_k_kn > 1:
            encoder_outputs, pseudo_labels = top_k_selection(encoder_outputs, labels, random=self.random_choice)
        else:
            pseudo_labels = None

        # Our: Add langevin posterior inference here with prior model as N(0,1)
        # Suppose dim(z) = [bs, 1, 1, 768]
        # z_g_0 as initialization from encoder or N(0, 1)
        if self.sample_latent:
            z_g_0 = torch.randn_like(encoder_outputs.z).data
        else:
            z_g_0 = encoder_outputs.z

        if labels is not None and self.training:
            z_g_k, _ = sample_langevin_post_z(z=z_g_0)
            encoder_outputs.z = z_g_k

        if not self.training:
            z_g_k = z_g_0
            if self.gen_with_noise:
                z_g_k = encoder_outputs.z
                z_g_k += torch.randn_like(z_g_k).data / 10  # TODO what should be here?
                encoder_outputs.z = z_g_k

        outputs = self.model.decoder_forward(
            input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,  # put all the encoder outputs to the decoder
            decoder_attention_mask=decoder_attention_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[2]) + self.final_logits_bias
        classification_logits = outputs[1]

        total_loss = None
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            total_loss = masked_lm_loss
            masked_lm_loss = masked_lm_loss.item()

        cls_loss = None
        if classification_labels is not None:
            cls_loss_fct = CrossEntropyLoss()
            if pseudo_labels is not None and self.pseudo_confidence > 0:
                # print("classification_logits", classification_logits, classification_logits.shape)
                # print("classification_labels", classification_labels, classification_labels.shape)
                cls_logits = classification_logits.view(bsz, -1)
                cls_labels = torch.zeros_like(cls_logits)
                classification_labels = classification_labels.unsqueeze(-1)
                # classification_labels = torch.cat((classification_labels, pseudo_labels), dim=1)
                cls_labels = cls_labels.scatter_(1, pseudo_labels, self.pseudo_confidence)
                if not self.pseudo_label_only:
                    cls_labels = cls_labels.scatter_(1, classification_labels, 1)
                # print("classification_labels", classification_labels, classification_labels.shape)
                # print("cls_logits", cls_logits, cls_logits.shape)
                # print("cls_labels", cls_labels, cls_labels.shape)
                # input()
                
                cls_loss = cls_loss_fct(cls_logits, cls_labels)
            else:
                cls_loss = cls_loss_fct(classification_logits.view(bsz, -1), classification_labels.view(-1))
            total_loss = total_loss + self.cls_ratio * cls_loss if total_loss is not None else self.cls_ratio * cls_loss
            cls_loss = cls_loss.item()

        kl_loss = None
        if labels is not None and self.add_z_mse:
            kl_loss_fct = MSELoss()
            kl_loss = kl_loss_fct(z_g_0.view(-1), z_g_k.view(-1))
            total_loss = total_loss + vae_kl_weight * kl_loss if total_loss is not None else vae_kl_weight * kl_loss
            kl_loss = kl_loss.item()

        if not return_dict:
            output = (lm_logits, classification_logits,) + outputs[3:]
            return ((total_loss, masked_lm_loss, cls_loss, kl_loss) + output) if total_loss is not None else output

        # print("total loss", total_loss.item(), "lm loss", masked_lm_loss.item(), "cls loss", cls_loss.item(), "kl loss", kl_loss.item())
        # input()
        return Seq2SeqLMOutputWithLatent(
            loss=total_loss,
            lm_loss=masked_lm_loss,
            cls_loss=cls_loss,
            kl_loss=kl_loss,
            logits=lm_logits,
            classification_logits=classification_logits,
            pseudo_labels=pseudo_labels,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
    
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        knowledge_mask = kwargs["decoder_knowledge_mask"]
        ctx_input_ids = kwargs["decoder_ctx_input_ids"] if "decoder_ctx_input_ids" in kwargs else None
        ctx_attention_mask = kwargs["decoder_ctx_attention_mask"] if "decoder_ctx_attention_mask" in kwargs else None
        expand_size = attention_mask.shape[0] // knowledge_mask.shape[0]
        if expand_size > 1:
            # beam search: need to expand the knowledge mask
            expanded_return_idx = (
                torch.arange(knowledge_mask.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(
                    attention_mask.device)
            )
            knowledge_mask = knowledge_mask.index_select(0, expanded_return_idx)
        
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "decoder_shapes": kwargs["decoder_shapes"],
            "knowledge_mask": knowledge_mask,
            "ctx_input_ids": ctx_input_ids,
            "ctx_attention_mask": ctx_attention_mask,
        }


class FiDBartModel(BartModel):

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the BART forward method uses the input tensors to infer
    # dimensions used in the decoder.
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        decoder_encoder_attention_mask: Optional[torch.FloatTensor] = None,
        decoder_shapes: Optional[tuple] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:

        bsz, n_passages, passage_length = decoder_shapes
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            if return_dict:
                encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.view(bsz,
                                                                                           n_passages * passage_length,
                                                                                           -1)
            else:
                encoder_outputs[0] = encoder_outputs[0].view(bsz, n_passages * passage_length, -1)

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=decoder_encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class FiDBartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = FiDBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        decoder_encoder_attention_mask: Optional[torch.FloatTensor] = None,
        decoder_shapes: Optional[tuple] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            decoder_encoder_attention_mask=decoder_encoder_attention_mask,
            decoder_shapes=decoder_shapes,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        decoder_encoder_attention_mask = kwargs["decoder_encoder_attention_mask"]
        expand_size = encoder_outputs[0].shape[0] // decoder_encoder_attention_mask.shape[0]
        if expand_size > 1:
            # beam search: need to expand the knowledge mask
            expanded_return_idx = (
                torch.arange(decoder_encoder_attention_mask.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(
                    encoder_outputs[0].device)
            )
            decoder_encoder_attention_mask = decoder_encoder_attention_mask.index_select(0, expanded_return_idx)

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "decoder_shapes": kwargs["decoder_shapes"],
            "decoder_encoder_attention_mask": decoder_encoder_attention_mask,
        }

    def get_encoder_outputs(
        self, inputs_tensor: torch.Tensor, model_kwargs
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. get encoder outputs
        encoder_kwargs["return_dict"] = True
        encoder_kwargs["input_ids"] = inputs_tensor
        encoder_outputs = encoder(**encoder_kwargs)

        shapes = encoder_outputs.last_hidden_state.shape
        encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.view(1, shapes[0] * shapes[1], -1)

        return encoder_outputs
