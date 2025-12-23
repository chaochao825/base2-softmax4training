"""BitNet-style quantization for LLaMA models with configurable Softmax."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, LlamaDecoderLayer

from src.quant.bitlinear import BitLinear
from src.ops.base2_softmax import Base2Softmax


class BitNetLlamaAttention(nn.Module):
    """Modified LLaMA attention with BitLinear and configurable softmax."""
    
    def __init__(self, config: LlamaConfig, softmax_type: str = "standard", temperature: float = 1.0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # Use BitLinear for projections
        self.q_proj = BitLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = BitLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = BitLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = BitLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Configurable softmax
        if softmax_type == "base2":
            self.softmax = Base2Softmax(temperature=temperature)
        else:
            self.softmax = lambda x: torch.nn.functional.softmax(x, dim=-1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Repeat k/v heads if necessary
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Apply configurable softmax
        attn_weights = self.softmax(attn_weights)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value


class BitNetLlamaMLP(nn.Module):
    """LLaMA MLP with BitLinear layers."""
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = BitLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = BitLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = BitLinear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def convert_llama_to_bitnet(
    model: LlamaForCausalLM,
    softmax_type: str = "standard",
    temperature: float = 1.0,
    replace_embeddings: bool = False
) -> LlamaForCausalLM:
    """
    Convert a LLaMA model to use BitNet quantization.
    
    Args:
        model: Pre-trained LLaMA model
        softmax_type: 'standard' or 'base2'
        temperature: Temperature for softmax
        replace_embeddings: Whether to replace embeddings (usually keep as FP for stability)
    
    Returns:
        Modified model with BitLinear layers
    """
    for name, module in model.named_modules():
        if isinstance(module, LlamaDecoderLayer):
            # Replace attention
            module.self_attn = BitNetLlamaAttention(
                model.config,
                softmax_type=softmax_type,
                temperature=temperature
            )
            # Replace MLP
            module.mlp = BitNetLlamaMLP(model.config)
    
    return model


