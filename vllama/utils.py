import json
import os
from typing import Union, Dict, Any
from transformers import AutoConfig, PretrainedConfig

# Constants for VRAM calculation
BYTES_TO_GB = 1024**3
OVERHEAD_FACTOR = 0.20  # Heuristic 20% overhead factor for activations/system

def _get_bytes_per_parameter(dtype: str) -> float:
    """Returns the number of bytes per parameter (Cb) based on the data type."""
    dtype = dtype.lower()
    if dtype in ['float32', 'fp32']:
        return 4.0
    elif dtype in ['bfloat16', 'bf16', 'float16', 'fp16']:
        return 2.0  # Standard half-precision 
    elif dtype in ['int8']:
        return 1.0
    elif dtype in ['int4']:
        return 0.5
    else:
        # Support common string inputs like 'torch.bfloat16'
        if '16' in dtype:
             return 2.0
        elif '8' in dtype:
             return 1.0
        elif '4' in dtype:
             return 0.5
        raise ValueError(f"Unsupported or unknown precision type: {dtype}. Use fp32, bf16, int8, or int4.")

def _calculate_llm_parameters(config: PretrainedConfig) -> int:
    """
    Calculates the total number of static parameters (P) using first principles 
    based on a Decoder-Only (e.g., Llama/Mistral-style) architecture.
    """
    try:
        D_model = getattr(config, 'hidden_size')
        N_layers = getattr(config, 'num_hidden_layers')
        V = getattr(config, 'vocab_size')
        N_attn_heads = getattr(config, 'num_attention_heads')
        D_ff = getattr(config, 'intermediate_size')
        D_head = getattr(config, 'head_dim', D_model // N_attn_heads)
    except AttributeError as e:
        raise AttributeError(f"Config file is missing a critical parameter: {e}. Cannot perform accurate parameter count.")

    P_total = 0

    # 1. Embedding Layer Parameters (P_Embed) 
    P_Embed = V * D_model
    P_total += P_Embed

    # 2. Output Projection Layer Parameters (P_Output)
    # Check for weight tying (common in Llama/Mistral)
    if not getattr(config, 'tie_word_embeddings', True):
        P_Output = V * D_model
        P_total += P_Output

    # 3. Parameters per single Decoder Block (P_Block)
    
    # 3.1. Attention Layer Parameters (P_Attn)
    # Includes W_Q, W_K, W_V, W_O projection matrices 
    
    # Determine N_kv_heads for GQA/MQA parameter size reduction
    N_kv_heads = getattr(config, 'num_key_value_heads', N_attn_heads)
    
    
    # Q Projection: D_model * D_model
    P_Q = D_model * N_attn_heads * D_head
    
    # K and V Projections (adjusted for GQA/MQA)
    # Dimensions: D_model x (N_kv_heads * D_head)
    
    P_K = D_model * N_kv_heads * D_head
    P_V = D_model * N_kv_heads * D_head
    
    # O Projection: D_model * D_model
    P_O = N_attn_heads * D_head * D_model
    
    P_Attn = P_Q + P_K + P_V + P_O
    
    # 3.2. MLP Layer Parameters (P_MLP) - Gated MLP (up_proj, gate_proj, down_proj) 
    # P_MLP = (D_model * D_ff) * 2 + (D_ff * D_model)
    P_MLP = D_model * D_ff * 2 + D_ff * D_model
    
    # 3.3. Normalization Layers (P_Norm) - Usually 2 per block (e.g., pre-attention and post-attention)
    P_Norm = 2 * D_model
    
    P_Block = P_Attn + P_MLP + P_Norm
    
    # 4. Accumulate all blocks
    P_total += N_layers * P_Block
    
    # 5. Final Normalization Layer (e.g., Llama's model.norm)
    P_total += D_model
    
    return P_total


def calculate_llm_inference_vram(
    model_name_or_path: str,
    context_length: int,
    batch_size: int,
    dtype: str = 'bf16'
) -> Dict[str, Union[float, str]]:
    """
    Calculates the required VRAM for Transformer model inference given context length and batch size.

    :param model_name_or_path: Hugging Face model name or local path (to config.json).
    :param context_length: Maximum context length (L) assumed for KV Cache.
    :param batch_size: Inference batch size (B).
    :param dtype: Model loading precision (e.g., 'fp16', 'bf16', 'int8').
    :return: Dictionary containing the VRAM breakdown in GB.
    """
    
    # 1. Initialization and Configuration Loading
    try:
        config = AutoConfig.from_pretrained(model_name_or_path)
    except Exception as e:
        # Fallback for local config.json loading
        config_path = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(config_path):
             with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
                config = PretrainedConfig.from_dict(config_dict)
        else:
            raise FileNotFoundError(f"Could not load model configuration from {model_name_or_path}. Error: {e}")

    # 1.1 Get bytes per parameter and key dimensions
    C_b = _get_bytes_per_parameter(dtype)
    P_total = _calculate_llm_parameters(config)
    
    D_model = getattr(config, 'hidden_size')
    D_ff = getattr(config, 'intermediate_size')
    N_layers = getattr(config, 'num_hidden_layers')
    N_attn_heads = getattr(config, 'num_attention_heads')
    
    # 2. Static VRAM Calculation (VRAM_Weights)
    # This includes all static parameters: Embeddings, Attention, MLP, and Norms.
    VRAM_Weights_Bytes = P_total * C_b
    
    # 3. Dynamic VRAM Calculation (VRAM_KV) - GQA-aware
    
    # 3.1 Determine N_kv_heads (Key/Value Heads)
    # Prioritize num_key_value_heads for MQA/GQA detection 
    N_kv_heads = getattr(config, 'num_key_value_heads', N_attn_heads)
    
    # Ensure N_kv_heads does not exceed N_attn_heads (failsafe)
    if N_kv_heads > N_attn_heads:
        N_kv_heads = N_attn_heads 
    
    # 3.2 Determine Head Dimension (D_head)
    D_head = D_model // N_attn_heads
    if getattr(config, "head_dim"):
        D_head = getattr(config, "head_dim")
    
    # 3.3 Apply Generalized KV Cache Formula [3, 6]
    # VRAM_KV = 2 * L * B * N_layers * N_kv_heads * D_head * C_b
    VRAM_KV_Bytes = (
        2 *  # K and V matrices
        context_length *  # L (Context/Sequence Length)
        batch_size *  # B (Batch Size)
        N_layers *
        N_kv_heads *  # Key/Value heads (optimized for GQA/MQA)
        D_head *
        C_b  # Bytes per element
    )
    print(f"context_length: {context_length}, batch_size: {batch_size}, N_layers: {N_layers}, N_kv_heads: {N_kv_heads}, D_head: {D_head}, C_b: {C_b}")
    
    # 4. System Overhead (VRAM_Overhead)
    # Using 20% heuristic based on static weights 
    VRAM_Overhead_Bytes = VRAM_Weights_Bytes * OVERHEAD_FACTOR
    VRAM_Peak_Activation_Bytes = C_b * D_model * context_length * batch_size + C_b * D_ff * context_length * batch_size
    
    # 5. Aggregation
    VRAM_Total_Bytes = VRAM_Weights_Bytes + VRAM_KV_Bytes + VRAM_Overhead_Bytes + VRAM_Peak_Activation_Bytes
    
    return {
        "model_parameters_count": P_total,
        "bytes_per_parameter": C_b,
        "head_dimension_D_head": D_head,
        "kv_heads_N_kv": N_kv_heads,
        "static_vram_weights_GB": VRAM_Weights_Bytes / BYTES_TO_GB,
        "dynamic_vram_kv_cache_GB": VRAM_KV_Bytes / BYTES_TO_GB,
        "peak_activation_vram_GB": VRAM_Peak_Activation_Bytes / BYTES_TO_GB,
        "estimated_overhead_GB": VRAM_Overhead_Bytes / BYTES_TO_GB,
        "total_vram_excluding_kv_cache_GB": (VRAM_Total_Bytes - VRAM_KV_Bytes) / BYTES_TO_GB,
        "total_vram_required_GB": VRAM_Total_Bytes / BYTES_TO_GB
    }

# print(calculate_llm_inference_vram(
#     model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
#     context_length=32768,
#     batch_size=1,
#     dtype="bf16"
# ))
