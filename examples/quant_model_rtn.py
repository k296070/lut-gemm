# LUT-GEMM
# Copyright (c) 2024-present NAVER Cloud Corp. All rights reserved.
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

import torch
import transformers
import argparse

from transformers import (

    AutoModelForCausalLM,
)
from accelerate import (

    load_checkpoint_in_model,
)
from rtn_parameter import RTNParameter
from bcq_parameter import BCQParameter

import torch
import torch.nn as nn
from tqdm import tqdm
import gc
from .qmodule import ScaledActivation
from ..utils.module import set_op_by_name

from transformers.models.bloom.modeling_bloom import BloomBlock

EMBEDDING_KEYWORDS = ["embed"]
LM_HEAD_KEYWORDS = ["lm_head", "embed_out", "output"]


def scale_activations(module):
    param = next(module.parameters())
    dtype = param.dtype
    device = param.device
    if isinstance(module, BloomBlock):
        if isinstance(module.mlp.gelu_impl, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.gelu_impl, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.gelu_impl", act)
    elif "mptblock" in str(module.__class__.__name__).lower():
        if isinstance(module.ffn.act, ScaledActivation):
            return
        c = module.ffn.up_proj.out_features
        act = ScaledActivation(
            module.ffn.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "ffn.act", act)
    elif "falcon" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "bigcode" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.c_proj.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "neox" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)


# core quantization method (simulated quantization)
def pseudo_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w


@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit,
    q_config,
):
    from .pre_quant import get_blocks, get_named_linears

    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            m.cuda()
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, **q_config
            )
            m.cpu()


@torch.no_grad()
def real_quantize_model_weight(model, w_bit, group_size, init_only=False):
    from .qmodule import WQLinear
    from .pre_quant import get_blocks, get_named_linears

    layers = get_blocks(model)
    for i in tqdm(
        range(len(layers)),
        desc="real weight quantization..." + ("(init only)" if init_only else ""),
    ):
        layer = layers[i]
        named_linears = get_named_linears(layer)
        scale_activations(layer)

        for name, module in named_linears.items():
            if init_only:
                q_linear = WQLinear.from_linear(
                    module, w_bit, group_size, True
                )
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)

    torch.cuda.empty_cache()
    

layers = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='facebook/opt-125m',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--load_quant",
        type=str,
        default='facebook/opt-125m',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--qbits",
        type=int,
        default=4,
        help="Quantization Bits.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="quantization grouping size for weights",
    )
    args = parser.parse_args()

    return args

#def quant_model(model, module_to_not_convert:str = "lm_head"):
def quant_model(model, args):
    from .qmodule import WQLinear
    from .pre_quant import get_blocks, get_named_linears
    layers = get_blocks(model)
    for i in tqdm(
        range(len(layers))
    ):
        layer = layers[i]
        named_linears = get_named_linears(layer)
        scale_activations(layer)
        for name, module in named_linears.items():
            
            print(i, name, module," in")
            #original_weight = module.weight.clone().detach()
            # INT4 Quantization -> RTN
            #print(original_weight.dtype)
            #w_rtn = RTNParameter(original_weight)
            #scale, zero, w_quant, w_quant_shape = w_rtn.compress(
            #   in_ch_wise=False, qbits=args.qbits, group_size=args.group_size,
            #    perchannel=True, sym=False)
            #print("w_quant ",w_quant.shape)
            #print(w_quant.dtype)
            #print("scale" ,scale.shape)
            #print(scale.dtype)
            #print("zero ",zero.shape)
            #print(zero.dtype)
            # Convert INT4 -> BCQ4
            #alpha, binary, binary_shape, offset = w_rtn.convert_bcq_format(
            #    scale, zero, w_quant, qbits=args.qbits,
            #    do_packing=False, in_ch_wise=False)

            #print("Parameter size before packing")
            #print("  alpha.size()  =", alpha.size())
            #print("  binary.size() =", binary.size())
            #print("="*30)

            # Packing BCQ4 -> Packed Weight (uint8)
            #alpha, binary, binary_shape, offset = w_rtn.convert_bcq_format(
            #    scale, zero, w_quant, qbits=args.qbits,
            #    do_packing=True, in_ch_wise=False)

            #print("Parameter size after packing")
            #print("  alpha.size()  =", alpha.size())
            #print("  binary.size() =", binary.size())
            print("="*30)

    return model

def main():
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        )
    
    real_quantize_model_weight(
        model, w_bit=args.w_bit,group_size=args.group_size , init_only=True
    )    

    load_checkpoint_in_model(
        model,
        checkpoint=args.load_quant,
        device_map="auto",
        offload_state_dict=True,
    )

    model = quant_model(model, args)

if __name__ == "__main__":
    main()
