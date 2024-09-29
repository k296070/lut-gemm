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
from tqdm import tqdm
import torch
import transformers
import argparse
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    GPTQConfig,
    LlamaTokenizer

)
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)
from rtn_parameter import RTNParameter
from bcq_parameter import BCQParameter
from qmodule import WQLinear

def set_op_by_name(layer, name, new_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

def get_blocks(model):
    if model.__class__.__name__ == "LlamaForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        # layers = [model.model.layers, model.model.vision_tower.vision_tower.vision_model.encoder.layers]
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    else:
        raise NotImplementedError(type(model))
    return layers
def real_quantize_model_weight(model, w_bit, init_only=False):

    layers = get_blocks(model)
    for i in tqdm(
        range(len(layers)),
        desc="real weight quantization..." + ("(init only)" if init_only else ""),
    ):
        layer = layers[i]
        named_linears = get_named_linears(layer)

        for name, module in named_linears.items():
            if init_only:
                q_linear = WQLinear.from_linear(
                    module, w_bit, 128, True
                )
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)

layers = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, WQLinear)}

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
    layers = model.model.layers
    print(layers)
    for i in tqdm(
        range(len(layers)),
    ):
        layer = layers[i]
        named_linears = get_named_linears(layer)
    
    
        for name, module in named_linears.items():

            print(i, name,module)
            #original_weight = module.weight.clone().detach()
            # INT4 Quantization -> RTN
            #w_rtn = RTNParameter()
            #scale, zero, w_quant, w_quant_shape = w_rtn.compress(
            #    in_ch_wise=False, qbits=args.qbits, group_size=args.group_size,
            #    perchannel=True, sym=False)
            #scale = f["model.layers.30.self_attn.v_proj.scales"]
            #zero = f["model.layers.30.self_attn.v_proj.scaled_zeros"]
            #w_quant =  f["model.layers.30.self_attn.v_proj.qweight"]

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
            #print("="*30)

            #my_state_dict = {
            #    'model.layers.30.self_attn.v_proj.alpha': alpha, 
            #    'model.layers.30.self_attn.v_proj.binary': binary, 
            #    'model.layers.30.self_attn.v_proj.zero': zero
            #}

            #torch.save(my_state_dict,"layer30_v_proj_weight_packed.pt")


def main():
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        )
    
    real_quantize_model_weight(
        model, w_bit=args.qbits, init_only=True
    )
    max_memory = [v.split(":") for v in ([])]
    max_memory = {(int(k) if k.isdigit() else k): v for k, v in max_memory}
    kwargs = {"max_memory": max_memory} if len(max_memory) else {}
    device_map = infer_auto_device_map(
        model,
        no_split_module_classes=[
            "OPTDecoderLayer",
            "LlamaDecoderLayer",
            "BloomBlock",
            "MPTBlock",
            "DecoderLayer",
        ],
        **kwargs,
    )
    load_checkpoint_in_model(
        model,
        checkpoint=args.load_quant,
        device_map=device_map,
        offload_state_dict=True,
    )

    model = quant_model(model, args)


if __name__ == "__main__":
    main()
