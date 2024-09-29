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

from rtn_parameter import RTNParameter
from bcq_parameter import BCQParameter


layers = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--model_name_or_path",
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
def quant_model(model, args,f):
    #layers = model.model.layers

    #for i in tqdm(
    #    range(len(layers)),
    #):
    #    layer = layers[i]
    #    named_linears = get_named_linears(layer)
    #
    #
    #    for name, module in named_linears.items():

    #        print(i, name,module)
            #original_weight = module.weight.clone().detach()
            # INT4 Quantization -> RTN
            w_rtn = RTNParameter()
            #scale, zero, w_quant, w_quant_shape = w_rtn.compress(
            #    in_ch_wise=False, qbits=args.qbits, group_size=args.group_size,
            #    perchannel=True, sym=False)
            scale = f["model.layers.30.self_attn.v_proj.scales"]
            zero = f["model.layers.30.self_attn.v_proj.scaled_zeros"]
            w_quant =  f["model.layers.30.self_attn.v_proj.qweight"]

            # Convert INT4 -> BCQ4
            alpha, binary, binary_shape, offset = w_rtn.convert_bcq_format(
                scale, zero, w_quant, qbits=args.qbits,
                do_packing=False, in_ch_wise=False)

            print("Parameter size before packing")
            print("  alpha.size()  =", alpha.size())
            print("  binary.size() =", binary.size())
            print("="*30)

            # Packing BCQ4 -> Packed Weight (uint8)
            alpha, binary, binary_shape, offset = w_rtn.convert_bcq_format(
                scale, zero, w_quant, qbits=args.qbits,
                do_packing=True, in_ch_wise=False)

            print("Parameter size after packing")
            print("  alpha.size()  =", alpha.size())
            print("  binary.size() =", binary.size())
            print("="*30)

            my_state_dict = {
                'model.layers.30.self_attn.v_proj.alpha': alpha, 
                'model.layers.30.self_attn.v_proj.binary': binary, 
                'model.layers.30.self_attn.v_proj.zero': zero
            }
            torch.save(my_state_dict,"layer30_v_proj_weight_packed.pt")


def main():
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        )
    
    f= torch.load("../llm-awq/quant_cache/Meta-Llama-3-8B-w4-g128-awq-lutgemm-v2.pt")
    model = quant_model(model, args,f)

    torch.save({
    'alpha': model.alpha,
    'binary': model.binary,
    'w_quant': model.w_quant
    }, 'LLama3-8B-lut-gemm.pt')

if __name__ == "__main__":
    main()
