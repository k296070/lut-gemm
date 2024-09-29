CUDA_VISIBLE_DEVICES=1 python ./examples/quant_model_rtn.py \
    --model_name_or_path ../llm-awq/quant_cache/Meta-Llama-3-8B-w4-g128-awq-v2.pt \
    --qbits 4 \
    --group_size 128