CUDA_VISIBLE_DEVICES=1 python ./examples/quant_model_rtn.py \
    --model_name_or_path ../model/Meta-Llama-3-8B \
    --load_quant ../llm-awq/quant_cache/Meta-Llama-3-8B-w3-g128-awq-lutgemm-v2.pt\
    --qbits 3 \
    --group_size 128