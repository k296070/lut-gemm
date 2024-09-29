CUDA_VISIBLE_DEVICES=1 python ./examples/quant_model_rtn.py \
    --model_name_or_path ../model/Meta-Llama-3-8B \
    --qbits 3 \
    --group_size 128