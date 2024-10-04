Phi-3-medium-4k-instruct

MODEL=Meta-Llama-3-8B-Instruct

python ./examples/quant_model_rtn.py \
    --model_name_or_path ../models/$MODEL\
    --output_name ./outputs/$MODEL-w4-g128-lutgemm.pt\
    --load_quant ../llm-awq/quant_cache/$MODEL-w4-g128-awq-lutgemm-v2.pt\
    --qbits 4 \
    --group_size 128


    