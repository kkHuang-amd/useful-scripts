MODEL=/dockerx/data/deepseek-ai/DeepSeek-V4-Pro/

ATOM_DISABLE_MMAP=true ATOM_MOE_GU_ITLV=1 AITER_BF16_FP8_MOE_BOUND=0 python3 -m atom.entrypoints.openai_server --model ${MODEL} --server-port 8000 -tp 8 --kv_cache_dtype fp8 --trust-remote-code --enable-dp-attention
