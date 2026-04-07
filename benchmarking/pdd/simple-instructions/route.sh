#python -m sglang_router.launch_router \
#  --pd-disaggregation --mini-lb \
#  --prefill 10.235.58.248:8000 \
#  --decode 10.235.58.247:8000 \
#  --host 0.0.0.0 --port 70000

python -m sglang_router.launch_router \
  --pd-disaggregation \
  --mini-lb \
  --port 8000 \
  --prefill http://10.235.58.248:8888 \
  --decode http://10.235.58.247:8888
