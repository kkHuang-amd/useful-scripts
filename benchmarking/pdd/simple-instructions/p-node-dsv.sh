
#--disaggregation-ib-device `rdma dev show | awk -F': ' '{print $2}' | awk '{print $1}' | paste -sd ','` \
#--disaggregation-ib-device "ionic_0,ionic_1,ionic_2,ionic_3" \
#--model-path /dockerx/data/models/DeepSeek-R1-0528-MXFP4/ \
python -m sglang.launch_server \
  --model-path /dockerx/data/models/gpt-oss-120b/ \
  --disaggregation-mode prefill \
  --host 10.235.58.248 --port 8888 \
  --tp-size 8 \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device "ionic_0,ionic_1,ionic_2,ionic_3" \
  --disable-radix-cache \
  --trust-remote-code
