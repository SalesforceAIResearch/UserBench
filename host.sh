export CUDA_VISIBLE_DEVICES=0,1,2,3

# Example:
vllm serve Qwen/Qwen3-8B \
   --max-model-len 32768 \
   --gpu-memory-utilization 0.9 \
   --tensor-parallel-size 4 \
   --enable-auto-tool-choice \
   --tool-call-parser hermes \
   --chat-template tool_template/hermes.jinja \
   --port 8000

