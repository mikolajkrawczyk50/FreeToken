# Running FreeToken with Ornith-1.5-35B & Pi Coding Agent

This guide documents how to configure, run, and optimize **FreeToken** serving the **Ornith-1.5-35B-Q4_K_M** model on an **AMD Radeon RX 6800** (16 GB VRAM, RDNA2 / `gfx1030`) with a **64,000-token context window**, **4,096-token chunked prefill**, and integration with **Pi Coding Agent** (`pi`).

---

## 1. System Requirements & Architecture

- **Model**: `vcruz305/Ornith-1.5-35B-A3B-GGUF` (Qwen3.5-MoE architecture)
  - 40 decoder layers (hybrid GDN linear attention + full attention every 4 layers)
  - 256 routed experts + 1 shared expert (top-8 routing)
  - Mixed quantization: `ffn_gate_exps` is `IQ3_S` / `Q4_K`, `ffn_down_exps` is `Q4_K` (layers 0–4) and `IQ3_S` / `Q4_K` (layers 5–39)
- **Target Hardware**:
  - GPU: AMD Radeon RX 6800 (16 GB VRAM, RDNA2 / `gfx1030`)
  - Host RAM: 64 GB+ recommended (holds offloaded expert banks in pinned host memory)
- **Software Stack**:
  - Linux ROCm (ROCm 6.x / PyTorch HIP)
  - FreeToken with Triton attention & GDN kernels
  - Client: [Pi Coding Agent](https://github.com/mariozechner/pi) (or any OpenAI-compatible client)

---

## 2. Environment Variables

On RDNA2 GPUs (`gfx1030`), hardware lacks native bfloat16 dot instructions (`v_dot2_f32_bf16`). The following environment variables must be exported before launching:

```bash
export PYTHONUNBUFFERED=1
export PYTORCH_ROCM_ARCH=gfx1030
export FREETOKEN_ROCM_ARCH=gfx1030
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

---

## 3. Checkpoint Formats: FTW vs GGUF

FreeToken supports serving directly from the raw GGUF file or from FreeToken's optimized **FTW (FreeToken Weight)** format.

### Recommended: FTW Format (Fast Path)
FTW repacks model weights and MoE expert banks into contiguous, aligned shards for parallel chunked `O_DIRECT` streaming:
- **Dense weight load**: ~4 seconds
- **Expert bank load**: ~30 seconds (saturates host NVMe read speeds up to 1+ GB/s)
- **Zero parsing overhead**: Tensor shapes and quant types are pre-indexed in `freetoken_weight.json`.

#### Converting GGUF to FTW
If you need to generate or regenerate the FTW checkpoint:
```bash
python3 -m freetoken.cli convert \
  --model-path /home/user/Ornith-1.5-35B-Q4_K_M.gguf \
  --output-dir /home/user/Ornith-1.5-35B-FTW \
  --dtype float16 \
  --moe-backend offload
```
Output directory structure:
```
/home/user/Ornith-1.5-35B-FTW/
├── freetoken-00000.ftw       # Sharded weights & expert banks
├── freetoken-00001.ftw
├── freetoken-00002.ftw
├── freetoken_weight.json     # Metadata, tensor offsets, and quant types
└── source_metadata.gguf      # GGUF header and dense layer metadata
```

---

## 4. Launching the FreeToken Server

### Production Command (FTW Fast Path)

```bash
export PYTHONUNBUFFERED=1
export PYTORCH_ROCM_ARCH=gfx1030
export FREETOKEN_ROCM_ARCH=gfx1030
export HSA_OVERRIDE_GFX_VERSION=10.3.0

python3 -m freetoken.cli serve \
  --model-path /home/user/Ornith-1.5-35B-FTW \
  --served-model-name Ornith-1.5-35B-Q4_K_M.gguf \
  --dtype float16 \
  --moe-backend offload \
  --attention-backend triton \
  --cuda-graph-max-bs 0 \
  --memory-ratio 0.88 \
  --max-prefill-length 4096 \
  --kv-reserve-tokens 64000 \
  --enable-special-token-ckpt \
  --tool-call-parser qwen3_coder \
  --port 8000
```

### Fallback: Direct GGUF Launch
You can also point `--model-path` directly to the `.gguf` file:
```bash
python3 -m freetoken.cli serve \
  --model-path /home/user/Ornith-1.5-35B-Q4_K_M.gguf \
  --served-model-name Ornith-1.5-35B-Q4_K_M.gguf \
  --dtype float16 \
  --moe-backend offload \
  --attention-backend triton \
  --cuda-graph-max-bs 0 \
  --memory-ratio 0.88 \
  --max-prefill-length 4096 \
  --kv-reserve-tokens 64000 \
  --enable-special-token-ckpt \
  --tool-call-parser qwen3_coder \
  --port 8000
```

### Argument Details

| Argument | Value | Purpose |
|---|---|---|
| `--model-path` | `/home/user/Ornith-1.5-35B-FTW` | Checkpoint location (FTW directory or `.gguf` file). |
| `--served-model-name` | `Ornith-1.5-35B-Q4_K_M.gguf` | The model name reported via `/v1/models` matching agent configuration. |
| `--dtype` | `float16` | Required on RDNA2 (`gfx1030`) to avoid missing hardware BF16 dot product instructions. |
| `--moe-backend` | `offload` | Stores 256 MoE experts in pinned host RAM and manages an active GPU cache. |
| `--attention-backend` | `triton` | Enables Triton kernels for full attention and hybrid GDN recurrent state. |
| `--cuda-graph-max-bs` | `0` | Disables CUDA graphs (required when using dynamic MoE offloading). |
| `--memory-ratio` | `0.88` | Uses 88% of VRAM (~14 GiB) for weights, KV cache, and MoE cache, leaving ~1.9 GiB free headroom. |
| `--max-prefill-length` | `4096` | Chunked prefill chunk size (4096 tokens/chunk delivers ~410–450 tokens/sec throughput). |
| `--kv-reserve-tokens` | `64000` | Reserves KV cache space for 64,000 tokens (~1.22 GiB) before filling remaining VRAM with MoE cache. |
| `--enable-special-token-ckpt` | *flag* | Semantic anchor checkpoints for prefix caching across agent turns and tool calls. |
| `--tool-call-parser` | `qwen3_coder` | Native Qwen 3.5 Coder tool-call parser for automated tool invocation. |
| `--port` | `8000` | Port for the OpenAI-compatible HTTP server. |

---

## 5. Configuring and Running Pi Coding Agent

### Pi Configuration (`~/.pi/agent/models.json`)

Configure FreeToken as a provider in `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "freetoken": {
      "name": "FreeToken",
      "baseUrl": "http://127.0.0.1:8000/v1",
      "apiKey": "freetoken",
      "api": "openai-completions",
      "models": [
        {
          "id": "Ornith-1.5-35B-Q4_K_M.gguf",
          "name": "Ornith 1.5 35B (FreeToken)",
          "reasoning": true,
          "contextWindow": 64000,
          "maxTokens": 8192,
          "compat": {
            "supportsReasoningEffort": false
          },
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

### Verifying Pi Setup

List available models to verify the provider:
```bash
pi --list-models freetoken
```
Expected output:
```
provider   model                       context  max-out  thinking  images
freetoken  Ornith-1.5-35B-Q4_K_M.gguf  64K      8.2K     yes       no
```

### Running Pi

- **Interactive Agent Session**:
  ```bash
  pi --provider freetoken --model Ornith-1.5-35B-Q4_K_M.gguf
  ```

- **Single Instruction (Print Mode)**:
  ```bash
  pi --provider freetoken --model Ornith-1.5-35B-Q4_K_M.gguf -p "Run 'uname -r' and summarize."
  ```

---

## 6. Runtime Performance & Prefix Caching

On the RX 6800 (16 GB VRAM) serving Ornith-1.5-35B:

1. **MoE Cache**: Auto-allocates **4,830 expert slots** in VRAM alongside the **64,055-token KV cache**.
2. **Chunked Prefill (4096 tokens)**:
   - Chunk 1 (cold start): ~115 tokens/sec
   - Subsequent chunks: **410–454 tokens/sec**
3. **Radix Prefix Cache**:
   - Full agent system prompts and history (~19,456 tokens) are cached across turns.
   - Subsequent requests achieve **19,456 tokens cache hit**, prefilling only delta tokens in <1 second (effective throughput **~4,900 tokens/sec**).
4. **Generation / Decode Throughput**: **~25–26 tokens/sec**.

---

## 7. Troubleshooting

- **Checking Server Readiness**:
  ```bash
  curl -s http://127.0.0.1:8000/v1/models | jq
  ```
- **Port Conflict**:
  If port 8000 is occupied, find and kill the lingering process:
  ```bash
  lsof -i :8000
  kill -9 <PID>
  ```
- **Monitoring GPU VRAM / Utilization**:
  ```bash
  rocm-smi
  ```
- **Triton / JIT Kernels**:
  On first startup, FreeToken hipifies and compiles HIP kernels (`kernel/csrc/gguf/`). These are cached in `~/.cache/freetoken/` or local `csrc/` directories so subsequent boots are near-instantaneous.
