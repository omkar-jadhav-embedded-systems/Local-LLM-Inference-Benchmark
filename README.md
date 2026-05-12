# 🚀 Local-LLM-Inference-Benchmark

A comprehensive, automated Python benchmarking suite designed to evaluate the real-world inference performance of local Large Language Models (LLMs) running via [Ollama](https://ollama.com/). 

This project goes beyond simple "Tokens Per Second" (TPS). Utilizing multi-threading and the `psutil` library, it measures the holistic impact of models on consumer-grade hardware, including RAM bottlenecks, CPU utilization, context processing latency (Time to First Token), and the behavioral effects of temperature.

## 💻 Hardware Environment
*   **Processor:** Intel Core Ultra 7 165H
*   **Graphics:** Intel Arc Graphics (Integrated)
*   **RAM:** 32 GB
*   **OS:** Windows

## 🤖 Models Tested
| Model | Size | Architecture |
| :--- | :--- | :--- |
| `qwen2.5-coder:1.5b` | 1.0 GB | Dense |
| `codellama:7b` | 3.8 GB | Dense |
| `mistral:7b-instruct` | 4.4 GB | Dense |
| `qwen2.5-coder:7b` | 4.7 GB | Dense |
| `deepseek-coder-v2:16b` | 8.9 GB | **MoE (Mixture of Experts)** |
| `qwen2.5-coder:14b` | 9.0 GB | Dense |

## 📊 Key Insights & Discoveries

### 1. The MoE Advantage (Mixture of Experts)
Counter-intuitively, the **16B DeepSeek model vastly outperformed the 7B dense models**. Despite being an 8.9 GB file, its MoE architecture only activates a fraction of its parameters during generation. It consistently delivered **~13-14 TPS** on long contexts, compared to ~6-7 TPS for the 7B dense models, proving MoE is the superior architecture for hardware-constrained systems.

### 2. Hitting the "VRAM Wall"
The `qwen2.5-coder:14b` model (9.0 GB) perfectly demonstrated the catastrophic failure of exceeding available high-speed memory. When processing a long prompt, system RAM usage spiked to **29.3 GB**, pushing the system into heavy swap usage. Latency exploded to **129.5 seconds** (over 2 minutes) and generation collapsed to **1.94 TPS**. 

### 3. The "Cost of Context"
Time to First Token (TTFT) was tracked across Short (1 sentence) and Long (~550 tokens) prompts. For all models, initial latency scaled dramatically with prompt size. For example, `mistral:7b-instruct` jumped from **1.3s** (Short) to **27.8s** (Long), proving that prompt processing (prefill), not text generation, is the primary latency bottleneck for document-analysis tasks.

### 4. Instruction Adherence vs. Temperature
By iterating through temperatures `0.0` to `1.0`, the data captured the exact point of "hallucination". At high temperatures (0.8+), output token counts fluctuated wildly (e.g., jumping from 34 to 72 tokens for the exact same prompt), demonstrating the loss of strict instruction adherence in favor of creative generation.

## 🛠️ How to Run the Benchmark

1. Ensure Ollama is installed and your desired models are pulled.
2. Install the required Python dependencies:
   ```bash
   pip install pandas psutil
