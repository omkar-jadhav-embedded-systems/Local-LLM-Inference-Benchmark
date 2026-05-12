import ollama
import time
import pandas as pd
import psutil
import threading
import subprocess

# --- Global variables for the hardware monitoring thread ---
is_generating = False
cpu_readings = []
ram_readings_gb = []


def monitor_hardware():
    """
    This function runs in a background thread, recording CPU and RAM usage.
    """
    global is_generating, cpu_readings, ram_readings_gb
    while is_generating:
        cpu_readings.append(psutil.cpu_percent(interval=None))
        ram_readings_gb.append(psutil.virtual_memory().used / (1024 ** 3))
        time.sleep(0.2)


# --- Static Metadata & Prompts ---
MODEL_METADATA = {
    'qwen2.5-coder:1.5b': {'Size': '1.0 GB', 'Arch': 'Dense'},
    'qwen2.5-coder:7b': {'Size': '4.7 GB', 'Arch': 'Dense'},
    'qwen2.5-coder:14b': {'Size': '9.0 GB', 'Arch': 'Dense'},
    'mistral:7b-instruct': {'Size': '4.4 GB', 'Arch': 'Dense'},
    'codellama:7b': {'Size': '3.8 GB', 'Arch': 'Dense'},
    'deepseek-coder-v2:16b': {'Size': '8.9 GB', 'Arch': 'MoE'}
}

LONG_PROMPT = """
A transformer model is a neural network that learns context and thus meaning by tracking relationships in sequential data, like the words in this sentence.
Transformer models apply an evolving set of mathematical techniques, called attention or self-attention, to detect subtle ways even distant data elements in a series influence and depend on each other.
Invented and introduced by Google in 2017, transformers are one of the most important, recent advances in artificial intelligence (AI), and are the basis for most of the large language models (LLMs) in the news today, such as ChatGPT, Gemini and others.
A key characteristic of transformer models is their parallel processing capability. Unlike recurrent neural networks (RNNs) that process data sequentially, transformers can process all input tokens simultaneously. This parallelization allows for significantly faster training on large datasets and enables the models to learn from vast amounts of text.
The attention mechanism is the core innovation of the transformer. It allows the model to weigh the importance of different words in the input sequence when producing the output. For each output element, the model has access to the entire input sequence and can selectively focus on the most relevant parts. This ability to capture long-range dependencies is a major advantage over previous architectures.
The model architecture consists of an encoder and a decoder. The encoder processes the input sequence and creates a contextual representation, while the decoder generates the output sequence one token at a time, using the encoder's representation and the previously generated tokens. This setup is particularly effective for tasks like machine translation, text summarization, and question answering, where understanding the full context of the input is crucial for generating a coherent and accurate output.
Now, based on the text above, explain what the 'attention mechanism' is in one sentence.
"""

SHORT_PROMPT = "Explain what the 'attention mechanism' is in one sentence."


def run_full_benchmark(model_name, prompt, prompt_type, temperature, baseline_ram):
    """
    Runs a benchmark capturing performance, hardware, and temperature metrics.
    """
    global is_generating, cpu_readings, ram_readings_gb
    print(f"Testing {model_name} | {prompt_type} | Temp={temperature:.1f}...", end=" ", flush=True)

    cpu_readings, ram_readings_gb = [], []

    try:
        is_generating = True
        monitor_thread = threading.Thread(target=monitor_hardware)
        monitor_thread.start()

        start_time = time.perf_counter()
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            stream=False,
            options={"temperature": temperature}
        )
        end_time = time.perf_counter()

        is_generating = False
        monitor_thread.join()

        # --- CALCULATIONS ---
        total_latency = end_time - start_time
        eval_count = response.get('eval_count', 0)
        eval_duration_ns = response.get('eval_duration', 1)
        tps = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0
        avg_cpu = sum(cpu_readings) / len(cpu_readings) if cpu_readings else 0
        peak_ram = max(ram_readings_gb) if ram_readings_gb else baseline_ram
        model_ram_cost = peak_ram - baseline_ram

        print("Done.")

        meta = MODEL_METADATA.get(model_name, {'Size': 'N/A', 'Arch': 'N/A'})
        return {
            "Model": model_name,
            "Arch": meta['Arch'],
            "Size": meta['Size'],
            "Prompt Type": prompt_type,
            "Temperature": temperature,
            "TPS (Speed)": round(tps, 2),
            "Total Latency (s)": round(total_latency, 2),
            "Avg CPU %": round(avg_cpu, 1),
            "Peak RAM (GB)": round(peak_ram, 1),
            "Model RAM Cost (GB)": round(model_ram_cost, 1),
            "Tokens Generated": eval_count
        }
    except Exception as e:
        is_generating = False
        print(f"Failed: {e}")
        return None


# --- SCRIPT EXECUTION ---
models_to_test = [
    'qwen2.5-coder:1.5b', 'qwen2.5-coder:7b', 'qwen2.5-coder:14b',
    'mistral:7b-instruct', 'codellama:7b', 'deepseek-coder-v2:16b'
]
temperatures_to_test = [i / 10.0 for i in range(11)]
prompts_to_test = {"Short": SHORT_PROMPT, "Long": LONG_PROMPT}

benchmark_results = []

print("Unloading all models for a clean baseline memory reading...")
for model in models_to_test:
    subprocess.run(["ollama", "stop", model], capture_output=True, text=True)
time.sleep(5)
baseline_ram_gb = psutil.virtual_memory().used / (1024 ** 3)
print(f"System Baseline RAM: {baseline_ram_gb:.1f} GB")

for model in models_to_test:
    print(f"\n{'=' * 60}\n--- BENCHMARKING MODEL: {model} ---\n{'=' * 60}")

    # Load the model once before iterating through tests
    print(f"Loading {model} into memory...", end=" ", flush=True)
    try:
        ollama.generate(model=model, prompt="Hello", stream=False)
        print("Loaded.")
    except Exception as e:
        print(f"Load failed for {model}: {e}. Skipping.")
        continue

    for prompt_type, prompt in prompts_to_test.items():
        for temp in temperatures_to_test:
            result = run_full_benchmark(model, prompt, prompt_type, temp, baseline_ram_gb)
            if result:
                benchmark_results.append(result)

    # Unload the model to free up memory for the next one
    subprocess.run(["ollama", "stop", model], capture_output=True, text=True)
    print(f"--- Unloaded {model} ---")
    time.sleep(2)

# --- FINAL REPORT GENERATION ---
df = pd.DataFrame(benchmark_results)

column_order = [
    "Model", "Arch", "Size", "Prompt Type", "Temperature", "TPS (Speed)",
    "Total Latency (s)", "Avg CPU %", "Peak RAM (GB)", "Model RAM Cost (GB)", "Tokens Generated"
]
df = df[column_order]

print("\n" + "=" * 160)
print("                                                         ULTIMATE LOCAL LLM PERFORMANCE & HARDWARE REPORT")
print("=" * 160)
print(df.sort_values(by=["Model", "Prompt Type", "Temperature"]).to_string(index=False))
print("=" * 160)
