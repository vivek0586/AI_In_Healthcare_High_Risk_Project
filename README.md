# Health insurance claims and prior authorization on Parity 1.0

This study uses the cases and reference answers released by Pendral Inc. in Parity suite version 1.0, with the model registry frozen on September 2, 2026. The study changes the response contracts, uses a subset, uses one local model and one run per case, and reports its own field-level measures. See the [task catalog](https://paritybench.ai/tasks) and [benchmark license](https://paritybench.ai/licence).

The payment comparison covers 12 single-claim in-network benefit cases and two document-precedence cases. The missing-documentation check uses 12 prior authorization cases spanning lumbar MRI, bariatric surgery, and continuous glucose monitoring. Case IDs are fixed in the code. No cases or gold answers were created. The program reads the policy, scenario, and published answer from the original item page. Only the policy and scenario enter model prompts. The distributed archive links to source pages and includes source hashes, numerical and categorical scoring references, and this study's model responses. It does not distribute the task text or the publisher's model transcripts.

The extra rule-mode labels are this study's annotations, with one reading per case. In BEN-009 the mode records the underlying benefit before the out-of-pocket cap zeroes the charge. BEN-022 is classified as a copay based on the publisher's labeled payment breakdown, since the policy extract does not explicitly name nurse practitioner visits. Its nominal copay is only an arithmetic check input because the allowed amount is below it. ADJ-017's rate does not determine the answer because the entire allowed amount falls within the remaining deductible. ADJ-018 uses the mandate stated by the benchmark as a fact of the case.

## Reproduce

Use Python 3.9 or later to score the bundled results. Run from the repository directory. The scorer downloads the published reference answers on its first run. Open `Parity_Results.ipynb` to view the executed analysis without a download.

```powershell
python analyze.py
```

To collect new predictions, use the public [llama.cpp b12242 release](https://github.com/ggml-org/llama.cpp/releases/tag/b12242) and the [official Qwen2.5-0.5B-Instruct GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF), file qwen2.5-0.5b-instruct-q4_k_m.gguf. Expected GGUF SHA-256 is 74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db. Start the server from a directory containing the executable and model. The original environment used a Windows CPU with four model threads.

```powershell
./llama-server.exe -m ./qwen2.5-0.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8312 --ctx-size 4096 --threads 4 --parallel 1 --no-webui
```

In another terminal, run the study with a fresh output path. It fetches the original case pages into a local cache. The calculator control reads that same cache, so it can run offline once the pages have been fetched.

```powershell
python study.py --output rerun
python calculator_check.py
```

The supplied analysis reads the archived `results` directory. To analyze a new run, keep the original archive and run the scorer from a separate copy where `results` contains the new predictions and references. Recomputed results can differ with backend version, hardware, or source page changes. Each run stores original source links, SHA-256 hashes of input text and page bytes, saved model outputs, and per-case scores.
