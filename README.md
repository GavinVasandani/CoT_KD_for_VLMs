# CoT\_KD\_for\_VLMs: Chain-of-Thought Knowledge Distillation Pipelines

This repository provides pipeline scripts to automate the end-to-end process of Chain-of-Thought (CoT) knowledge distillation for Vision-Language Models (VLMs) in 3D Visual Question Answering (VQA) tasks. The pipelines support multiple benchmarks (e.g., Hypo3D and ScanQA), model architectures (e.g., LLaVA-OV and Qwen2-VL), and student-teacher training paradigms (e.g., LoRA fine-tuning, KL distillation).

---

## 📃 Pipeline Scripts Overview

### 1. `blind_cot_inference_pipeline.sh`

Runs the full pipeline on the **Hypo3D** benchmark and evaluates both **blind** and **normal** performance for zero-shot, standard LoRA, and CoT-distilled models.

#### Main stages:

* Baseline zero-shot evaluation
* Standard LoRA adapter fine-tuning (teacher)
* CoT + raw logits generation
* Student LoRA finetuning with CoT + logits
* Blind and normal evaluation of multiple variants

#### Example:

```bash
bash blind_cot_inference_pipeline.sh
```

---

### 2. `cot_inference_pipeline_hypo3d.sh`

Generic and modular pipeline for **Hypo3D**, parameterized by teacher/student model and dataset. Use this to run CoT distillation on a teacher-student VLM pair that will be fine-tuned for Hypo3D task.

#### Usage:

```bash
bash cot_inference_pipeline_hypo3d.sh <teacher_model_id> <student_model_id> <benchmark_json>
```

#### Example:

```bash
bash cot_inference_pipeline_hypo3d.sh \
  llava-hf/llava-onevision-qwen2-7b-ov-hf \
  llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
  hypo_dataset/contextvqa_full.json
```

---

### 3. `cot_inference_pipeline_scanqa.sh`

Same structure as the Hypo3D pipeline but tailored for **ScanQA** data and image directories. Use this to run CoT distillation on a teacher-student VLM pair that will be fine-tuned for ScanQA task.

#### Example:

```bash
bash cot_inference_pipeline_scanqa.sh \
  llava-hf/llava-onevision-qwen2-7b-ov-hf \
  llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
  scanqa_dataset/contextual_scanqa_full.json
```

---

### 4. `scanqa_modular_cot_inference_pipeline.sh`

Modular ScanQA pipeline with dynamic model type detection (LLaVA-OV or Qwen2-VL). Prepares data splits, generates CoT/logits, fine-tunes student, and evaluates performance.

#### Example:

```bash
bash scanqa_modular_cot_inference_pipeline.sh \
  llava-hf/llava-onevision-qwen2-7b-ov-hf \
  llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
  scanqa_dataset/contextual_scanqa_full.json
```

---

### 5. `randomize_test_cot_inference_pipeline.sh`

Lightweight script for testing finetuning using CoT and logits with minimal setup. Uses environment variables for dataset paths.

#### Usage:

```bash
export VAL_DATA_PATH=hypo_dataset/cot_contextvqa_split/val_contextvqa.json
export TRAIN_DATA_PATH=hypo_dataset/cot_contextvqa_split/train_contextvqa.json
export TEST_DATA_PATH=hypo_dataset/cot_contextvqa_split/test_contextvqa.json
bash randomize_test_cot_inference_pipeline.sh
```

---

## 📁 Directory and Script Assumptions

Each script expects the following directory layout:

```
2D-VLM/
├── llava-ov/
│   ├── evaluate.py
│   ├── cot_gen.py
│   ├── cot_lora_finetuning.py
│   └── model_with_lora_evaluate.py
hypo_dataset/
├── dataset_split.py
└── contextvqa_full.json
```

* If using ScanQA, update `images_dir` paths to point to ScanNet-compatible top-view images.
* Adapter weights, intermediate CoT generations, and logits are saved inside a unique `pipeline_run_*` directory per run.

---

## 👁️ Evaluation Coverage

All pipelines perform the following evaluations:

* **Zero-shot student** (no adapter)
* **Standard LoRA-finetuned student** (CE loss only)
* **CoT-distilled student** (KL loss + CoT supervision)
* **Blind vs Normal** (for vision dependency analysis)

---

## 🔧 Example Environment Setup

```bash
# Create symbolic links for dataset folders
ln -s /path/to/Hypo3D hypo_dataset
ln -s /path/to/ScanQA scanqa_dataset
```

You may want to adjust script paths for your local cluster/HPC environment.

---

## 📃 Citation & Contact

This work was developed as part of a research project on Chain-of-Thought distillation for VLMs.

Feel free to open issues or reach out for clarifications or extensions!
