# Program that outputs LoRA adapter trained on --train_data (arg 0), for model --model_id (arg 1),
# training data uses images from --images_dir (arg 2)
# Also does loss function that only considers model prediction and reference answer.
# So cross-entropy loss func. 
# Same as lora_finetune.py but collate_fn is different.

import os
import gc
import json
import torch
from PIL import Image
import argparse
import pandas as pd
import numpy as np
import transformers

from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# Function to extract all non-NaN values for each scene_id and return as a list
def extract_non_nan_values(df):
    extracted_values = {}
    for index, row in df.iterrows():
        scene_id = row['scene_id']
        if pd.notna(scene_id):
            scene_values = {}
            for direction in ['Front', 'Back', 'Left', 'Right']:
                value = row[direction]
                if pd.notna(value):
                    scene_values[direction] = value
            if scene_values:
                extracted_values[scene_id] = scene_values
    return extracted_values

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, images_dir, processor, prompt_template, max_length=512):
        with open(data_path, "r") as f:
            raw_data = json.load(f)
        self.images_dir = images_dir
        self.processor = processor
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.samples = []

        for scene_id, qa_list in raw_data.items():
            for qa in qa_list:
                self.samples.append({
                    "scene_id": scene_id,
                    "question": qa["question"],
                    "answer": qa["answers"][0]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        scene_id = sample["scene_id"]
        image_path = os.path.join(self.images_dir, f"{scene_id}.png")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = image.resize((384, 384))

        # Only use the question in the prompt
        prompt = self.prompt_template.format(sample["question"])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        full_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        # Tokenize as prompt-answer pair
        tokenized = self.processor.tokenizer(
            full_prompt,
            sample["answer"],
            return_tensors="pt",
            padding="max_length",
            truncation="only_second",
            max_length=self.max_length
        )

        for k, v in tokenized.items():
            tokenized[k] = v.squeeze(0)

        prompt_ids = self.processor.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )["input_ids"].squeeze(0)
        prompt_len = prompt_ids.size(0)

        labels = tokenized["input_ids"].clone()
        labels[:prompt_len] = -100
        tokenized["labels"] = labels

        return tokenized

def collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        sequences = [item[key] for item in batch]
        pad_val = -100 if key == "labels" else 0
        collated[key] = pad_sequence(sequences, batch_first=True, padding_value=pad_val)
    return collated

def compute_metrics(eval_preds):
    def flatten(x):
        if isinstance(x, list) and len(x) == 1 and isinstance(x[0], list):
            return x[0]
        return x

    preds, labels = eval_preds

    # Handle raw logits if present
    if isinstance(preds, (torch.Tensor, np.ndarray)) and preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)
    if isinstance(labels, (torch.Tensor, np.ndarray)) and labels.ndim == 3:
        labels = np.argmax(labels, axis=-1)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().tolist()
    elif isinstance(preds, np.ndarray):
        preds = preds.tolist()

    preds = [flatten(p) for p in preds]
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().tolist()
    elif isinstance(labels, np.ndarray):
        labels = labels.tolist()

    labels = [flatten(l) for l in labels]

    # Decode tokens
    decoded_preds = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Normalize text for comparison
    def normalize(text):
        return text.strip().lower()

    def partial_match(predicted, reference):
        pred_tokens = predicted.split()
        ref_tokens = reference.split()
        common = set(pred_tokens).intersection(set(ref_tokens))
        return len(common) / len(ref_tokens) if ref_tokens else 0

    partial_match_scores = [
        partial_match(normalize(p), normalize(l)) for p, l in zip(decoded_preds, decoded_labels)
    ]
    partial_match_score = sum(partial_match_scores) / len(partial_match_scores)

    # Log partial match as eval accuracy
    return {"eval_accuracy": partial_match_score}

import time

def main():
    # Clear GPU memory at start
    torch.cuda.empty_cache()
    gc.collect()

    print(f"Transformers version: {transformers.__version__}")

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--train_data", type=str, default="../../hypo_dataset/train_contextvqa.json", help="Path to the training JSON file")
    parser.add_argument("-m", "--model_id", type=str, default="llava-hf/llava-onevision-qwen2-7b-ov-hf", help="Teacher model ID to finetune with LoRA")
    parser.add_argument("-i", "--images_dir", type=str, default="/gpfs/home/ym621/gavin/Hypo3D/hypo_dataset/dataset/top_view_with_label_rotated", help="Directory with scene images")
    parser.add_argument("-e", "--eval_data", type=str, default="../../hypo_dataset/val_contextvqa.json", help="Path to validation JSON file")
    parser.add_argument("-p", "--pipeline_run_dir", type=str, required=True, help="Directory to save finetuned model and outputs")
    args = parser.parse_args()

    prompt_template = (
        "Given a top-view of a 3D scene, answer the question:\n"
        "Question: {}\n\n"
        "The answer should be a single word or short phrase.\n\n"
        "The answer is:"
    )

    global processor
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model_id, 
        use_flash_attention_2=False,
        device_map="auto"
    )

    model.config.use_cache = False

    print("-" * 80)
    print("INFO: Printing signature of model foward func")

    import inspect
    print(inspect.signature(model.forward))
    print(model.forward.__doc__)
    print("-" * 80)

    for name, module in model.named_modules():
        print(name)
        
    num_vision_encoder_layers = 26
    vision_modules = []
    for i in range(num_vision_encoder_layers):
        for proj in ["k_proj", "v_proj", "q_proj"]:
            vision_modules.append(f"vision_tower.vision_model.encoder.layers.{i}.self_attn.{proj}")

    num_language_model_layers = 24
    language_modules = []
    for i in range(num_language_model_layers):
        for proj in ["k_proj", "v_proj", "q_proj"]:
            language_modules.append(f"language_model.model.layers.{i}.self_attn.{proj}")

    target_modules = language_modules + vision_modules

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    df = pd.read_excel("/gpfs/home/ym621/gavin/Hypo3D/hypo_dataset/axis_def_hypo.xlsx", 
                         sheet_name='Sheet1', engine='openpyxl')
    train_dataset = Dataset(args.train_data, args.images_dir, processor, prompt_template, df)
    eval_dataset = Dataset(args.eval_data, args.images_dir, processor, prompt_template, df)

    training_args = TrainingArguments(
        output_dir="lora_finetuned_llava",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=50,
        fp16=True,
        gradient_accumulation_steps=4,
        report_to="wandb",
        run_name="NormalLoraFinetuningLLAVA_OV",
        label_names=["labels"],
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model_name = args.model_id.replace("/", "_")
    full_path = os.path.join(args.pipeline_run_dir, f"lora_finetuned_{model_name}")
    os.makedirs(full_path, exist_ok=True)
    model.save_pretrained(full_path)
    print(f"Standard LLaVA-OV LoRA adapter saved to: {full_path}")

if __name__ == "__main__":
    main()
