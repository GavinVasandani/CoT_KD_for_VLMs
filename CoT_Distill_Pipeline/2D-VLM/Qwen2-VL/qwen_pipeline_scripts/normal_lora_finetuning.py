import os
import gc
import json
import torch
from PIL import Image
import argparse
import pandas as pd
import numpy as np
import transformers

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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

# Define a simple Dataset to load training examples from your JSON file.
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, images_dir, processor, prompt_template, df, max_length=512):
        with open(data_path, "r") as f:
            raw_data = json.load(f)
        self.images_dir = images_dir
        self.processor = processor
        self.prompt_template = prompt_template
        self.samples = []
        self.df = df
        self.max_length = max_length
        for scene_id, changes_list in raw_data.items():
            for change in changes_list:
                context_change = change["context_change"]
                for qa in change["questions_answers"]:
                    question = qa["question"]
                    answer = qa["answer"]
                    self.samples.append({
                        "scene_id": scene_id,
                        "context_change": context_change,
                        "question": question,
                        "answer": answer
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        scene_id = sample["scene_id"]
        image_path = os.path.join(self.images_dir, f"{scene_id}.png")
        # Open and convert image to RGB
        image = Image.open(image_path).convert("RGB")

        scene_orientation = extract_non_nan_values(self.df[self.df['scene_id'] == scene_id])
        scene_orientation = " ".join(
            f"The {item} was located at the {direction.lower()} of the scene."
            for scene_id, directions in scene_orientation.items()
            for direction, item in directions.items()
        )
        prompt = self.prompt_template.format(scene_orientation, sample["context_change"], sample["question"])
    
        # Messages combines the text prompt (composed of scene orientation, context change and question) with path to image
        # Gives multimodal input to tokenize and feed into model
        # So messages is dict of dicts so we pass a dict content which has dict with key type and val images and for the image key it has the actual image content/path to img        
        # Same for the prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # return {"messages": messages}
        return messages

def compute_metrics(eval_preds):
    
    def flatten(x):
        # If x is a list with one element that is itself a list then return that sublist.
        if isinstance(x, list) and len(x) == 1 and isinstance(x[0], list):
            print("STATUS: Found nested list")
            return x[0]
        return x

    print("-" * 80)
    print("STATUS: Started compute metrics after eval on eval_dataset")
    
    preds, labels = eval_preds

    # Check if preds or labels are torch.Tensor or np.ndarray and whether they contain raw logits.
    # For raw logits, we expect three dimensions: (batch_size, sequence_length, vocab_size).
    if isinstance(preds, (torch.Tensor, np.ndarray)):
        if preds.ndim == 3:
            print("STATUS: preds likely contains raw logits, applying argmax to convert to token IDs.")
            if isinstance(preds, torch.Tensor):
                preds = torch.argmax(preds, dim=-1)
            else:  # np.ndarray case
                preds = np.argmax(preds, axis=-1)
        else:
            print("STATUS: preds likely contains token IDs.")
    else:
        print("STATUS: preds is neither torch.Tensor nor np.ndarray; assuming token IDs.")

    # For labels, this is less common but checking in case they're raw logits.
    if isinstance(labels, (torch.Tensor, np.ndarray)):
        if labels.ndim == 3:
            print("STATUS: labels likely contains raw logits, applying argmax to convert to token IDs.")
            if isinstance(labels, torch.Tensor):
                labels = torch.argmax(labels, dim=-1)
            else:
                labels = np.argmax(labels, axis=-1)
        else:
            print("STATUS: labels likely contains token IDs.")
    else:
        print("STATUS: labels is neither torch.Tensor nor np.ndarray; assuming token IDs.")

    # Convert predictions to list-of-lists
    print("STATUS: Convert preds to list of lists")
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().tolist()
    elif isinstance(preds, np.ndarray):
        preds = preds.tolist()

    # Flatten preds to prevent nested list cases.
    print("STATUS: Flattening preds")
    preds = [flatten(p) for p in preds]

    # Replace masked labels (-100) with the pad token id and convert to list-of-lists if needed.
    print("STATUS: Replace masked tokens with token ids")
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().tolist()
    elif isinstance(labels, np.ndarray):
        labels = labels.tolist()

    print("STATUS: Flattening labels")
    labels = [flatten(l) for l in labels] 

    # Inspecting input shape and type of elements in preds
    print("STATUS: Inspecting input shape and type of element in preds")
    for idx, p in enumerate(preds):
        print(f"preds[{idx}] type: {type(p)}; length: {len(p)}")

    # Decode preds and labels in chunks to avoid processing huge batches at once.
    print("STATUS: Decode preds and labels in chunks of 50")
    chunk_size = 50
    decoded_preds = []
    for i in range(0, len(preds), chunk_size):
        print(f"STATUS: Processing chunk {i // chunk_size} of preds")
        decoded_preds.extend(processor.batch_decode(preds[i:i + chunk_size], skip_special_tokens=True))
    decoded_labels = []
    for i in range(0, len(labels), chunk_size):
        print(f"STATUS: Processing chunk {i // chunk_size} of labels")
        decoded_labels.extend(processor.batch_decode(labels[i:i + chunk_size], skip_special_tokens=True))

    # Helper function to ensure each element is a string.
    def safe_str(x):
        if not isinstance(x, str):
            try:
                x = str(x)
            except Exception:
                x = ""
        return x

    # Compute exact match score after decoding.
    print("STATUS: Compute EM after decoding preds, labels tokens to strings")
    exact_matches = [1 if safe_str(p).strip() == safe_str(l).strip() else 0 
                     for p, l in zip(decoded_preds, decoded_labels)]
    print("STATUS: Printing decoded_preds, decoded_labels")
    print(f"The decoded pred: {p}, the decoded label: {l}" for p, l in zip(decoded_preds, decoded_labels))
    exact_match_score = sum(exact_matches) / len(exact_matches)
    return {"exact_match": exact_match_score}

import time

def clear_memory():
    # Delete variables if they exist in the current global scope
    if "inputs" in globals():
        del globals()["inputs"]
    if "model" in globals():
        del globals()["model"]
    if "processor" in globals():
        del globals()["processor"]
    if "trainer" in globals():
        del globals()["trainer"]
    if "peft_model" in globals():
        del globals()["peft_model"]
    if "bnb_config" in globals():
        del globals()["bnb_config"]
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def main():
    # Clear GPU memory at start
    clear_memory()
    torch.cuda.empty_cache()
    gc.collect()

    print(f"Transformers version: {transformers.__version__}")
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--train_data", type=str, default="../../hypo_dataset/train_contextvqa.json", help="Path to the training JSON file")
    parser.add_argument("-m", "--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Teacher model ID to finetune with LoRA")
    parser.add_argument("-i", "--images_dir", type=str, default="/gpfs/home/ym621/gavin/Hypo3D/hypo_dataset/dataset/top_view_with_label_rotated", help="Directory with scene images")
    parser.add_argument("-e", "--eval_data", type=str, default="../../hypo_dataset/val_contextvqa.json", help="Path to validation JSON file")
    parser.add_argument("-p", "--pipeline_run_dir", type=str, required=True, help="Directory to save generated LoRA adapter to.")
    args = parser.parse_args()

    prompt_template = (
        "Given a top-view of a 3D scene, mentally rotate the image to align with the specified orientation.\n\n"
        "Scene Orientation: {}\n\n"
        "Now, given a context change, imagine how the scene would look after the change has been applied. Then, answer a question based on the changed scene.\n\n"
        "Context Change: {}\n"
        "Question: {}\n\n"
        "The answer should be a single word or short phrase.\n\n"
        "The answer is:"
    )

    global processor
    processor = AutoProcessor.from_pretrained(args.model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto", # need to set to cuda?
        # quantization_config=bnb_config
    )

    # model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False

    print("-" * 80)
    print("INFO: Printing signature of model foward func")

    import inspect
    print(inspect.signature(model.forward))
    print(model.forward.__doc__)
    print("-" * 80)

    # Output named modules of Qwen2-VL - need to use this to decide
    print("Model modules:")
    for name, module in model.named_modules():
        print(name)

    num_visual_layers = 32
    num_lang_layers = 28

    print(f"The number of vision model layers is: {num_visual_layers}") 
    print(f"The number of language model layers is: {num_lang_layers}")
    
    vision_modules = []
    for i in range(num_visual_layers):
        vision_modules.append(f"visual.blocks.{i}.attn.qkv")

    language_modules = []
    for i in range(num_lang_layers):
        for proj in ["k_proj", "v_proj", "q_proj"]:
            language_modules.append(f"model.model.layers.{i}.self_attn.{proj}")

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

    # model.gradient_checkpointing_enable()
    # model.enable_input_require_grads()

    model.print_trainable_parameters()

    df = pd.read_excel("/gpfs/home/ym621/gavin/Hypo3D/hypo_dataset/axis_def_hypo.xlsx", 
                         sheet_name='Sheet1', engine='openpyxl')
    train_dataset = Dataset(args.train_data, args.images_dir, processor, prompt_template, df)
    eval_dataset = Dataset(args.eval_data, args.images_dir, processor, prompt_template, df)

    training_args = TrainingArguments(
        output_dir="lora_finetuned_qwen",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=2, # reduces from 4 to 2 - to reduce memory usage
        # gradient_checkpointing=True,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=100,
        eval_strategy="no",
        # predict_with_generate=True,
        bf16=True,
        gradient_accumulation_steps=4,
        report_to="wandb",
        run_name="NormalLoraFinetuningQWEN",
        label_names=["labels"],
        # optim="adamw_torch",
        optim="adamw_bnb_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    # trainer.train()
    # model_name = args.model_id.replace("/", "_")
    # model.save_pretrained(f"lora_finetuned_qwen_{model_name}")

    trainer.train()
    model_name = args.model_id.replace("/", "_")
    full_path = os.path.join(args.pipeline_run_dir, f"lora_finetuned_{model_name}")
    os.makedirs(full_path, exist_ok=True)
    model.save_pretrained(full_path)
    print(f"Standard Qwen LoRA adapter saved to: {full_path}")

if __name__ == "__main__":
    main()
