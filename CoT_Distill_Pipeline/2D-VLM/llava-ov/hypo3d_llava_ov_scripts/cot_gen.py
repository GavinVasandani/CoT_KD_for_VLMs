# Script to generate chain of thought and model outputs via inference.
# Script also writes chain of thought and outputs to json file and saves raw_logits.
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import torch
from PIL import Image
import json
import os
import re
from word2number import w2n
import argparse
import pandas as pd
from tqdm import tqdm
from bert_score import score as bert_score

def save_json(data, filename, pipeline_run_dir):
    os.makedirs(pipeline_run_dir, exist_ok=True)
    full_path = os.path.join(pipeline_run_dir, filename)
    with open(full_path, "w") as file:
        json.dump(data, file, indent=4)
    print(f"Saved enriched train data with CoT to {full_path}")

def read_csv(path):
    with open(path, "r") as file:
        data = file.readlines()
    return data

# Function to extract all non-NaN values for each scene_id and return as a list
def extract_non_nan_values(df):
    extracted_values = {}

    # Iterate over each row and extract the values
    for index, row in df.iterrows():
        scene_id = row['scene_id']
        if pd.notna(scene_id):
            # Create a dictionary to hold the non-NaN values for the current scene_id
            scene_values = {}

            # Extract non-NaN values for 'Front', 'Back', 'Left', 'Right'
            for direction in ['Front', 'Back', 'Left', 'Right']:
                value = row[direction]
                if pd.notna(value):
                    scene_values[direction] = value

            # Add the non-NaN values for the current scene_id to the overall dictionary
            if scene_values:
                extracted_values[scene_id] = scene_values

    return extracted_values

# Load the Excel file
columns_to_load = ["scene_id"]

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--data_path", type=str, default="reason3d/recategorize_test_data.json", help="Path to the JSON file containing the test data")
parser.add_argument("-m", "--teacher_model_id", type=str, default="llava-hf/llava-onevision-qwen2-7b-ov-hf", help="Model ID of the teacher model to be used.")
parser.add_argument("-a", "--adapter_dir", type=str, default="lora_finetuned_llava_llava-hf_llava-onevision-qwen2-7b-ov-hf", help="Path to your LoRA adapter directory")
parser.add_argument("-s", "--student_model_id", type=str, default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", help="Model ID of the student model to be used.")
parser.add_argument("-p", "--pipeline_run_dir", type=str, required=True, help="Directory to save finetuned model and outputs")
args = parser.parse_args()

data = json.load(open(args.data_path))
# data = {k: data[k] for k in list(data.keys())[300:400]}

# Use teacher model's tokenizer and hence vocab:
# processor = AutoProcessor.from_pretrained(args.teacher_model_id, use_fast=True) # SPEEDUP CHANGE: added use_fast=True 

# Use student model's tokenizer and hence vocab:
processor = AutoProcessor.from_pretrained(args.student_model_id, use_fast=True)

if'72b' in args.teacher_model_id:
    load_in_4bit = True
else:
    load_in_4bit = False

from transformers import BitsAndBytesConfig


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    args.teacher_model_id,
    use_flash_attention_2=False,
    quantization_config=quantization_config,
    device_map="auto"
)

from peft import PeftModel

model = PeftModel.from_pretrained(
    base_model,
    args.adapter_dir,
    device_map="auto",
    torch_dtype=torch.float16
)

model.config.use_cache = True

images_dir = "/gpfs/home/ym621/gavin/Hypo3D/hypo_dataset/dataset/top_view_with_label_rotated"

SYSTEM_INSTRUCTIONS = """
You are a visual reasoning assistant.

When I ask a question, you must consider the scene, the context change then give an answer.

The answer should be a single word or short phrase.
"""

def run_model_inference(phase: str,
                        model_vis_obs: str = None,
                        context_change_obs: str = None,
                        orientation: str = None,
                        change: str = None,
                        question: str = None):

    # pick the right template
    if phase == "visual":
        text_prompt = visual_obs_template.format(orientation=orientation)
    elif phase == "change":
        text_prompt = context_change_template.format(
            model_vis_obs=model_vis_obs,
            change=change
        )
    elif phase == "question":  # phase == "question"
        text_prompt = question_template.format(
            model_vis_obs=model_vis_obs,
            change=change,
            context_change_obs=context_change_obs,
            question=question
        )
    else: # phase == "retry"
        text_prompt = retry_prefix + question_template.format(
            model_vis_obs=model_vis_obs,
            change=change,
            context_change_obs=context_change_obs,
            question=question
        )

    print(f"The prompt is: {text_prompt}")

    messages = [
        { "role": "system", "content": SYSTEM_INSTRUCTIONS },
        {
            "role": "user",
            "content": [
                { "type": "image", "image": local_image },
                { "type": "text",  "text":  text_prompt }
            ]
        }
    ]

    # prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(images=local_image, text=prompt, return_tensors="pt").to("cuda:0", torch.float16)

    # Move inputs to GPU
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

    # Inference: Generation of the outputs
    # Possible that max_new_tokens isn't high enough for complex scenes, so might have to increase.
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=256,
        return_dict_in_generate=True,
        output_scores=True,        
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        early_stopping=True,
        num_beams=2, 
    )

    # generated_ids_trimmed = [
    #     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    # ]
    
    # output_text = processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )[0].strip()

    # so (batch_size, prompt_len + gen_len) tensor storing generated token IDs
    sequences = generated_ids.sequences

    # so list of gen_len of tensor (batch_size, vocab_size) so gives raw logits for each word in vocab for i_th position generated token
    scores = generated_ids.scores

    # end of prompt len
    prompt_len = inputs["input_ids"].size(1)

    # for the generated token IDs trim of the prompt tokens
    answer_ids = sequences[:, prompt_len:] # (batch_size, gen_len)

    # Added this to use answer_ids from generated_ids to decode
    output_text = processor.batch_decode(
        answer_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    # stack scores into a single tensor (batch_size, gen_len, vocab_size)
    all_logits = torch.stack(scores, dim=1)

    # slice off batch-dim and move to CPU
    answer_logits = all_logits[0, :, :].cpu() # so (gen_len, vocab_size) tensor moved to cpu so for each generated i_th token it gives logits for entire vocab
    # then we put this through softmax to generate probabilities

    answer_token_ids = answer_ids[0].cpu() # remove batch_size dim, so only 1D tensor of gen_len so this is a list of the token IDs for the generated output and move to cpu

    return output_text, answer_token_ids, answer_logits

def single_pass(prompt):
    return run_model_inference(prompt)
    
def convert_words_to_digits(text):
    words = text.split()
    converted_words = []
    for word in words:
        try:
            # Attempt to convert the word to a number
            number = w2n.word_to_num(word)
            converted_words.append(str(number))
        except ValueError:
            # If the word is not a number, keep it as is
            converted_words.append(word)
    return ' '.join(converted_words)
    
def normalize_text(text):
    """
    Normalize the input text by converting to lowercase, removing certain words/phrases,
    replacing specific terms, removing punctuation, and converting words to digits.
    """
    # Convert to lowercase
    text = text.lower()

    # Define replacements for specific terms
    replacements = {
        'back and right': 'back right',
        'back and left': 'back left',
        'front and right': 'front right',
        'front and left': 'front left',
        'behind and to the right': 'back right',
        'behind and to the left': 'back left',
        'in front and to the right': 'front right',
        'to the': '',
        'by the': '',
        'on the': '',
        'near': '',
        'next': '',
        'corner': '',
        'behind': 'back',
        'bottom': 'back',
        'top': 'front',
        'right side': 'right',
        'left side': 'left',
        'front side': 'front',
        'back side': 'back',
        'in front of': 'front',
        'on the left of': 'left',
        'on the right of': 'right',
        'on the left': 'left',
        'on the right': 'right',
        'north': 'front',
        'south': 'back',
        'east': 'right',
        'west': 'left',
        'northwest': 'front left',
        'northeast': 'front right',
        'southwest': 'back left',
        'southeast': 'back right',
        'forward': 'front',
        'backward': 'back',
        'bottom of': 'back',
        "left of": 'left',
        "right of": 'right',
        "front of": 'front',
        "back of": 'back'
    }
        
    # Use regex for efficient replacements
    sorted_replacements = sorted(replacements.keys(), key=len, reverse=True)
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, sorted_replacements)) + r')\b')
    text = pattern.sub(lambda match: replacements[match.group(0)], text)
    # Remove articles (e.g., "a", "an", "the")
    text = re.sub(r'\b(?:a|an|the)\b', '', text).strip()

    # Remove punctuation except letters, digits, and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Convert number words to digits (if applicable)
    text = convert_words_to_digits(text)

    return text
    
def f1_score(predicted, reference):
    pred_tokens = set(predicted.split())
    ref_tokens = set(reference.split())

    common_tokens = pred_tokens.intersection(ref_tokens)
    if len(common_tokens) == 0:
        return 0

    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(ref_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def partial_match_score(predicted, reference):
    pred_tokens = predicted.split()
    ref_tokens = reference.split()
    common_tokens = set(pred_tokens).intersection(set(ref_tokens))
    return len(common_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0

# Text prompt

visual_obs_template = '''
Given a top-view of a 3D scene, mentally rotate the image to align with the specified orientation.

Scene Orientation: {orientation}

Now, describe everything you see, listing each object and its location relative to the scene’s center (e.g. “The couch is at the back‐left corner, the plant is near the front‐right edge).

'''

context_change_template = '''
The visual observations you determined for the scene are: {model_vis_obs}.

Now, given a context change, describe how the scene looks after the change has been applied. List any changed or removed objects and their positions.

Context Change: {change}

'''

question_template = '''
Answer this question: 

Based on the visual observations: {model_vis_obs}

Context change: {change}

Scene after change: {context_change_obs}

Question: {question}

The answer should be a single word or short phrase.

The answer is:
'''

retry_prefix = '''
    The previous output had an empty answer.
    Please make sure an answer is given.
'''

# Additional imports for defining sub-question types
def get_sub_question_type(question):
    if question.lower().startswith("what"):
        return "What"
    elif question.lower().startswith("is"):
        return "Is"
    elif question.lower().startswith("how"):
        return "How"
    elif question.lower().startswith("can"):
        return "Can"
    elif question.lower().startswith("which"):
        return "Which"
    elif question.lower().startswith("does"):
        return "Does"
    elif question.lower().startswith("are"):
        return "Are"
    elif question.lower().startswith("where"):
        return "Where"
    else:
        return "Others"
    
# Metrics initialization per question type
total_questions = 0
exact_matches = 0
partial_match_scores = []
bert_precision_scores = []
bert_recall_scores = []
bert_f1_scores = []
total_questions_per_type = {}
exact_matches_per_type = {}
partial_match_scores_per_type = {}
bert_precision_scores_per_type = {}
bert_recall_scores_per_type = {}
bert_f1_scores_per_type = {}

df = pd.read_excel("/gpfs/home/ym621/gavin/Hypo3D/hypo_dataset/axis_def_hypo.xlsx", sheet_name='Sheet1', engine='openpyxl')

# Main loop
overall_qs_idx = 0
for scene_id, changes_list in tqdm(list(data.items())):
    image_path = os.path.join(images_dir, f"{scene_id}.png")
    local_image = Image.open(image_path)
            
    scene_orientation = extract_non_nan_values(df[df['scene_id'] == scene_id])
    scene_orientation = " ".join(
        f"The {item} was located at the {direction.lower()} of the scene."
        for scene_id, directions in scene_orientation.items()
        for direction, item in directions.items()
    )

    # For a new scene ask about the visual observations of the scene
    model_vis_obs, vis_obs_token_ids, vis_obs_logits = run_model_inference("visual", orientation=scene_orientation)

    print(f"Scene visual observations: {model_vis_obs}")

    for i, changes in tqdm(enumerate(changes_list)):
        # changes_list is list of dictionaries
        # i is current for loop index, changes is dictionary at index i of the list of dictionaries (changes_list)
        # so access a particular kv pair in a changes dict by doing: changes[key_string]
        context_change = changes['context_change']
        question_answers = changes['questions_answers']

        # Ask about the changes to a scene after a context change is applied
        # Phase 2a: context change
        context_change_obs, change_obs_token_ids, change_obs_logits = run_model_inference("change",
                                                model_vis_obs=model_vis_obs,
                                                change=context_change)

        print(f"Scene after context change observations: {context_change_obs}")
        
        for j, qa in tqdm(enumerate(question_answers)):
            overall_qs_idx += 1
            # j is index of current question-answer pair
            # qa is dictionary at index j of the list of dictionaries of question-answers (question_answers)
            # so access a particular question answer kv pair in the question_answers dict by doing: qa[key_string]
            question_type = qa['question_type']
            question = qa['question']
            answer = qa['answer']

            print("-" * 80)

            output_json = None
            raw_output = None
            attempt = 0

            cot_text = None
            answer_text = None

            prompt_status = "question"

            # Keep repeat prompting question until valid, non-empty output is given    
            while True:
                print(f"Attempt: {attempt}.\n")

                answer_text, answer_token_ids, answer_logits = run_model_inference(prompt_status,
                                            model_vis_obs=model_vis_obs,
                                            context_change_obs=context_change_obs,
                                            change=context_change,
                                            question=question)

                print(f"Model output: {answer_text}")

                answer_text = answer_text.strip()

                # empty outptu check and retry logic
                if answer_text == "":
                    if (attempt > 2):
                        break
                    # else retry
                    prompt_status = "retry"
                    attempt += 1

                else:
                    break

            model_answer = answer_text

            # print(f"Model reasoning is: {cot_text}")
            print(f"The model output is: {answer_text}")
            print(f"The reference answer is: {answer}")

            # Store fields in json file
            qa["question_number"] = overall_qs_idx
            qa["visual_observations"] = model_vis_obs
            qa["after_context_change"] = context_change_obs
            qa["model_answer"] = model_answer

            # Store visual_observation logits
            os.makedirs(f"{args.pipeline_run_dir}/teacher_cache/{overall_qs_idx}", exist_ok=True)
            torch.save({"token_ids": vis_obs_token_ids, "logits": vis_obs_logits}, f"{args.pipeline_run_dir}/teacher_cache/{overall_qs_idx}/visual_observations.pt")

            # Store context_change logits
            torch.save({"token_ids": change_obs_token_ids, "logits": change_obs_logits}, f"{args.pipeline_run_dir}/teacher_cache/{overall_qs_idx}/context_change.pt")

            # Store answer logits
            torch.save({"token_ids": answer_token_ids, "logits": answer_logits}, f"{args.pipeline_run_dir}/teacher_cache/{overall_qs_idx}/answer.pt")
            
            # Metrics calculation
            predicted_answer = normalize_text(model_answer)
            reference_answer = normalize_text(answer)
            
            print(f'Processed scene {scene_id}, change {i + 1}, question {j + 1}')

            print("-" * 80)

            # Initialize metrics for new question types
            if question_type not in total_questions_per_type:
                total_questions_per_type[question_type] = 0
                exact_matches_per_type[question_type] = 0
                partial_match_scores_per_type[question_type] = []

                bert_precision_scores_per_type[question_type] = []
                bert_recall_scores_per_type[question_type] = []
                bert_f1_scores_per_type[question_type] = []

            # Exact Match
            if predicted_answer == reference_answer:
                exact_matches += 1  
                exact_matches_per_type[question_type] += 1

            # Partial Match Score
            partial_match = partial_match_score(predicted_answer, reference_answer)
            partial_match_scores.append(partial_match)
            partial_match_scores_per_type[question_type].append(partial_match)

            # Computing BERTScore and getting precision, recall and F1
            P, R, F1 = bert_score([predicted_answer], [reference_answer], lang="en", verbose=False)

            # Extracting individual scores
            bert_precision = P[0].item()
            bert_recall = R[0].item()
            bert_f1 = F1[0].item()

            print(f"The BERT precision score: {bert_precision:.4f}")
            print(f"The BERT recall score: {bert_recall:.4f}")
            print(f"The BERT f1 score: {bert_f1:.4f}")

            bert_precision_scores.append(bert_precision)
            bert_precision_scores_per_type[question_type].append(bert_precision)
            
            bert_recall_scores.append(bert_recall)
            bert_recall_scores_per_type[question_type].append(bert_recall)

            bert_f1_scores.append(bert_f1)
            bert_f1_scores_per_type[question_type].append(bert_f1)

            total_questions += 1
            total_questions_per_type[question_type] += 1

    model_name = args.teacher_model_id.replace("/", "_")
    # need to actually write to project dir
    save_json(data, f"{model_name}_cot_gen_train_data.json", args.pipeline_run_dir)

question_type_average_scores = {}   

# Calculate average metrics for each question type
for question_type in total_questions_per_type:
    exact_match_score_per_type = (exact_matches_per_type[question_type] / total_questions_per_type[question_type]) * 100
    average_partial_match_per_type = sum(partial_match_scores_per_type[question_type]) / len(partial_match_scores_per_type[question_type]) * 100
    
    average_bert_precision_scores_per_type = sum(bert_precision_scores_per_type[question_type]) / len(bert_precision_scores_per_type[question_type])
    average_bert_recall_scores_per_type = sum(bert_recall_scores_per_type[question_type]) / len(bert_recall_scores_per_type[question_type])
    average_bert_f1_scores_per_type = sum(bert_f1_scores_per_type[question_type]) / len(bert_f1_scores_per_type[question_type])

    # Print results for each question type
    print(f"Question Type: {question_type}")
    print(f"  Exact Match Score: {exact_match_score_per_type:.2f}%")
    print(f"  Partial Match Score: {average_partial_match_per_type:.2f}%")
    print(f"  BERT Precision Score: {average_bert_precision_scores_per_type*100:.4f}%")
    print(f"  BERT Recall Score: {average_bert_recall_scores_per_type*100:.4f}%")
    print(f"  BERT F1 Score: {average_bert_f1_scores_per_type*100:.4f}%")

    question_type_average_scores[question_type] = [("Exact Match Score: ", exact_match_score_per_type), 
                                                   ("Partial Match Score: ", average_partial_match_per_type), ("BERT Precision Score: ", average_bert_precision_scores_per_type*100), 
                                                   ("BERT Recall Score: ", average_bert_recall_scores_per_type*100), 
                                                   ("BERT F1 Score: ", average_bert_f1_scores_per_type*100)]

# Calculate overall average metrics
exact_match_score = (exact_matches / total_questions) * 100
average_partial_match_score = sum(partial_match_scores) / len(partial_match_scores) * 100

average_bert_precision_score = sum(bert_precision_scores) / len(bert_precision_scores)
average_bert_recall_score = sum(bert_recall_scores) / len(bert_recall_scores) 
average_bert_f1_score = sum(bert_f1_scores) / len(bert_f1_scores)

# Print overall results and write to text file
print("\nOverall Metrics:")
print(f"Exact Match Score: {exact_match_score:.2f}%")
print(f"Partial Match Score: {average_partial_match_score:.2f}%")
print(f"BERT Precision Score: {average_bert_precision_score*100:.4f}%")
print(f"BERT Recall Score: {average_bert_recall_score*100:.4f}%")
print(f"BERT F1 Score: {average_bert_f1_score*100:.4f}%")

# Prepare filename (sanitize teacher_model_id for a path)
model_name = args.teacher_model_id.replace("/", "_")
os.makedirs(args.pipeline_run_dir, exist_ok=True)

results_path = os.path.join(args.pipeline_run_dir, f"{model_name}_cot_gen_results")

# Use an f-string so args.teacher_model_id is filled in
with open(results_path, "w") as fout:
    fout.write("Overall Metrics:\n")
    fout.write(f"Exact Match Score: {exact_match_score:.2f}%\n")
    fout.write(f"Partial Match Score: {average_partial_match_score:.2f}%\n")
    fout.write(f"BERT Precision Score: {average_bert_precision_score*100:.4f}%\n")
    fout.write(f"BERT Recall Score: {average_bert_recall_score*100:.4f}%\n")
    fout.write(f"BERT F1 Score: {average_bert_f1_score*100:.4f}%\n")

    fout.write("Question Type Metrics:\n")
    # Write all question type specific scores
    for question_type, question_type_scores in question_type_average_scores.items():
        fout.write(f"Question Type: {question_type}")
        for score_type, score in question_type_scores:
            fout.write(f"{score_type}: {score:.4f}%\n")

print(f"\nWrote overall metrics to {results_path}")
