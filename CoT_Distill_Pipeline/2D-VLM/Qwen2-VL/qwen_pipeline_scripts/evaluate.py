from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import json
import os
import re
from word2number import w2n
import argparse
import pandas as pd
from bert_score import score as bert_score

def save_json(data, filename, pipeline_run_dir):
    os.makedirs(pipeline_run_dir, exist_ok=True)
    full_path = os.path.join(pipeline_run_dir, filename)
    with open(full_path, "w") as file:
        json.dump(data, file, indent=4)
    print(f"Saved predictions to {full_path}")
        
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
# file_path = "dataset/ContextQA/Axis Definition.xlsx"
columns_to_load = ["scene_id"]  # Change if needed

# df = pd.read_excel(file_path, sheet_name='Sheet1', engine='openpyxl')


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--data_path", type=str, default="reason3d/recategorize_test_data.json", help="Path to the JSON file containing the test data")
parser.add_argument("-m", "--model_id", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="Model ID of the model to be evaluated")
parser.add_argument("-p", "--pipeline_run_dir", type=str, required=True, help="Path to directory to save output files, e.g., pipeline_run/")
args = parser.parse_args()
# Load the model and move it to the GPU
model = Qwen2VLForConditionalGeneration.from_pretrained(
    args.model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set min and max pixels
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=min_pixels, max_pixels=max_pixels)

# Load a local image
data = json.load(open(args.data_path))
# images_dir = "/gpfs/home/ym621/dataset/2D_VLM_data/top_view_no_label"
images_dir = "/gpfs/home/ym621/gavin/Hypo3D/hypo_dataset/dataset/top_view_with_label_rotated"

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
template = '''
Given a top-view of a 3D scene, mentally rotate the image to align with the specified orientation.

Scene Orientation: {}

Now, given a context change, imagine how the scene would look after the change has been applied. Then, answer a question based on the changed scene.

Context Change: {}
Question: {}

The answer should be a single word or short phrase.

The answer is:
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

# read xlxs file
# df = pd.read_excel("Axis Definition.xlsx", sheet_name='Sheet1', engine='openpyxl')
df = pd.read_excel("/gpfs/home/ym621/gavin/Hypo3D/hypo_dataset/axis_def_hypo.xlsx", sheet_name='Sheet1', engine='openpyxl')

# Main loop
for scene_id, changes_list in list(data.items()):
    image_path = os.path.join(images_dir, f"{scene_id}.png")
    local_image = Image.open(image_path)
            
    scene_orientation = extract_non_nan_values(df[df['scene_id'] == scene_id])
    scene_orientation = " ".join(
        f"The {item} was located at the {direction.lower()} of the scene."
        for scene_id, directions in scene_orientation.items()
        for direction, item in directions.items()
    )
    
    for i, changes in enumerate(changes_list):
        context_change = changes['context_change']
        question_answers = changes['questions_answers']
        
        
        for j, qa in enumerate(question_answers):
            question_type = qa['question_type']
            question = qa['question']
            answer = qa['answer']
            
            # Prepare the prompt and inputs
            text_prompt = template.format(scene_orientation, context_change, question)
            
            # text_prompt = template.format(context_change, question)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": local_image},
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # Move inputs to GPU
            # inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

            # Inference: Generation of the outputs
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
            
            # Decode to get outputs
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
         
            qa['predicted_answer'] = output_text

            print("-" * 80)
            print(f"The question is: {question}")

            print(f"Reference answer is: {answer}")
            print(f"Model predicted answer is: {output_text}")
            
            print(f'Processed scene {scene_id}, change {i + 1}, question {j + 1}')
            
            print("-" * 80)

            # Metrics calculation
            predicted_answer = normalize_text(output_text)
            reference_answer = normalize_text(answer)
            
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
            
    # Not really needed anymore
    save_json(data, f"{args.model_id.split('/')[1]}_no_label_align.json", args.pipeline_run_dir)

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
print(f"BERT Precision Score: {average_bert_precision_score*100:.4f}")
print(f"BERT Recall Score: {average_bert_recall_score*100:.4f}%")
print(f"BERT F1 Score: {average_bert_f1_score*100:.4f}%")

# Prepare filename (sanitize model_id for a path)
model_name = args.model_id.replace("/", "_")
# Make sure pipeline_run_dir is an existing dir
os.makedirs(args.pipeline_run_dir, exist_ok=True)

# Path: pipeline_run_dir/model_name_zero_shot_evaluation_results.txt
results_path = os.path.join(args.pipeline_run_dir, f"{model_name}_zero_shot_evaluation_results.txt")

# Use an f-string so args.model_id is filled in
# output_path = f"{output_dir}/evaluation_overall_metrics_{model_name}_without_cot.txt"
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

