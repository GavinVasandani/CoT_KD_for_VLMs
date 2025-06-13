# Script to visualize logits by doing softmax on logits for an output and seeing if the highest probability token matches output generated

import torch
from transformers import AutoTokenizer
import argparse

model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

parser = argparse.ArgumentParser()
parser.add_argument("-rl", "--raw_logits_path", type=str, default="teacher_cache/1.pt", help="Path to raw_logits to visualize")
args = parser.parse_args()

# Load in raw logits and token IDs for a particular question
meta_d = torch.load(args.raw_logits_path)

# raw_logits is 2D tensor of shape: [tokens_generated, vocab_size]
# so each token generated has list of size [vocab_size] which gives logit for each possible vocab
# the actual generated token for is determined by converting these logits to probabilities via. softmax
# then choosing the top probability in the vocab for the token is the generated token
raw_logits = meta_d["logits"]

# tensor of shape: [tokens_generated]
# so tensor size is the number of tokens generated for the output
token_ids = meta_d["token_ids"]

# convert raw logits to probabilities
probs = torch.softmax(raw_logits, dim=-1)
# probs is 2D tensor where probs[i] is the probabilities of the vocab for the i_th generated token
# the probabilities for a token i so all vals in probs[i] should add to 1

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Get the highest prob tokens for i_th token generated and compare against actual token id
# Vis actual token ids and top 5 alternatives derived from raw logits for all output generated tokens (raw_logits.shape[0])
for i in range(raw_logits.shape[0]):
    # topk is kv pair of: k - the top 5 probabilties, and v the corresponding token IDs that have these top 5 probs
    topk = probs[i].topk(5)

    # actual generated token id
    chosen_id = token_ids[i].item()
    
    # decoding the generated token id into an actual token
    chosen_token_str = tokenizer.decode([chosen_id])

    print(f"From token_ids: Token {i} is: {chosen_token_str}")
    print(f"The top 5 alternatives for the {i}_th generated token derived from raw_logits is:")
    count = 0
    for score, idx in zip(topk.values, topk.indices):
        count += 1
        # so token_ids derived from raw logits of the i_th generated token
        raw_token_id = idx.item()
        raw_str = tokenizer.decode([raw_token_id])
        print(f"The {count}_th alternative token: {raw_str}")


 