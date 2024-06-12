# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional
import fire
from llama import Dialog, Llama
import json
from tqdm import tqdm
from scipy.special import softmax


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 8192,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    system_instruction = "You are a professional fact checker. You will be provided with a claim and evidence related to the claim. Your task is to classify the claim into the following three labels: Supported, Refuted, or Unsure. Your classification decision should solely rely on the claim and the evidences provided, but not the pre-trained knowledge you have learned. First, output the class that you predict at the first token, and then provide the reasoning behind your decision. Note that the first token should be one of the following: 'Supported', 'Refuted', or 'Unsure'.\nNote that the original label space is very large, so we grouped all the original labels into Supported, Refuted, and Unsure based on the following guidelines: \n1. We grouped labels like 'mostly true', 'partially true' into 'Supported'. Here is all the original labels mapped to 'Supported': 'a little baloney', 'accurate', 'authorship confirmed!', 'conclusion: accurate', 'confirmed authorship!', 'correct', 'correct attribution', 'correct attribution!', 'determination: mostly true', 'determination: true', 'fact', 'factscan score: true', 'half true', 'half-true', 'in-the-green', 'mostly true', 'mostly truth!', 'mostly-correct', 'mostly_true', 'no flip', 'outdated', 'partially true', 'partly true', 'true', 'truth!', 'verdict: true', 'verified'.\n2. We grouped labels like 'mostly false', 'partially false' into 'Refuted'. Here is all the original labels mapped to 'Refuted': 'a lot of baloney', 'cherry picks', 'conclusion: false', 'determination: barely true', 'determination: false', 'determination: huckster propaganda', 'determination: misleading', 'disputed!', 'distorts the facts', 'exaggerated', 'exaggerates', 'factscan score: false', 'factscan score: misleading', 'fake', 'false', 'fiction', 'fiction!', 'full flop', 'half flip', 'in-the-red', 'inaccurate attribution!', 'incorrect', 'incorrect attribution!', 'legend', 'misattributed', 'miscaptioned', 'misleading', 'misleading!', 'mixture', 'mostly false', 'mostly fiction!', 'mostly_false', 'not the whole story', 'pants on fire!', 'scam', 'scam!', 'some baloney', 'spins the facts', 'understated', 'unsupported', 'verdict: false', 'we rate this claim false'\n3. Here is all the labels mapped to 'Unsure': 'conclusion: unclear', 'in-between', 'no evidence', 'unobservable', 'unproven', 'unproven!', 'unverified', 'verdict: unsubstantiated'."

    system_instruction_end = " Remember that the first generated token should be one of the following: 'Supported', 'Refuted', or 'Unsure'."

    # load json file ./data/MultiFC/dev_prompt_data_v1.0.json
    with open('./data/MultiFC/dev_prompt_data_v1.0.json', 'r') as f:
        dev_data = json.load(f)
    
    # Iterate through the list in increments of four elements
    for i in tqdm(range(0, len(dev_data), 4)):
        batch = dev_data[i:i+4]
        if len(batch) < 4:
            continue

        dialogs: List[Dialog] = [
            [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": batch[0]['prompt'] + system_instruction_end},
            ],
            [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": batch[1]['prompt'] + system_instruction_end},
            ],
            [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": batch[2]['prompt'] + system_instruction_end},
            ],
            [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": batch[3]['prompt'] + system_instruction_end},
            ],
        ]
        results = generator.chat_completion_with_factcheck_labels(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=True, 
        )

        dev_data[i]['orig_probs'] = results[0]['factcheck_label_probs'].copy()
        dev_data[i+1]['orig_probs'] = results[1]['factcheck_label_probs'].copy()
        dev_data[i+2]['orig_probs'] = results[2]['factcheck_label_probs'].copy()
        dev_data[i+3]['orig_probs'] = results[3]['factcheck_label_probs'].copy()

        unnormalized_probs = [results[0]['factcheck_label_probs']['Supported'], results[0]['factcheck_label_probs']['Refuted'], results[0]['factcheck_label_probs']['Unsure']]
        normalized_probs = softmax(unnormalized_probs)
        results[0]['factcheck_label_probs']['Supported'] = normalized_probs[0]
        results[0]['factcheck_label_probs']['Refuted'] = normalized_probs[1]
        results[0]['factcheck_label_probs']['Unsure'] = normalized_probs[2]

        unnormalized_probs = [results[1]['factcheck_label_probs']['Supported'], results[1]['factcheck_label_probs']['Refuted'], results[1]['factcheck_label_probs']['Unsure']]
        normalized_probs = softmax(unnormalized_probs)
        results[1]['factcheck_label_probs']['Supported'] = normalized_probs[0]
        results[1]['factcheck_label_probs']['Refuted'] = normalized_probs[1]
        results[1]['factcheck_label_probs']['Unsure'] = normalized_probs[2]

        unnormalized_probs = [results[2]['factcheck_label_probs']['Supported'], results[2]['factcheck_label_probs']['Refuted'], results[2]['factcheck_label_probs']['Unsure']]
        normalized_probs = softmax(unnormalized_probs)
        results[2]['factcheck_label_probs']['Supported'] = normalized_probs[0]
        results[2]['factcheck_label_probs']['Refuted'] = normalized_probs[1]
        results[2]['factcheck_label_probs']['Unsure'] = normalized_probs[2]

        unnormalized_probs = [results[3]['factcheck_label_probs']['Supported'], results[3]['factcheck_label_probs']['Refuted'], results[3]['factcheck_label_probs']['Unsure']]
        normalized_probs = softmax(unnormalized_probs)
        results[3]['factcheck_label_probs']['Supported'] = normalized_probs[0]
        results[3]['factcheck_label_probs']['Refuted'] = normalized_probs[1]
        results[3]['factcheck_label_probs']['Unsure'] = normalized_probs[2]

        dev_data[i]['generated_label'] = results[0]['tokens'][0]
        dev_data[i+1]['generated_label'] = results[1]['tokens'][0]
        dev_data[i+2]['generated_label'] = results[2]['tokens'][0]
        dev_data[i+3]['generated_label'] = results[3]['tokens'][0]

        dev_data[i]['softmax_probs'] = results[0]['factcheck_label_probs']
        dev_data[i+1]['softmax_probs'] = results[1]['factcheck_label_probs']
        dev_data[i+2]['softmax_probs'] = results[2]['factcheck_label_probs']
        dev_data[i+3]['softmax_probs'] = results[3]['factcheck_label_probs']

        dev_data[i]['generation'] = results[0]['generation']['content']
        dev_data[i+1]['generation'] = results[1]['generation']['content']
        dev_data[i+2]['generation'] = results[2]['generation']['content']
        dev_data[i+3]['generation'] = results[3]['generation']['content']

        print(dev_data[i])

        # save the results to a file
        with open('./data/MultiFC/dev_prompt_data_v1.0_generated_AddMappingRules.json', 'w') as f:
            json.dump(dev_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    fire.Fire(main)d
