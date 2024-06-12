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

    system_instruction_1 = "You are a professional fact checker. You will be provided with a claim and some evidences related to the claim. Your task is to classify the claim into the following three labels: Supported, Refuted, or Unsure. You need to first analyze the given the claim and the evidences as a fact checker, and then derive the classification decision based on your analysis. Your final classification decision should solely rely on the claim and the evidences provided, but not the pre-trained knowledge you have learned. \nNote that the original label space is very large, so we grouped all the original labels into Supported, Refuted, and Unsure based on the following guidelines: \n1. We grouped labels like 'mostly true', 'partially true' into 'Supported'. Here is all the original labels mapped to 'Supported': 'a little baloney', 'accurate', 'authorship confirmed!', 'conclusion: accurate', 'confirmed authorship!', 'correct', 'correct attribution', 'correct attribution!', 'determination: mostly true', 'determination: true', 'fact', 'factscan score: true', 'half true', 'half-true', 'in-the-green', 'mostly true', 'mostly truth!', 'mostly-correct', 'mostly_true', 'no flip', 'outdated', 'partially true', 'partly true', 'true', 'truth!', 'verdict: true', 'verified'.\n2. We grouped labels like 'mostly false', 'partially false' into 'Refuted'. Here is all the original labels mapped to 'Refuted': 'a lot of baloney', 'cherry picks', 'conclusion: false', 'determination: barely true', 'determination: false', 'determination: huckster propaganda', 'determination: misleading', 'disputed!', 'distorts the facts', 'exaggerated', 'exaggerates', 'factscan score: false', 'factscan score: misleading', 'fake', 'false', 'fiction', 'fiction!', 'full flop', 'half flip', 'in-the-red', 'inaccurate attribution!', 'incorrect', 'incorrect attribution!', 'legend', 'misattributed', 'miscaptioned', 'misleading', 'misleading!', 'mixture', 'mostly false', 'mostly fiction!', 'mostly_false', 'not the whole story', 'pants on fire!', 'scam', 'scam!', 'some baloney', 'spins the facts', 'understated', 'unsupported', 'verdict: false', 'we rate this claim false'\n3. Here is all the labels mapped to 'Unsure': 'conclusion: unclear', 'in-between', 'no evidence', 'unobservable', 'unproven', 'unproven!', 'unverified', 'verdict: unsubstantiated'. \n I will ask you to output in two rounds. In the first round, you need to output the process of analyzing the claim and the evidences as a fact checker, and in the second round, you need to output the final classification decision based on your analysis. Now, this is the first round. Please analyze the claim and the evidences as a fact checker and output the process of your analysis."

    system_instruction_2 = "Now, this is the second round. Based on the above analysis you have done, output your final classification decision in only one token. Your response should be one of the following: 'Supported', 'Refuted', or 'Unsure'."

    # load json file ./data/MultiFC/dev_prompt_data_v1.0.json
    with open('./data/MultiFC/dev_prompt_data_v1.0.json', 'r') as f:
        dev_data = json.load(f)
    
    # Iterate through the list in increments of four elements
    for i in tqdm(range(0, len(dev_data), 4)):
        batch = dev_data[i:i+4]
        if len(batch) < 4:
            continue

        # dialog round 1
        dialogs_1: List[Dialog] = [
            [
                {"role": "system", "content": system_instruction_1},
                {"role": "user", "content": batch[0]['prompt']},
            ],
            [
                {"role": "system", "content": system_instruction_1},
                {"role": "user", "content": batch[1]['prompt']},
            ],
            [
                {"role": "system", "content": system_instruction_1},
                {"role": "user", "content": batch[2]['prompt']},
            ],
            [
                {"role": "system", "content": system_instruction_1},
                {"role": "user", "content": batch[3]['prompt']},
            ],
        ]
        results_1 = generator.chat_completion_with_factcheck_labels(
            dialogs_1,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=True, 
        )
        # results_1[0]['generation'] is the model generatinon for only this round: {"role": "assistant", "content": "real generation"}

        # dialog round 2
        dialogs_2: List[Dialog] = [
            [
                {"role": "system", "content": system_instruction_1},
                {"role": "user", "content": batch[0]['prompt']},
                results_1[0], 
                {"role": "system", "content": system_instruction_2}, 
            ],
            [
                {"role": "system", "content": system_instruction_1},
                {"role": "user", "content": batch[1]['prompt']},
                results_1[1], 
                {"role": "system", "content": system_instruction_2}, 
            ],
            [
                {"role": "system", "content": system_instruction_1},
                {"role": "user", "content": batch[2]['prompt']},
                results_1[2], 
                {"role": "system", "content": system_instruction_2}, 
            ],
            [
                {"role": "system", "content": system_instruction_1},
                {"role": "user", "content": batch[3]['prompt']},
                results_1[3], 
                {"role": "system", "content": system_instruction_2}, 
            ],
        ]
        results_2 = generator.chat_completion_with_factcheck_labels(
            dialogs_2,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=True, 
        )

        # save the classification result from results_2 to the dev_data
        dev_data[i]['orig_probs'] = results_2[0]['factcheck_label_probs'].copy()
        dev_data[i+1]['orig_probs'] = results_2[1]['factcheck_label_probs'].copy()
        dev_data[i+2]['orig_probs'] = results_2[2]['factcheck_label_probs'].copy()
        dev_data[i+3]['orig_probs'] = results_2[3]['factcheck_label_probs'].copy()

        # normalize the probabilities using softmax
        for j in range(len(results_2)):
            unnormalized_probs = [results_2[j]['factcheck_label_probs']['Supported'], results_2[j]['factcheck_label_probs']['Refuted'], results_2[j]['factcheck_label_probs']['Unsure']]
            normalized_probs = softmax(unnormalized_probs)
            results_2[j]['factcheck_label_probs']['Supported'] = normalized_probs[0]
            results_2[j]['factcheck_label_probs']['Refuted'] = normalized_probs[1]
            results_2[j]['factcheck_label_probs']['Unsure'] = normalized_probs[2]
        
        # unnormalized_probs = [results_2[0]['factcheck_label_probs']['Supported'], results_2[0]['factcheck_label_probs']['Refuted'], results_2[0]['factcheck_label_probs']['Unsure']]
        # normalized_probs = softmax(unnormalized_probs)
        # results_2[0]['factcheck_label_probs']['Supported'] = normalized_probs[0]
        # results_2[0]['factcheck_label_probs']['Refuted'] = normalized_probs[1]
        # results_2[0]['factcheck_label_probs']['Unsure'] = normalized_probs[2]

        # unnormalized_probs = [results_2[1]['factcheck_label_probs']['Supported'], results_2[1]['factcheck_label_probs']['Refuted'], results_2[1]['factcheck_label_probs']['Unsure']]
        # normalized_probs = softmax(unnormalized_probs)
        # results_2[1]['factcheck_label_probs']['Supported'] = normalized_probs[0]
        # results_2[1]['factcheck_label_probs']['Refuted'] = normalized_probs[1]
        # results_2[1]['factcheck_label_probs']['Unsure'] = normalized_probs[2]

        # unnormalized_probs = [results_2[2]['factcheck_label_probs']['Supported'], results_2[2]['factcheck_label_probs']['Refuted'], results_2[2]['factcheck_label_probs']['Unsure']]
        # normalized_probs = softmax(unnormalized_probs)
        # results_2[2]['factcheck_label_probs']['Supported'] = normalized_probs[0]
        # results_2[2]['factcheck_label_probs']['Refuted'] = normalized_probs[1]
        # results_2[2]['factcheck_label_probs']['Unsure'] = normalized_probs[2]

        # unnormalized_probs = [results_2[3]['factcheck_label_probs']['Supported'], results_2[3]['factcheck_label_probs']['Refuted'], results_2[3]['factcheck_label_probs']['Unsure']]
        # normalized_probs = softmax(unnormalized_probs)
        # results_2[3]['factcheck_label_probs']['Supported'] = normalized_probs[0]
        # results_2[3]['factcheck_label_probs']['Refuted'] = normalized_probs[1]
        # results_2[3]['factcheck_label_probs']['Unsure'] = normalized_probs[2]

        # keep saving the rest of the classification result from results_2 to the dev_data
        dev_data[i]['generated_label'] = results_2[0]['tokens'][0]
        dev_data[i+1]['generated_label'] = results_2[1]['tokens'][0]
        dev_data[i+2]['generated_label'] = results_2[2]['tokens'][0]
        dev_data[i+3]['generated_label'] = results_2[3]['tokens'][0]

        dev_data[i]['softmax_probs'] = results_2[0]['factcheck_label_probs']
        dev_data[i+1]['softmax_probs'] = results_2[1]['factcheck_label_probs']
        dev_data[i+2]['softmax_probs'] = results_2[2]['factcheck_label_probs']
        dev_data[i+3]['softmax_probs'] = results_2[3]['factcheck_label_probs']

        # save the justification from results_1 to the dev_data
        dev_data[i]['generation'] = results_1[0]['generation']['content']
        dev_data[i+1]['generation'] = results_1[1]['generation']['content']
        dev_data[i+2]['generation'] = results_1[2]['generation']['content']
        dev_data[i+3]['generation'] = results_1[3]['generation']['content']


        # save dev_data to a file
        with open('./data/MultiFC/dev_prompt_data_v1.0_generated_AddMappingRules_explanation_classification.json', 'w') as f:
            json.dump(dev_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    fire.Fire(main)d
