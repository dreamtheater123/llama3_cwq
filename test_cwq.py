# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

# from typing import List, Optional
# import fire
# from llama import Dialog, Llama
# import json
# from tqdm import tqdm
# from scipy.special import softmax


# def main(
#     ckpt_dir: str,
#     tokenizer_path: str,
#     temperature: float = 0.6,
#     top_p: float = 0.9,
#     max_seq_len: int = 8192,
#     max_batch_size: int = 4,
#     max_gen_len: Optional[int] = None,
# ):
#     """
#     Examples to run with the models finetuned for chat. Prompts correspond of chat
#     turns between the user and assistant with the final one always being the user.

#     An optional system prompt at the beginning to control how the model should respond
#     is also supported.

#     The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

#     `max_gen_len` is optional because finetuned models are able to stop generations naturally.
#     """
#     generator = Llama.build(
#         ckpt_dir=ckpt_dir,
#         tokenizer_path=tokenizer_path,
#         max_seq_len=max_seq_len,
#         max_batch_size=max_batch_size,
#     )

#     system_instruction = "You are a professional fact checker. You will be provided with a claim and evidence related to the claim. Your task is to classify the claim into the following three labels: Supported, Refuted, or Unsure. Your classification decision should solely rely on the claim and the evidences provided, but not the pre-trained knowledge you have learned. First, output the class that you predict at the first token, and then provide the reasoning behind your decision. Note that the first token should be one of the following: 'Supported', 'Refuted', or 'Unsure'."

#     # load json file ./data/MultiFC/dev_prompt_data_v1.0.json
#     with open('./data/MultiFC/dev_prompt_data_v1.0.json', 'r') as f:
#         dev_data = json.load(f)
    
#     # Iterate through the list in increments of four elements
#     for i in tqdm(range(0, len(dev_data), 4)):
#         batch = dev_data[i:i+4]
#         if len(batch) < 4:
#             continue

#         dialogs: List[Dialog] = [
#             [
#                 {"role": "system", "content": system_instruction},
#                 {"role": "user", "content": 'test case: generate the word "Supported" at the first token.'},
#             ],
#             [
#                 {"role": "system", "content": system_instruction},
#                 {"role": "user", "content": 'test case: generate the word "Refuted" at the first token.'},
#             ],
#             [
#                 {"role": "system", "content": system_instruction},
#                 {"role": "user", "content": 'test case: generate the word "Unsure" at the first token.'},
#             ],
#             [
#                 {"role": "system", "content": system_instruction},
#                 {"role": "user", "content": batch[3]['prompt']},
#             ],
#         ]
#         results = generator.chat_completion_with_factcheck_labels(
#             dialogs,
#             max_gen_len=max_gen_len,
#             temperature=temperature,
#             top_p=top_p,
#             logprobs=True, 
#         )

#         dev_data[i]['generation'] = results[0]['generation']['content']
#         dev_data[i+1]['generation'] = results[1]['generation']['content']
#         dev_data[i+2]['generation'] = results[2]['generation']['content']
#         dev_data[i+3]['generation'] = results[3]['generation']['content']

#         # obtain the logits of the first generated token
#         print(results[0]['generation']['content'])
#         print(results[0]['tokens'])
#         print(results[0]['logprobs'])
#         print(type(results[0]['tokens']), type(results[0]['logprobs']))
#         print(len(results[0]['generation']['content']), len(results[0]['tokens']), len(results[0]['logprobs']))
        
#         print(results[0]['generation']['content'])
#         print(results[0]['factcheck_label_probs'])
#         # perform softmax on results[0]['factcheck_label_probs']['Supported'], results[0]['factcheck_label_probs']['Refuted'], results[0]['factcheck_label_probs']['Unsure']
#         unnormalized_probs = [results[0]['factcheck_label_probs']['Supported'], results[0]['factcheck_label_probs']['Refuted'], results[0]['factcheck_label_probs']['Unsure']]
#         normalized_probs = softmax(unnormalized_probs)
#         results[0]['factcheck_label_probs']['Supported'] = normalized_probs[0]
#         results[0]['factcheck_label_probs']['Refuted'] = normalized_probs[1]
#         results[0]['factcheck_label_probs']['Unsure'] = normalized_probs[2]
#         print(normalized_probs, type(normalized_probs))

#         print(results[1]['generation']['content'])
#         print(results[1]['factcheck_label_probs'])
#         print(results[2]['generation']['content'])
#         print(results[2]['factcheck_label_probs'])

#         exit()

#         # # save the results to a file
#         # with open('./data/MultiFC/dev_prompt_llama3_generated.json', 'w') as f:
#         #     json.dump(dev_data, f, ensure_ascii=False, indent=4)


# if __name__ == "__main__":
#     fire.Fire(main)

a = 1
b = 2

c = a + b
if c == 3:
    print("c is equal to 3")
else:
    print("c is not equal to 3")

pass

