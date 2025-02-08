import torch
from process_opinions_qa import generate_prompt
from transformers import DynamicCache
import random

def generate_logprob_dictionary(logprobs, tokenizer, k=10):
    res_dict=dict()
    topklogprobs=logprobs.topk(k=k)
    logprob_vals=topklogprobs.values
    logprob_inds=topklogprobs.indices
    
    res_dict['text']=tokenizer.decode([logprob_inds[0].item()])
    res_dict['logprob']=logprob_vals[0].item()
    res_dict['top_k_logprobs']=dict()
    for i in range(k):
        res_dict['top_k_logprobs'][tokenizer.decode([logprob_inds[i].item()])]= \
            logprob_vals[i].item()
        
    return res_dict


def generate_responses_logprobs(prompt, model, tokenizer):
    stopping_criteria = None
    raw_request={
        "prompt":prompt,
        "stop_sequences": [],
        "temperature":1e-7,
        "max_new_tokens":1,
        "top_p":1,
        "num_return_sequences":1,
        "echo_prompt":False,
    }
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    encoded_input = tokenizer(raw_request["prompt"], return_tensors="pt", return_token_type_ids=False).to(device)
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    
    with torch.no_grad():
        output = model.generate(
                        **encoded_input,
                        temperature=None,
                        num_return_sequences=raw_request["num_return_sequences"],
                        max_new_tokens=raw_request["max_new_tokens"],
                        top_p=raw_request["top_p"],
                        do_sample=False,
                        return_dict_in_generate=True,
                        output_scores=True,
                        stopping_criteria=stopping_criteria,
        #                 top_k=raw_request["top_k"],
                    )
    sequences = output.sequences
    scores = output.scores

    # print(tokenizer.decode(sequences[0]))

    # calculate log probs when scores do not return inf
    if torch.isinf(scores[0]).sum().item()==0:

        # Compute logprobs of generated tokens for each completed sequence.
        all_generated_tokens_logprobs = []
        for completion_id in range(raw_request["num_return_sequences"]):
    #         print(f'completion id: {completion_id}')
            generated_tokens_logprobs = []
            for i in range(len(sequences[completion_id]) - len(encoded_input.input_ids[0])):
    #             print(f'i: {i}')
                logprobs = torch.nn.functional.log_softmax(scores[i][completion_id], dim=0)
    #             print(f'logprobs: {len(logprobs)}')
                # Get log probability of chosen token.
                j = i + len(encoded_input.input_ids[0])
    #             print(f'j: {j}')
                generated_tokens_logprobs.append(logprobs[sequences[completion_id][j]].item())
            all_generated_tokens_logprobs.append(generated_tokens_logprobs)

        # print(f'Log Probs: {all_generated_tokens_logprobs}')
    else:
        print('Log Probs not calculated')
        
    
    res_dict=generate_logprob_dictionary(logprobs, tokenizer)
    return res_dict

# ref: https://gist.github.com/vgel/8a2497dc45b1ded33287fa7bb6cc1adc
def get_reasoning_response(prompt, model, tokenizer, if_print=True):
    _, _start_think_token, end_think_token = tokenizer.encode("<think></think>")
    
    replacements=["\nWait, but", "\nHmm", "\nSo"]
    
    think_responses=list()
    for chunk in reasoning_effort(prompt, 128, model, tokenizer, replacements,end_think_token):
        if if_print:
            print(chunk, end="", flush=True)
        think_responses.append(chunk)
    return ''.join(think_responses)

@torch.inference_mode
def reasoning_effort(question: str, min_thinking_tokens: int, model, tokenizer, replacements, end_think_token):
    tokens = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "<think>\n" + ""},
        ],
        continue_final_message=True,
        return_tensors="pt",
    )
    tokens = tokens.to(model.device)
    kv = DynamicCache()
    n_thinking_tokens = 0

    print(tokenizer.decode(list(tokens[0])))
    while True:
        out = model(input_ids=tokens, past_key_values=kv, use_cache=True)
        next_token = torch.multinomial(
            torch.softmax(out.logits[0, -1, :], dim=-1), 1
        ).item()
        kv = out.past_key_values

        if (
            next_token in (end_think_token, model.config.eos_token_id)
            and n_thinking_tokens < min_thinking_tokens
        ):
            replacement = random.choice(replacements)
            print(replacement)
            replacement_tokens = tokenizer.encode(replacement)
            n_thinking_tokens += len(replacement_tokens)
            tokens = torch.tensor([replacement_tokens]).to(tokens.device)
        elif next_token == model.config.eos_token_id:
            break
        else:
            dec_token = tokenizer.decode([next_token])
            yield tokenizer.decode([next_token])
            n_thinking_tokens += 1
            tokens = torch.tensor([[next_token]]).to(tokens.device)