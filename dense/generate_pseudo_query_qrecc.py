
import os
import torch
from dataclasses import dataclass, field
from typing import Optional
import json

import torch
from tqdm import tqdm, trange
from transformers import HfArgumentParser

import pandas as pd
import torch
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LlamaForCausalLM, MistralForCausalLM

tqdm.pandas()

nums = []
for i in range(1, 10):
    nums.append(str(i))
pattern = []
for i in range(1, 21):
    pattern.append(str(i) + '. ')
pattern.append("\"")
pattern.append(".:")

def get_prompt(pos_doc, cur_q, context=None, QR=False):
    '''
    <s>[INST] <<SYS>>
    You are given a chitchat conversation between a "User" and a "Bot". Your goal is to generate a 
    response to the last user turn, which in turn should be based on the given "Knowledge". 
    <</SYS>>
    prompt = "You are a query generator LLM, an intelligent assistant that can generate a search query for the given relevant passage based on their relevancy. The given relevant passage is:\n" + pos_doc
    '''
    if context is None:
        system_prompt = "<s>[INST] <<SYS>> \n You are a query generator LLM, an intelligent assistant that can generate a search query for the given relevant passage based on their relevancy. \n"
        user_prompt = "The given relevant passage is: \"" + pos_doc + "\"" + "\nThe output format should be in a list with index e.g., 1. 2. 3.. Only respond with the generation results, do not say any word or explain. [/INST]"
    else:
        if not QR:
            #system_prompt = "<s>[INST] <<SYS>> \n You are a conversational query generator LLM, an intelligent assistant that can generate a search query for the given relevant passage based on their relevancy and also rely on the given historical information-seeking multi-turn dialog context to satisfy current turn query's information need.  \n"
            #user_prompt = "The given relevant passage is: \"" + pos_doc + "\"" + "\nThe historical information-seeking multi-turn dialog context is: \"" + context + "\"" + "\nThe current turn query is:  \"" + cur_q + "\"" + "\nThe output format should be in a list with index e.g., 1. 2. 3.. Only respond with the generation results, do not say any word or explain. [/INST]"
            system_prompt = "<s>[INST] <<SYS>> \n You are a conversational query generator LLM, an intelligent assistant that can generate a search query for the given relevant passage based on their relevancy and also rely on the given historical information-seeking context to satisfy current turn question's information need. <</SYS>>\n"
            user_prompt = "The given relevant passage is: \"" + pos_doc + "\"" + "\nThe current question and historical information-seeking context is: \"" + context + "\"" + "\nThe output format should be in a list with index e.g., 1. 2. 3.. Only respond with the generation results, do not say any word or explain. [/INST]"
        else:
            system_prompt = "<s>[INST] <<SYS>> \n You are a conversational query rewriter LLM, an intelligent assistant that can help reformulate the current turn question into rewrite that can fully express the user's information needs based on the given historical information-seeking multi-turn dialog context.  \n"
            user_prompt = "The historical information-seeking multi-turn dialog context is: \"" + context + "\"" + "\nThe current turn question is:  \"" + cur_q + "\"" + "\nThe output format should be in a list with index e.g., 1. 2. 3.. Only respond with the generation results, do not say any word or explain. Note that you should always try to rewrite it. Never ask for clarification or say you don't understand it in the generated rewrite. [/INST]"
    prompt = system_prompt + user_prompt
    return prompt

model_name = "llama"
if model_name == "llama":
    tokenizer = AutoTokenizer.from_pretrained("/media/nvme/fengran/mem_aug/models/llama-2-7b-chat-hf")
    model = LlamaForCausalLM.from_pretrained("/media/nvme/fengran/mem_aug/models/llama-2-7b-chat-hf")
elif model_name == "mistral":
    tokenizer = AutoTokenizer.from_pretrained("/media/nvme/abbas/convqa/models/Mistral-7B-Instruct-v0.2")
    model = MistralForCausalLM.from_pretrained("/media/nvme/abbas/convqa/models/Mistral-7B-Instruct-v0.2")
elif model_name == "gemma":
    tokenizer = AutoTokenizer.from_pretrained("/media/nvme/abbas/convqa/models/gemma-7b-it")
    model = MistralForCausalLM.from_pretrained("/media/nvme/abbas/convqa/models/gemma-7b-it")

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device=5)

'''
ori_file = "/media/nvme/fengran/QReCC/test_with_doc.json"
valid_qid = []
with open(ori_file, 'r') as f:
    data = f.readlines()
    for line in data:
        record = json.loads(line)
        if len(record["pos_docs_pids"]) > 0:
            valid_qid.append(record["sample_id"])
'''
context_file = "/media/nvme/fengran/QReCC/test_llama_context_qr.json"
input_file = "/media/nvme/fengran/QReCC/test_with_rewritten_q_new.json"
output_file = "/media/nvme/fengran/QReCC/llama_test_with_doc_newcontext_search_q.json"
with open(input_file) as f, open(context_file) as f1, open(output_file, 'w') as g:
    # Write a line of text to the file
    cur = 0
    data = f.readlines()[cur:]
    context_data = f1.readlines()[cur:]
    n = len(data)
    for i in tqdm(trange(n)):
        record = json.loads(data[i])
        record_1 = json.loads(context_data[i])
        context = record_1["context"]
        sample_id = record["sample_id"]
        cur_utt_text = record['cur_utt_text']
        ctx_utts_text = record['ctx_utts_text']
        pos_docs = record["pos_docs"][0]
        #if sample_id not in valid_qid:
        #    continue
        #cur_response_text = record["cur_response_text"]
        #oracle_utt_text = record["oracle_utt_text"]
        if len(record["pos_docs_pids"]) == 0:
        #    g.write(json.dumps(record) + "\n")
            continue
        #else:
        #    pos_docs = record["pos_docs"][0]
        use_context = True
        QR = False
        if use_context:
            context = ""
            for j in range(len(ctx_utts_text)):
                context += ctx_utts_text[j] + ' '
            if len(context) - 512 > 0:
                start = len(context) - 512
            else:
                start = 0
            if not QR:
                prompt = get_prompt(pos_docs[:384], cur_utt_text, context[start:], QR)
            else:
                prompt = get_prompt(None, cur_utt_text, context[start:], QR)
        else:
            prompt = get_prompt(pos_docs[0][:384], cur_utt_text, None, QR)
        #prompt = pipe.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        out = pipe([prompt], eos_token_id=pipe.tokenizer.eos_token_id, temperature=0.7,
        do_sample = True,
        repetition_penalty=1.1,
        max_new_tokens=1500)
        patt = "[/INST]"
        idx = out[0][0]['generated_text'].index(patt)
        query_list = out[0][0]['generated_text'][idx+len(patt):].strip().split('\n')
        new_query_list = []
        bad_qid = []
        key_name = None
        if use_context:
            if not QR:
                key_name = model_name + "_context_generated_query"
            else:
                key_name = model_name + "_context_rewritten_query"
        else:
            key_name = "llama-2-7b-chat-hf_generated_query"
        #try:
        #    if "Sure!" in query_list[0]:
        #        begin = 1
        #    else:
        #        begin = 0
        #    for j in range(begin, len(query_list)):
                #patt = '.'
                #idx = query_list[j].index(patt)
                #query = query_list[j][idx + 2:]
                #new_query_list.append(query)
            #record[key_name] = new_query_list
            #g.write(json.dumps(record) + "\n")
        #except:
        new_record = {}
        new_record["sample_id"] = sample_id
        generated_query = query_list
        if not QR and ("Sure" in generated_query[0] or "some potential search queries" in generated_query[0]):
            generated_query = generated_query[1:]
        if "Please let me know" in generated_query[-1]:
            if len(generated_query) > 1:
                generated_query = generated_query[:-1]
            elif len(generated_query) == 1:
                idx = generated_query[0].index("Please let me know")
                generated_query[0] = generated_query[0][:idx]
        while '' in generated_query:
            generated_query.remove('')
        new_generated_query = [key for key in generated_query if len(key.split()) > 1 or (len(key.split()) == 1 and '1' not in key)]
        if len(new_generated_query) == 0:
            new_generated_query = generated_query
        new_record[key_name] = new_generated_query
        g.write(json.dumps(new_record) + "\n")
    #g.writelines(bad_qid)

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="/media/nvme/fengran/output/QR_qrecc/Flant5large-oracle.json")
    parser.add_argument("--test_file_path_2", type=str, default="/media/nvme/fengran/output/QR_qrecc/Flant5large-answer.json")

    args = parser.parse_args()
    generate_search_q(args)
'''
