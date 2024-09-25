import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')
import pandas as pd

import argparse
import torch
from utils import check_dir_exist_or_build, pstore, pload, split_and_padding_neighbor, set_seed, load_collection
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
import json
from tqdm import tqdm, trange
import random
from itertools import combinations
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from models import load_model

def padding_seq_to_same_length(input_ids, max_pad_length, pad_token = 0):
    padding_length = max_pad_length - len(input_ids)
    padding_ids = [pad_token] * padding_length
    attention_mask = []

    if padding_length <= 0:
        attention_mask = [1] * max_pad_length
        input_ids = input_ids[:max_pad_length]
    else:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + padding_ids
            
    assert len(input_ids) == max_pad_length
    assert len(attention_mask) == max_pad_length
  
    return input_ids, attention_mask

class T5FT_context(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        #with open(args.target_file_path, encoding="utf-8") as f:
        #    target_data = f.readlines()

        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        idx = 0
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)
        for line in tqdm(data):
            record = json.loads(line)
            #target_record = json.loads(target_data[idx])
            sample_id, context, target = record['sample_id'], record["context"], ''#record["target"]
            #if args.decode_type == "pseudo_p":
            #    target = target_record["mistral_context_generated_query"][0]
            #if len(target) == 0:
            #    continue
            utt = tokenizer.encode(context, add_special_tokens = True, max_length = args.max_concat_length)
            utt, utt_mask = padding_seq_to_same_length(utt, max_pad_length = args.max_concat_length)
        
            target_seq = target
            target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_query_length, truncation=True)    

            if args.collate_fn_type == "flat_concat_for_train":
                labels = target_encoding.input_ids
                labels = torch.tensor(labels)
                labels[labels == tokenizer.pad_token_id] = -100
                labels = labels.tolist()
            else:
                labels = []
            self.examples.append([sample_id, utt, utt_mask, labels])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_labels": [],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_input_ids"].append(example[1])
                collated_dict["bt_attention_mask"].append(example[2])
                collated_dict["bt_labels"].append(example[3])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text"])
            if args.collate_fn_type == "flat_concat_for_test":
                not_need_to_tensor_keys.add("bt_labels")

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn

class T5RewriterIRDataset_qrecc(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        with open(args.target_file_path, encoding="utf-8") as f:
            target_data = f.readlines()

        #with open(args.train_file_path_2, encoding="utf-8") as f2:
        #    data_2 = f2.readlines()
        #    data.extend(data_2)
        
        n = len(target_data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)
        seen_qid = []
        idx = 0
        for line in tqdm(data):
            record = json.loads(line)
            target_record = json.loads(target_data[idx])
            flat_concat = []
            sample_id = record['sample_id']
            #if sample_id in seen_qid:
            #    continue
            #else:
            #    seen_qid.append(sample_id)
            cur_utt_text = record['cur_utt_text']
            ctx_utts_text = record['ctx_utts_text']
            pos_docs_pids = record["pos_docs_pids"]
            if len(pos_docs_pids) == 0:
                continue
            #cur_response_text = record["cur_response_text"]
            oracle_utt_text = record["oracle_utt_text"]
            if "llama-2-7b-chat-hf_context_rewritten_query" in record:
                search_q_list = record["llama-2-7b-chat-hf_context_rewritten_query"]
                label_text = search_q_list[0]
            elif "llama-2-7b-chat-hf_context_rewritten_query" in target_record:
                search_q_list = target_record["llama-2-7b-chat-hf_context_rewritten_query"]
                label_text = search_q_list[0]
            #oracle = tokenizer.encode(oracle_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            #oracle, oracle_mask = padding_seq_to_same_length(oracle, max_pad_length = args.max_query_length)
            #if "pos_docs" in record and len(record["pos_docs"]) > 0:
            #    pos_docs_text = record["pos_docs"]
            #else:
            #    continue
            
            if args.use_prefix:
                #cur_utt_text = "Please generate rewritten query for the given historical information-seeking multi-turn dialog context to satisfy current query information needs. \n The current query is: " + cur_utt_text + "\n"
                cur_utt_text = "[Question]: " + cur_utt_text
                first_context = True
            idx += 1
                
            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            flat_concat.extend(cur_utt)
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                if args.use_prefix and first_context:
                    #ctx_utts_text[j] = "The historical information-seeking dialog context is: " + ctx_utts_text[j]
                    ctx_utts_text[j] = "[Context]: " + ctx_utts_text[j]
                    first_context = False
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length, truncation=True) # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt) 

            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length = args.max_concat_length)

            if args.collate_fn_type == "flat_concat_for_train":
                #for idx in range(min(len(search_q_list), args.query_num)):
                if args.decode_type == "oracle":
                    target_seq = oracle_utt_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_query_length, truncation=True)    
                elif args.decode_type == "search_q":
                    target_seq = label_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_query_length, truncation=True)    

                labels = target_encoding.input_ids
                labels = torch.tensor(labels)
                labels[labels == tokenizer.pad_token_id] = -100
                labels = labels.tolist()
                
                #for idx in range(len(pos_docs)):
                pos_docs, neg_docs, pos_docs_mask, neg_docs_mask = [], [], [], []
                #pos_docs.extend(tokenizer.encode(pos_docs_text[0], add_special_tokens=True, max_length = args.max_doc_length))
                #neg_docs.extend(tokenizer.encode(random_neg_docs_text[0], add_special_tokens=True, max_length = args.max_doc_length))
                #pos_docs, pos_docs_mask = padding_seq_to_same_length(pos_docs, max_pad_length = args.max_doc_length)
                #neg_docs, neg_docs_mask = padding_seq_to_same_length(neg_docs, max_pad_length = args.max_doc_length)
                #breakpoint()
                self.examples.append([sample_id, 
                                flat_concat,
                                flat_concat_mask,
                                labels,
                                pos_docs,
                                pos_docs_mask,
                                neg_docs,
                                neg_docs_mask])
            else:
                labels, oracle, oracle_mask = [], [], []
                pos_docs, neg_docs, pos_docs_mask, neg_docs_mask = [], [], [], []
                self.examples.append([sample_id, 
                                        flat_concat,
                                        flat_concat_mask,
                                        labels,
                                        pos_docs,
                                        pos_docs_mask,
                                        neg_docs,
                                        neg_docs_mask])

    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_labels": [],
                             "bt_pos_docs": [],
                             "bt_pos_docs_mask": [],
                             "bt_neg_docs": [],
                             "bt_neg_docs_mask": [],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_input_ids"].append(example[1])
                collated_dict["bt_attention_mask"].append(example[2])
                collated_dict["bt_labels"].append(example[3])
                collated_dict["bt_pos_docs"].append(example[4])
                collated_dict["bt_pos_docs_mask"].append(example[5])
                collated_dict["bt_neg_docs"].append(example[6])
                collated_dict["bt_neg_docs_mask"].append(example[7])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text"])
            if args.collate_fn_type == "flat_concat_for_test":
                not_need_to_tensor_keys.add("bt_labels")

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn



class T5RewriterIRDataset_topiocqa(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        with open(args.target_file_path, encoding="utf-8") as f:
            target_data = f.readlines()

        n = len(target_data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        i = 0

        for line in tqdm(data):
            record = json.loads(line)
            target_record = json.loads(target_data[i])
            flat_concat = []
            ctx_utts_text = record['cur_utt_text'].strip().split(" [SEP] ") # [q1, a1, q2, a2, ...]
            cur_utt_text = ctx_utts_text[-1] 
            ctx_utts_text = ctx_utts_text[:-1]
            if "llama-2-7b-chat-hf_context_rewritten_query" in record:
                search_q_list = record["llama-2-7b-chat-hf_context_rewritten_query"]
                label_text = search_q_list[0]
            elif "llama-2-7b-chat-hf_context_rewritten_query" in target_record:
                search_q_list = target_record["llama-2-7b-chat-hf_context_rewritten_query"]
                label_text = search_q_list[0]
            i += 1

            #if "ResponseAndReasoning_Query" in record:
                #oracle_utt_text = record["ResponseAndReasoning_Query"]
            #    pos_doc = record["ResponseAndReasoning_Response"]
            #elif "ResponseThenReasoning_Query" in record:
                #oracle_utt_text = record["ResponseThenReasoning_Query"]
            #    pos_doc = record["ResponseThenReasoning_Response"]
            if args.use_prefix:
                #cur_utt_text = "The current turn query is: " + cur_utt_text
                cur_utt_text = "[Current question]: " + cur_utt_text
                first_context = True
                
            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            flat_concat.extend(cur_utt)
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                if args.use_prefix and first_context:
                    #ctx_utts_text[j] = "The historical information-seeking dialog context is:" + ctx_utts_text[j]
                    ctx_utts_text[j] = "[Context]: " + ctx_utts_text[j]
                    first_context = False
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length, truncation=True) # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt) 

            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length = args.max_concat_length)

            if args.collate_fn_type == "flat_concat_for_train":
                #for idx in range(min(len(oracle_utt_text_list), args.query_num)):
                    #label_text = oracle_utt_text_list[idx]
                if args.decode_type == "oracle":
                    target_seq = oracle_utt_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_concat_length, truncation=True)    
                elif args.decode_type == "search_q":
                    target_seq = label_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_response_length, truncation=True)    

                labels = target_encoding.input_ids
                labels = torch.tensor(labels)
                labels[labels == tokenizer.pad_token_id] = -100
                labels = labels.tolist()
                
                pos_docs, neg_docs, pos_docs_mask, neg_docs_mask = [], [], [], []
                #pos_docs.extend(tokenizer.encode(pos_doc, add_special_tokens=True, max_length = args.max_doc_length))
                #neg_docs.extend(tokenizer.encode(record["neg_docs"], add_special_tokens=True, max_length = args.max_doc_length))
                #pos_docs, pos_docs_mask = padding_seq_to_same_length(pos_docs, max_pad_length = args.max_doc_length)
                #neg_docs, neg_docs_mask = padding_seq_to_same_length(neg_docs, max_pad_length = args.max_doc_length)
            
                self.examples.append([record['sample_id'], 
                                flat_concat,
                                flat_concat_mask,
                                labels,
                                pos_docs,
                                pos_docs_mask,
                                neg_docs,
                                neg_docs_mask])
            else:
                labels = []
                pos_docs, neg_docs, pos_docs_mask, neg_docs_mask = [], [], [], []
                self.examples.append([record['sample_id'], 
                                        flat_concat,
                                        flat_concat_mask,
                                        labels,
                                        pos_docs,
                                        pos_docs_mask,
                                        neg_docs,
                                        neg_docs_mask])

    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_labels": [],
                             "bt_pos_docs": [],
                             "bt_pos_docs_mask": [],
                             "bt_neg_docs": [],
                             "bt_neg_docs_mask": [],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_input_ids"].append(example[1])
                collated_dict["bt_attention_mask"].append(example[2])
                collated_dict["bt_labels"].append(example[3])
                collated_dict["bt_pos_docs"].append(example[4])
                collated_dict["bt_pos_docs_mask"].append(example[5])
                collated_dict["bt_neg_docs"].append(example[6])
                collated_dict["bt_neg_docs_mask"].append(example[7])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text"])
            if args.collate_fn_type == "flat_concat_for_test":
                not_need_to_tensor_keys.add("bt_labels")

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn

class Retrieval_qrecc(Dataset):
    def __init__(self, args, tokenizer, filename, collection=None):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        
        #with open(args.test_file_path_1, encoding="utf-8") as f:
        #    data_1 = f.readlines()

        #with open(args.test_file_path_2, encoding="utf-8") as f:
        #    data_2 = f.readlines()

        n = len(data)
        idx = 0

        for i in tqdm(trange(n)):
            record = json.loads(data[i])
            # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
            sample_id = record['sample_id']
            flat_qa_concat = []
            #flat_qp_concat = []
            ctx_utts_text = record["ctx_utts_text"] # [q1, a1, q2, a2, ...]
            cur_utt_text = record["cur_utt_text"] 
            cur_response_text = record["cur_response_text"]
            if 'pos_docs' in record:
                pos_docs_text = record['pos_docs']
            pos_docs_pids = record['pos_docs_pids']
            #if len(pos_docs_pids) == 0:
            #    continue
            #if args.is_train:
                #bm25_hard_neg_docs = record['bm25_hard_neg_docs'][0]
                #pseudo_prepos_docs = record["pseudo_prepos_docs"]
                #prepos_neg_docs = record["prepos_neg_docs"]
            #rel_label = record['rel_label']

            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            flat_qa_concat.extend(cur_utt)
            '''
            oracle_record = json.loads(data_1[idx])
            answer_record = json.loads(data_2[idx])
            idx += 1
            rewritten_query_text = oracle_record["oracle_utt_text"] 
            generated_response_text = answer_record["answer_utt_text"]
            rewritten_query = tokenizer.encode(rewritten_query_text, add_special_tokens = True, max_length = args.max_query_length)
            generated_response = tokenizer.encode(generated_response_text, add_special_tokens = True, max_length = args.max_response_length)
            flat_qa_concat.extend(generated_response)
            flat_qa_concat.extend(rewritten_query)
            '''

            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length, truncation=True) # not remove [CLS]
                if len(flat_qa_concat) + len(utt) > args.max_concat_length:
                    flat_qa_concat += utt[:args.max_concat_length - len(flat_qa_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_qa_concat.extend(utt)

            flat_qa_concat, flat_qa_concat_mask = padding_seq_to_same_length(flat_qa_concat, max_pad_length = args.max_concat_length)

            # doc 
            pos_docs, neg_docs, pos_docs_mask, neg_docs_mask = [], [], [], []
            if args.is_train:
                #pos_docs.extend(tokenizer.encode(pos_docs_text[0], add_special_tokens=True, max_length=args.max_doc_length, truncation=True))
                #neg_docs.extend(tokenizer.encode(bm25_hard_neg_docs, add_special_tokens=True, max_length=args.max_doc_length, truncation=True))
                #pos_docs, pos_docs_mask = padding_seq_to_same_length(pos_docs, max_pad_length = args.max_doc_length)
                #neg_docs, neg_docs_mask = padding_seq_to_same_length(neg_docs, max_pad_length = args.max_doc_length)
                self.examples.append([sample_id, flat_qa_concat, flat_qa_concat_mask, pos_docs, pos_docs_mask, neg_docs, neg_docs_mask])
            else:
                self.examples.append([sample_id, flat_qa_concat, flat_qa_concat_mask, pos_docs, pos_docs_mask, neg_docs, neg_docs_mask])
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_conv_qa": [],
                             "bt_conv_qa_mask": [],
                             "bt_pos_docs":[],
                             "bt_pos_docs_mask":[],
                             "bt_neg_docs":[],
                             "bt_neg_docs_mask":[],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_conv_qa"].append(example[1])
                collated_dict["bt_conv_qa_mask"].append(example[2])
                collated_dict["bt_pos_docs"].append(example[3])
                collated_dict["bt_pos_docs_mask"].append(example[4])
                collated_dict["bt_neg_docs"].append(example[5])
                collated_dict["bt_neg_docs_mask"].append(example[6])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn

class ConvGQR_Retrieval(Dataset):
    def __init__(self, args, tokenizer, filename_oracle, filename_answer):
        self.examples = []
        
        with open(filename_oracle, encoding="utf-8") as f:
            data_1 = f.readlines()

        with open(filename_answer, encoding="utf-8") as f:
            data_2 = f.readlines()

        n = len(data_1)

        for i in tqdm(trange(n)):
            oracle_record = json.loads(data_1[i])
            answer_record = json.loads(data_2[i])
            # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
            sample_id = oracle_record['sample_id']
            if args.from_LLM:
                #if "ResponseAndReasoning" in oracle_record:
                #    rewritten_query_text = oracle_record["ResponseAndReasoning_Query"] 
                #    generated_response_text = answer_record["ResponseAndReasoning_Response"]
                #elif "ResponseThenReasoning" in oracle_record:
                rewritten_query_text = oracle_record["oracle_utt_text"] 
                generated_response_text = answer_record["ResponseThenReasoning_Response"]
            else:
                rewritten_query_text = oracle_record["oracle_utt_text"] 
                generated_response_text = answer_record["answer_utt_text"]
            combine_text = rewritten_query_text + ' ' + generated_response_text

            rewritten_query = tokenizer.encode(rewritten_query_text, add_special_tokens = True, max_length = args.max_query_length)
            generated_response = tokenizer.encode(generated_response_text, add_special_tokens = True, max_length = args.max_response_length)
            combine = tokenizer.encode(combine_text, add_special_tokens = True, max_length = args.max_concat_length)

            rewritten_query, rewritten_query_mask = padding_seq_to_same_length(rewritten_query, max_pad_length = args.max_query_length)
            generated_response, generated_response_mask = padding_seq_to_same_length(generated_response, max_pad_length = args.max_response_length)
            combine, combine_mask = padding_seq_to_same_length(combine, max_pad_length = args.max_concat_length)
            self.examples.append([sample_id, rewritten_query, rewritten_query_mask, generated_response, generated_response_mask, combine, combine_mask])
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_rewritten_query": [],
                             "bt_generated_response": [],
                             "bt_rewritten_query_mask": [],
                             "bt_generated_response_mask": [],
                             "bt_combine": [],
                             "bt_combine_mask": [],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_rewritten_query"].append(example[1])
                collated_dict["bt_rewritten_query_mask"].append(example[2])
                collated_dict["bt_generated_response"].append(example[3])
                collated_dict["bt_generated_response_mask"].append(example[4])
                collated_dict["bt_combine"].append(example[5])
                collated_dict["bt_combine_mask"].append(example[6])


            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn
        
class Search_q_Retrieval(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        if args.expansion:
            with open(args.exp_filename, encoding="utf-8") as f:
                exp_data = f.readlines()

        n = len(data)

        for i in tqdm(trange(n)):
            record = json.loads(data[i])
            sample_id = record["sample_id"]
            # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
            '''
            elif "llama-2-7b-chat-hf_incontext_generated_query" in record:
                generated_query_text = record["llama-2-7b-chat-hf_incontext_generated_query"][0]
            
            elif "llama-2-7b-chat-hf_reorder_context_generated_query" in record:
                generated_query_text = record["llama-2-7b-chat-hf_reorder_context_generated_query"][0]
            '''
            #sample_id = record['sample_id']
            if "llama-2-7b-chat-hf_context_generated_query" in record:
                generated_query_text = record["llama-2-7b-chat-hf_context_generated_query"][0]
            elif "llama_context_generated_query" in record:
                generated_query_text = record["llama_context_generated_query"][0]
            elif "query" in record:
                generated_query_text = record["query"]
            elif "text" in record:
                generated_query_text = record["text"]
            elif "mistral_context_generated_query" in record:
                generated_query_text = record["mistral_context_generated_query"][0]
            elif "Mistral_context_rewritten_query" in record:
                generated_query_text = record["Mistral_context_rewritten_query"][0]
            #elif "llama-2-7b-chat-hf_context_generated_query" in record:
            #    generated_query_text = record["llama-2-7b-chat-hf_context_generated_query"][0]
            elif "llama-2-7b-chat-hf_reorder_context_rewritten_query" in record:
                generated_query_text = record["llama-2-7b-chat-hf_reorder_context_rewritten_query"][0]
            elif "oracle_utt_text" in record:
                if args.expansion:
                    exp_record = json.loads(exp_data[i])
                    generated_query_text = record["oracle_utt_text"] + ' ' +  exp_record["text"]
                else:
                    generated_query_text = record["oracle_utt_text"]
            elif "Editor_rewrite" in record:
                sample_id = "QReCC-Test_" + str(record["Conversation_no"]) + '_' + str(record["Turn_no"])
                if len(record['Editor_rewrite']) == 6:
                    generated_query_text = record['Editor_rewrite']['choices'][0]['message']['content']
                else:
                    generated_query_text = record['Editor_rewrite']
            #elif "llama-2-7b-chat-hf_context_rewritten_query" in record:
            #    generated_query_text = record["llama-2-7b-chat-hf_context_rewritten_query"][0]
            elif "ResponseAndReasoning_Query" in record:
                if args.expansion:
                    generated_query_text = record["ResponseAndReasoning_Query"] + ' ' + record["ResponseAndReasoning_Response"]
                else:
                    generated_query_text = record["ResponseAndReasoning_Query"]
            elif "ResponseThenReasoning_Query" in record:
                if args.expansion:
                    generated_query_text = record["ResponseThenReasoning_Query"] + ' ' + record["ResponseThenReasoning_Response"]
                else:
                    generated_query_text = record["ResponseThenReasoning_Query"]
            elif "flant5-large_context_rewritten_query" in record:
                generated_query_text = record["flant5-large_context_rewritten_query"]
            else:
                raise ValueError
            generated_query = tokenizer.encode(generated_query_text, add_special_tokens = True, max_length = args.max_concat_length)
            
            generated_query, generated_query_mask = padding_seq_to_same_length(generated_query, max_pad_length = args.max_concat_length)

            self.examples.append([sample_id, generated_query, generated_query_mask])
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_rewrite": [],
                             "bt_rewrite_mask": [],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_rewrite"].append(example[1])
                collated_dict["bt_rewrite_mask"].append(example[2])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn

class keywords_Retrieval(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        with open(args.QR_filename, encoding="utf-8") as f:
            QR_data = f.readlines()

        n = len(data)

        for i in tqdm(trange(n)):
            record = json.loads(data[i])
            qr = json.loads(QR_data[i])
            sample_id = record["sample_id"]
            #if args.dataset == "topiocqa":
            #    ctx_utts_text = record['cur_utt_text'].strip().split(" [SEP] ") # [q1, a1, q2, a2, ...]
            #    cur_utt_text = ctx_utts_text[-1] 
            #    ctx_utts_text = ctx_utts_text[:-1]
            #elif args.dataset == "qrecc":
            #    cur_utt_text = record["cur_utt_text"] 
            #    ctx_utts_text = record["ctx_utts_text"]
            #queries = []
            #for j in range(len(ctx_utts_text)):
            #    if j % 2 == 0: 
            #        queries.append(ctx_utts_text[j])
            #queries.append(cur_utt_text)
            #generated_response = []
            generated_query_text = ""
            if len(record["text"]) == 0:
                generated_query_text = qr["text"]
            else:
                generated_query_text = record["text"]
            #if "generated_response" in record:
                #turn_id = int(sample_id.split('_')[-1])
                #for idx in range(turn_id - 1):
                #    his_example = json.loads(data[i - idx - 1])
                #    generated_response.append(his_example["generated_response"])
                #generated_response.append(record["generated_response"])
                #generated_query_text += qr["oracle_utt_text"] + ' ' + record["generated_response"]
            #generated_query_text += qr["oracle_utt_text"] + ' ' + record["generated_response"]
            #try:
            #    assert len(queries) == len(generated_response)
            #except:
            #    breakpoint()
            #for j in range(len(queries)):
            #    generated_query_text += queries[j] + ' ' + generated_response[j] + ' '
            #if args.keywords:
            #    generated_query_text += ' ' + record["keywords"]

            generated_query = tokenizer.encode(generated_query_text, add_special_tokens = True, max_length = args.max_concat_length)
            generated_query, generated_query_mask = padding_seq_to_same_length(generated_query, max_pad_length = args.max_concat_length)
            self.examples.append([sample_id, generated_query, generated_query_mask])
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_rewrite": [],
                             "bt_rewrite_mask": [],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_rewrite"].append(example[1])
                collated_dict["bt_rewrite_mask"].append(example[2])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn



class Retrieval_topiocqa(Dataset):
    def __init__(self, args, tokenizer, filename, collection=None):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        n = len(data)

        for i in tqdm(trange(n)):
            record = json.loads(data[i])
            # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
            sample_id = record['sample_id']
            flat_qp_concat = []
            ctx_utts_text = record['cur_utt_text'].strip().split(" [SEP] ") # [q1, a1, q2, a2, ...]
            cur_utt_text = ctx_utts_text[-1] 
            ctx_utts_text = ctx_utts_text[:-1]
            #last_response = record['last_response']
            #pos_docs_text = record['pos_docs'][0]
            #pos_docs_pids = record['pos_docs_pids'][0]

            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            flat_qp_concat.extend(cur_utt)
            
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1: # answer
                    max_length = args.max_response_length
                elif j % 2 == 0: # query
                    max_length = args.max_query_length
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length, truncation=True) # not remove [CLS]
                if len(flat_qp_concat) + len(utt) > args.max_concat_length:
                    flat_qp_concat += utt[:args.max_concat_length - len(flat_qp_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_qp_concat.extend(utt)
                    
            flat_qp_concat, flat_qp_concat_mask = padding_seq_to_same_length(flat_qp_concat, max_pad_length = args.max_concat_length)

            # doc 
            pos_docs, neg_docs, pos_docs_mask, neg_docs_mask, pseudo_prepos_docs, pseudo_prepos_docs_mask, prepos_neg_docs, prepos_neg_docs_mask = [], [], [], [], [], [], [], []
            '''
            if args.is_train:
                pos_docs.extend(tokenizer.encode(pos_docs_text, add_special_tokens=True, max_length=args.max_doc_length, truncation=True))
                neg_docs.extend(tokenizer.encode(bm25_hard_neg_docs, add_special_tokens=True, max_length=args.max_doc_length, truncation=True))
                pos_docs, pos_docs_mask = padding_seq_to_same_length(pos_docs, max_pad_length = args.max_doc_length)
                neg_docs, neg_docs_mask = padding_seq_to_same_length(neg_docs, max_pad_length = args.max_doc_length)
                if len(pseudo_prepos_docs) > 0:
                    pseudo_prepos_docs.extend(tokenizer.encode(random.choice(pseudo_prepos_docs), add_special_tokens=True, max_length=args.max_doc_length, truncation=True))
                    pseudo_prepos_docs, pseudo_prepos_docs_mask = padding_seq_to_same_length(pseudo_prepos_docs, max_pad_length = args.max_doc_length)
                if len(prepos_neg_docs) > 0:
                    prepos_neg_docs.extend(tokenizer.encode(random.choice(prepos_neg_docs), add_special_tokens=True, max_length=args.max_doc_length, truncation=True))
                    prepos_neg_docs, prepos_neg_docs_mask = padding_seq_to_same_length(prepos_neg_docss, max_pad_length = args.max_doc_length)
            '''
            self.examples.append([sample_id, flat_qp_concat, flat_qp_concat_mask])
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_conv_qp": [],
                             "bt_conv_qp_mask": [],
                             "bt_pos_docs":[],
                             "bt_pos_docs_mask":[],
                             "bt_neg_docs":[],
                             "bt_neg_docs_mask":[],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_conv_qp"].append(example[1])
                collated_dict["bt_conv_qp_mask"].append(example[2])
                #collated_dict["bt_pos_docs"].append(example[3])
                #collated_dict["bt_pos_docs_mask"].append(example[4])
                #collated_dict["bt_neg_docs"].append(example[5])
                #collated_dict["bt_neg_docs_mask"].append(example[6])
                

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn

class QR_qrecc(Dataset):
    def __init__(self, args, tokenizer, filename, collection=None):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        n = len(data)

        for i in tqdm(trange(n)):
            record = json.loads(data[i])
            # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
            sample_id = record['sample_id']
            flat_concat = []
            ctx_utts_text = record["ctx_utts_text"] # [q1, a1, q2, a2, ...]
            cur_utt_text = record["cur_utt_text"] 
            cur_response_text = record["cur_response_text"]
            oracle_utt_text = record["oracle_utt_text"]

            pos_docs_pids = record['pos_docs_pids']
            if len(pos_docs_pids) == 0:
                continue

            if args.use_prefix:
                #cur_utt_text = "Please generate rewritten query for the given historical information-seeking multi-turn dialog context to satisfy current query information needs. \n The current query is: " + cur_utt_text + "\n"
                cur_utt_text = "[Current question]: " + cur_utt_text
                first_context = True
                
            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            flat_concat.extend(cur_utt)
            for j in range(len(ctx_utts_text) - 1, -1, -1):
            #for j in range(len(ctx_utts_text)):
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                if args.use_prefix and first_context:
                    #ctx_utts_text[j] = "The historical information-seeking dialog context is:" + ctx_utts_text[j]
                    ctx_utts_text[j] = "[Context]: " + ctx_utts_text[j]
                    first_context = False
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length, truncation=True) # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt) 

            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length = args.max_concat_length)
            self.examples.append([sample_id, flat_concat, flat_concat_mask, cur_utt_text, ctx_utts_text, cur_response_text, oracle_utt_text])
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_conv_qa": [],
                             "bt_conv_qa_mask": [],
                             "bt_cur_utt_text":[],
                             "bt_ctx_utts_text":[],
                             "bt_cur_response_text":[],
                             "bt_oracle_utt_text":[],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_conv_qa"].append(example[1])
                collated_dict["bt_conv_qa_mask"].append(example[2])
                collated_dict["bt_cur_utt_text"].append(example[3])
                collated_dict["bt_ctx_utts_text"].append(example[4])
                collated_dict["bt_cur_response_text"].append(example[5])
                collated_dict["bt_oracle_utt_text"].append(example[6])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_ctx_utts_text", "bt_cur_response_text", "bt_oracle_utt_text"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn

class QR_topiocqa(Dataset):
    def __init__(self, args, tokenizer, filename, collection=None):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        n = len(data)

        for i in tqdm(trange(n)):
            record = json.loads(data[i])
            # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
            sample_id = record['sample_id']
            flat_concat = []
            ctx_utts_text = record['cur_utt_text'].strip().split(" [SEP] ") # [q1, a1, q2, a2, ...]
            cur_utt_text = ctx_utts_text[-1] 
            ctx_utts_text = ctx_utts_text[:-1]
            cur_response_text = record["cur_response_text"]

            if args.use_prefix:
                #cur_utt_text = "The current turn query is: " + cur_utt_text
                cur_utt_text = "[Current question]: " + cur_utt_text
                first_context = True
                
            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            flat_concat.extend(cur_utt)
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                if args.use_prefix and first_context:
                    #ctx_utts_text[j] = "The historical information-seeking dialog context is:" + ctx_utts_text[j]
                    ctx_utts_text[j] = "[Context]: " + ctx_utts_text[j]
                    first_context = False
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length, truncation=True) # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt) 

            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length = args.max_concat_length)
            self.examples.append([sample_id, flat_concat, flat_concat_mask, cur_utt_text, ctx_utts_text, cur_response_text])
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_conv_qa": [],
                             "bt_conv_qa_mask": [],
                             "bt_cur_utt_text":[],
                             "bt_ctx_utts_text":[],
                             "bt_cur_response_text":[],
                             "bt_oracle_utt_text":[],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_conv_qa"].append(example[1])
                collated_dict["bt_conv_qa_mask"].append(example[2])
                collated_dict["bt_cur_utt_text"].append(example[3])
                collated_dict["bt_ctx_utts_text"].append(example[4])
                collated_dict["bt_cur_response_text"].append(example[5])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_ctx_utts_text", "bt_cur_response_text", "bt_oracle_utt_text"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn

class QR_cast(Dataset):
    def __init__(self, args, tokenizer, filename, collection=None):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        n = len(data)

        for i in tqdm(trange(n)):
            record = json.loads(data[i])
            # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
            sample_id = record['sample_id']
            flat_concat = []
            cur_utt_text = record["cur_utt_text"]
            ctx_utts_text = record["ctx_utts_text"]

            if args.use_prefix:
                #cur_utt_text = "The current turn query is: " + cur_utt_text
                cur_utt_text = "[Current question]: " + cur_utt_text
                first_context = True
                
            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            flat_concat.extend(cur_utt)
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                max_length = 64
                if args.use_prefix and first_context:
                    #ctx_utts_text[j] = "The historical information-seeking dialog context is:" + ctx_utts_text[j]
                    ctx_utts_text[j] = "[Context]: " + ctx_utts_text[j]
                    first_context = False
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length, truncation=True) # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt) 

            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length = args.max_concat_length)
            self.examples.append([sample_id, flat_concat, flat_concat_mask, cur_utt_text, ctx_utts_text])
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_conv_qa": [],
                             "bt_conv_qa_mask": [],
                             "bt_cur_utt_text":[],
                             "bt_ctx_utts_text":[],
                             "bt_oracle_utt_text":[],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_conv_qa"].append(example[1])
                collated_dict["bt_conv_qa_mask"].append(example[2])
                collated_dict["bt_cur_utt_text"].append(example[3])
                collated_dict["bt_ctx_utts_text"].append(example[4])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_ctx_utts_text", "bt_cur_response_text", "bt_oracle_utt_text"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn

class llama_prompt_topiocqa(Dataset):
    def __init__(self, args, tokenizer, filename, collection=None):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        n = len(data)

        for i in tqdm(trange(n)):
            record = json.loads(data[i])
            sample_id = record["sample_id"]
            flat_concat = []
            cur_utt_text = record['cur_utt_text']
            ctx_utts_text = record['ctx_utts_text']
            cur_response_text = record["cur_response_text"]
            oracle_utt_text = record["oracle_utt_text"]
            
            #if "pos_docs" in record and len(record["pos_docs"]) > 0:
            #    pos_docs_text = record["pos_docs"]
            if len(record["pos_docs_pids"]) == 0:
                continue
            else:
                pos_docs = record["pos_docs"]

            prompt_query_rewrite = "You are a query rewriter LLM, an intelligent assistant that rewrite the current query by given a information seeking dialog context into a de-contextualized one. The current query is \"" + cur_utt_text + "\", and the dialogue context is \"" + ' '.join(ctx_utts_text) + "\""

            prompt_query_generation = "You are a query generator LLM, an intelligent assistant that can generate a search query for the given relevant passage based on their relevancy. The given relevant passage is \"" + pos_docs[0] + "\""

            prompt_response_generation = "You are a conversational question answering LLM, an intelligent assistant that can answer the current query by given a information seeking dialog context. The current query is \"" + cur_utt_text + ", and the dialogue context is \"" + ' '.join(ctx_utts_text) + "\""

            if args.task_type == "response":
                prompt_text = prompt_response_generation
            elif args.task_type == "rewrite":
                prompt_text = prompt_query_rewrite
            elif args.task_type == "generation":
                prompt_text = prompt_query_generation

            prompt = tokenizer.encode(prompt_text, add_special_tokens = True, max_length = args.max_concat_length)
            prompt, prompt_mask = padding_seq_to_same_length(prompt, max_pad_length = args.max_concat_length)
            self.examples.append([sample_id, prompt, prompt_mask])
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "prompt": [],
                             "prompt_mask": [],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["prompt"].append(example[1])
                collated_dict["prompt_mask"].append(example[2])
                

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_ctx_utts_text", "bt_cur_response_text", "bt_oracle_utt_text"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn
