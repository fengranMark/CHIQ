import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')

import time
import copy
import pickle
import random
import numpy as np
import csv
import argparse
import os

from os import path
from os.path import join as oj
import json
from tqdm import tqdm, trange

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from tensorboardX import SummaryWriter

from models import load_model
from utils import check_dir_exist_or_build, pstore, pload, set_seed, get_optimizer, print_res
from data_format import T5RewriterIRDataset_qrecc, T5RewriterIRDataset_topiocqa, T5FT_context
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

def save_model(args, model, query_tokenizer, save_model_order, epoch, step, loss):
    output_dir = oj(args.model_output_path, '{}-{}-best-model'.format("flant5large-search-nop", args.decode_type))
    check_dir_exist_or_build([output_dir])
    model_to_save = model.module if hasattr(model, 'module') else model
    #model_to_save.t5.save_pretrained(output_dir)
    model_to_save.save_pretrained(output_dir)
    query_tokenizer.save_pretrained(output_dir)
    logger.info("Step {}, Save checkpoint at {}".format(step, output_dir))

def cal_ranking_loss(query_embs, pos_doc_embs, neg_doc_embs):
    batch_size = len(query_embs)
    pos_scores = query_embs.mm(pos_doc_embs.T)  # B * B
    neg_scores = torch.sum(query_embs * neg_doc_embs, dim = 1).unsqueeze(1) # B * 1 hard negatives
    score_mat = torch.cat([pos_scores, neg_scores], dim = 1)    # B * (B + 1)  in_batch negatives + 1 BM25 hard negative 
    label_mat = torch.arange(batch_size).to(args.device) # B
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(score_mat, label_mat)
    return loss

def cal_kd_loss(query_embs, kd_embs):
    loss_func = nn.MSELoss()
    return loss_func(query_embs, kd_embs)


def train(args, log_writer):
    #passage_tokenizer, passage_encoder = load_model("ANCE_Passage", args.pretrained_passage_encoder)
    #passage_encoder = passage_encoder.to(args.device)
   
    query_tokenizer = T5Tokenizer.from_pretrained(args.pretrained_query_encoder)
    query_encoder = T5ForConditionalGeneration.from_pretrained(args.pretrained_query_encoder).to(args.device)

    if args.n_gpu > 1:
        query_encoder = torch.nn.DataParallel(query_encoder, device_ids = list(range(args.n_gpu)))

    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    
    # data prepare
    if args.dataset == "topiocqa":
        train_dataset = T5RewriterIRDataset_topiocqa(args, query_tokenizer, args.train_file_path)
        #train_dataset = T5FT_context(args, query_tokenizer, args.train_file_path)
    elif args.dataset == "qrecc":
        #train_dataset = T5FT_context(args, query_tokenizer, args.train_file_path)
        train_dataset = T5RewriterIRDataset_qrecc(args, query_tokenizer, args.train_file_path)
    train_loader = DataLoader(train_dataset, 
                                #sampler=train_sampler,
                                batch_size = args.batch_size, 
                                shuffle=True, 
                                collate_fn=train_dataset.get_collate_fn(args))

    logger.info("train samples num = {}".format(len(train_dataset)))
    
    total_training_steps = args.num_train_epochs * (len(train_dataset) // args.batch_size + int(bool(len(train_dataset) % args.batch_size)))
    num_warmup_steps = args.num_warmup_portion * total_training_steps
    
    optimizer = get_optimizer(args, query_encoder, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_training_steps)

    global_step = 0
    save_model_order = 0

    # begin to train
    logger.info("Start training...")
    logger.info("Total training epochs = {}".format(args.num_train_epochs))
    logger.info("Total training steps = {}".format(total_training_steps))
    
    num_steps_per_epoch = total_training_steps // args.num_train_epochs
    logger.info("Num steps per epoch = {}".format(num_steps_per_epoch))

    if isinstance(args.print_steps, float):
        args.print_steps = int(args.print_steps * num_steps_per_epoch)
        args.print_steps = max(1, args.print_steps)

    epoch_iterator = trange(args.num_train_epochs, desc="Epoch", disable=args.disable_tqdm)

    best_loss = 1000
    for epoch in epoch_iterator:
        query_encoder.train()
        #passage_encoder.eval()
        for batch in tqdm(train_loader,  desc="Step", disable=args.disable_tqdm):
            query_encoder.zero_grad()

            bt_conv_query = batch['bt_input_ids'].to(args.device) # B * len
            bt_conv_query_mask = batch['bt_attention_mask'].to(args.device)
            #bt_pos_docs = batch['bt_pos_docs'].to(args.device) # B * len one pos
            #bt_pos_docs_mask = batch['bt_pos_docs_mask'].to(args.device)
            #bt_neg_docs = batch['bt_neg_docs'].to(args.device) # B * len batch size negs
            #bt_neg_docs_mask = batch['bt_neg_docs_mask'].to(args.device)
            bt_oracle_query = batch['bt_labels'].to(args.device)
            #bt_oracle = batch['bt_oracle'].to(args.device)
            #bt_oracle_mask = batch['bt_oracle_mask'].to(args.device)
            
            output = query_encoder(input_ids=bt_conv_query, attention_mask=bt_conv_query_mask, labels=bt_oracle_query)
            decode_loss = output.loss  # B * dim
            #conv_query_embs = output.encoder_last_hidden_state[:, 0]

            #conv_query_embs = output.encoder_last_hidden_state.to(args.device)
            #s = torch.sum(conv_query_embs * bt_conv_query_mask.unsqueeze(-1).float(), axis=1)
            #d = bt_conv_query_mask.sum(axis=1, keepdim=True).float()
            #conv_query_embs = s / d
            #breakpoint()
            #with torch.no_grad():
            #breakpoint()
            #rewrite = query_encoder(input_ids=bt_oracle, attention_mask=bt_oracle_mask, labels=bt_oracle_query)
            #rewrite_embs = rewrite.encoder_last_hidden_state[:, 0]#.detach()
            #s = torch.sum(rewrite_embs * bt_oracle_mask.unsqueeze(-1).float(), axis=1)
            #d = bt_oracle_mask.sum(axis=1, keepdim=True).float()
            #rewrite_embs = s / d
            #align_loss = cal_kd_loss(conv_query_embs, rewrite_embs)


            #transform_dim = nn.Linear(1024, 768).to(args.device)
            #conv_query_embs = transform_dim(output.encoder_last_hidden_state[:, 0])

            #with torch.no_grad():
                # freeze passage encoder's parameters
                #pos_doc_embs = passage_encoder(bt_pos_docs, bt_pos_docs_mask).detach()  # B * dim
                #neg_doc_embs = passage_encoder(bt_neg_docs, bt_neg_docs_mask).detach()  # B * dim, hard negative

            #ranking_loss = cal_ranking_loss(conv_query_embs, pos_doc_embs, neg_doc_embs)
            #ranking_loss = cal_kd_loss(conv_query_embs, pos_doc_embs)
            #loss = decode_loss + args.alpha * ranking_loss
            align_loss = decode_loss
            loss = decode_loss #+ args.alpha * align_loss
            #if args.gradient_accumulation_steps:
            #    loss = loss / args.gradient_accumulation_steps
            loss.backward()
            #if args.gradient_accumulation_steps and global_step % args.gradient_accumulation_steps == 0:
            #    torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), args.max_grad_norm)
            #    optimizer.step()
            #    scheduler.step()
            #else:
            torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            if args.print_steps > 0 and global_step % args.print_steps == 0:
                logger.info("Epoch = {}, Global Step = {}, align_loss = {}, decode loss = {}, total loss = {}".format(
                                epoch + 1,
                                global_step,
                                align_loss.item(),
                                decode_loss.item(),
                                loss.item()))

            #log_writer.add_scalar("train_ranking_loss, decode_loss, total_loss", ranking_loss, decode_loss, loss, global_step)
            
            global_step += 1    # avoid saving the model of the first step.
            # save model finally
            if best_loss > loss:
                save_model(args, query_encoder, query_tokenizer, save_model_order, epoch, global_step, loss.item())
                best_loss = loss
                logger.info("Epoch = {}, Global Step = {}, align_loss = {}, decode loss = {}, total loss = {}".format(
                                epoch + 1,
                                global_step,
                                align_loss.item(),
                                decode_loss.item(),
                                loss.item()))
                
    logger.info("Training finish!")          
         


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_query_encoder", type=str, default="/media/nvme/fengran/checkpoints/flan-t5-large")
    #parser.add_argument("--pretrained_query_encoder", type=str, default="/media/nvme/fengran/checkpoints/llama-2-7b-chat-hf")
    #parser.add_argument("--pretrained_passage_encoder", type=str, default="/media/nvme/fengran/checkpoints/ad-hoc-ance-msmarco")

    #parser.add_argument("--train_file_path", type=str, default="/media/nvme/fengran/TopiOCQA/train.json")
    #parser.add_argument("--target_file_path", type=str, default="/media/nvme/fengran/TopiOCQA/Mistral_train_with_doc_context_search_q_reorder.json")
    #parser.add_argument("--log_dir_path", type=str, default="/media/nvme/fengran/train_topiocqa/Log")
    #parser.add_argument('--model_output_path', type=str, default="/media/nvme/fengran/output/train_topiocqa/Checkpoint")
    parser.add_argument("--train_file_path", type=str, default="/media/nvme/fengran/QReCC/train_with_doc.json")
    parser.add_argument("--target_file_path", type=str, default="/media/nvme/fengran/QReCC/train_with_rewritten_q_new.json")
    parser.add_argument("--log_dir_path", type=str, default="/media/nvme/fengran/train_qrecc/Log")
    parser.add_argument('--model_output_path', type=str, default="/media/nvme/fengran/output/train_qrecc/Checkpoint")
    
    parser.add_argument("--collate_fn_type", type=str, default="flat_concat_for_train")
    parser.add_argument("--decode_type", type=str, default="search_q")
    parser.add_argument("--dataset", type=str, default="qrecc")
    parser.add_argument("--use_prefix", type=bool, default=False)
    #parser.add_argument("--use_align", type=bool, default=False)
    #parser.add_argument('--query_num', type=int, default=2)
    parser.add_argument("--per_gpu_train_batch_size", type=int,  default=4)
    parser.add_argument("--use_data_percent", type=float, default=1)
    
    parser.add_argument("--num_train_epochs", type=int, default=10, help="num_train_epochs")
    parser.add_argument("--max_query_length", type=int, default=32, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=384, help="Max doc length")
    parser.add_argument("--max_response_length", type=int, default=64, help="Max response length")
    parser.add_argument("--max_concat_length", type=int, default=512, help="Max concatenation length of the session")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--alpha", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--disable_tqdm", type=bool, default=True)

    parser.add_argument("--print_steps", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--num_warmup_portion", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    args = parser.parse_args()

    # pytorch parallel gpu
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")#, args.local_rank)
    #device = "cpu"
    args.device = device

    return args


if __name__ == '__main__':
    args = get_args()
    set_seed(args)
    log_writer = SummaryWriter(log_dir = args.log_dir_path)
    train(args, log_writer)
    log_writer.close()
