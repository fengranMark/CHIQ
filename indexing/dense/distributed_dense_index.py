from re import S
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import gc
import sys
import os
sys.path.append('..')
sys.path.append('.')
import time
import argparse
import numpy as np
from os.path import join as oj
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from utils import set_seed, check_dir_exist_or_build, json_dumps_arguments, pstore, pload

from models import load_model

from libs import CollateClass
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,5,6,7'


def distributed_index_dataset_generator(collection_path, num_doc_per_block):
    with open(collection_path, "r") as f:
            docs = []
            for line in f:
                line = line.strip().split('\t') # doc_id, text
                if len(line) == 1:
                    line.append("")
                doc_id, doc = line[0], line[1]
                docs.append([doc_id, doc])
                if len(docs) == num_doc_per_block:
                    yield docs
                    docs = []
            yield docs


def dense_indexing(args):
    tokenizer, model = load_model(args.model_type, "doc", args.pretrained_doc_encoder_path)
    model.to(args.device)
    model = DDP(model, device_ids = [args.local_rank], output_device=args.local_rank)
    dist.barrier()

    indexing_batch_size = args.per_gpu_index_batch_size
    indexing_dataset_generator = distributed_index_dataset_generator(args.collection_path, args.num_docs_per_block)
    if args.model_type == "TCT-ColBERT":
        prefix = "[ D ] "   # note that [CLS] will be added by tokenizer with the "add_special_token" param
    else:
        prefix = ""
    collate_func = CollateClass(args, tokenizer, prefix=prefix)

    for cur_block_id, raw_docs in enumerate(indexing_dataset_generator):
        doc_ids = []
        doc_embeddings = []
        distributed_sampler = DistributedSampler(raw_docs)
        dataloader =  DataLoader(raw_docs, 
                                 sampler=distributed_sampler,
                                 batch_size=indexing_batch_size, 
                                 collate_fn=collate_func.collate_fn,
                                 shuffle=False)
        dist.barrier()
        
        id_is_int = True
        with torch.no_grad():
            model.eval()
            for batch in tqdm(dataloader, desc="Distributed Dense Indexing", position=0, leave=True):
                inputs = {k: v.to(args.device) for k, v in batch.items() if k not in {"id"}}
                batch_doc_embs = model(**inputs)
                batch_doc_embs = batch_doc_embs.detach().cpu().numpy()
                doc_embeddings.append(batch_doc_embs)
                for doc_id in batch["id"]:
                    try:
                        doc_id = int(doc_id)
                    except:
                        id_is_int = False
                    doc_ids.append(doc_id)

        doc_embeddings = np.concatenate(doc_embeddings, axis=0)
        if id_is_int:
            doc_ids = np.array(doc_ids)
        emb_output_path = oj(args.output_index_dir_path, "doc_emb_block.rank_{}.{}.pb".format(dist.get_rank(), cur_block_id))
        embid_output_path = oj(args.output_index_dir_path, "doc_embid_block.rank_{}.{}.pb".format(dist.get_rank(), cur_block_id))
        pstore(doc_embeddings, emb_output_path, high_protocol=True)
        pstore(doc_ids, embid_output_path, high_protocol=True)

        doc_ids = []
        doc_embeddings = []
        dist.barrier()
        
    if dist.get_rank() == 0:
        print("All docs have been stored in {} blocks.".format(cur_block_id + 1))


# a demo, please change the parameters when you use this function.
def merge_blocks_to_large_blocks():
    num_block = 8
    num_rank = 7
    expected_num_doc_per_block = 5000000
    embs = []
    embids = []
    new_block_id = 0
    for block_id in range(num_block):
        for rank in range(num_rank):
            emb_filename = "/media/nvme/fengran/index/ance_cast1920/doc_emb_block.rank_{}.{}.pb".format(rank, block_id)
            embid_filename = "/media/nvme/fengran/index/ance_cast1920/doc_embid_block.rank_{}.{}.pb".format(rank, block_id)
            cur_embs = pload(emb_filename)
            cur_embids = pload(embid_filename)
            embs.append(cur_embs)
            embids.extend(cur_embids)
            print("len embs = {}".format(sum([len(x) for x in embs])))
            print("len embids = {}".format(len(embids)))
            if len(embids) >= expected_num_doc_per_block:
                num_remain = len(embids) - expected_num_doc_per_block
                remain_embids = embids[-num_remain:]
                print("before concat")
                embs = np.concatenate(embs)
                print("after concat")
                remain_embs = embs[-num_remain:]
                
                pstore(embs[:expected_num_doc_per_block], "/media/nvme/fengran/index/ance_cast1920_merged/doc_emb_block.{}.pb".format(new_block_id), True)
                pstore(embids[:expected_num_doc_per_block], "/media/nvme/fengran/index/ance_cast1920_merged/doc_embid_block.{}.pb".format(new_block_id), True)
                new_block_id += 1
                embs = [remain_embs]
                embids = remain_embids
                gc.collect()

    if len(embids) > 0:
        embs = np.concatenate(embs)
        pstore(embs, "/media/nvme/fengran/index/ance_cast1920_merged/doc_emb_block.{}.pb".format(new_block_id), True)
        pstore(embids, "/media/nvme/fengran/index/ance_cast1920_merged/doc_embid_block.{}.pb".format(new_block_id), True)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')  # you need this argument in your scripts for DDP to work
    parser.add_argument('--n_gpu', type=int, default=7, help='The number of used GPU.')

    parser.add_argument("--model_type", type=str, default="ance")
    parser.add_argument("--collection_path", type=str, default="cast20/collection/collection.tsv")
    parser.add_argument("--pretrained_doc_encoder_path", type=str, default="checkpoints/ad-hoc-ance-msmarco")
    
    parser.add_argument("--output_index_dir_path", type=str, default="index/ance_cast1920")
    parser.add_argument("--force_emptying_dir", action="store_true", default=True)

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_data_percent", type=float, default=1.0, help="Percent of samples to use. Faciliating the debugging.")
    parser.add_argument("--per_gpu_index_batch_size", type=int, default=250)
    parser.add_argument("--num_docs_per_block", type=int, default=5000000)

    parser.add_argument("--max_doc_length", type=int, default=384, help="Max doc length, consistent with \"Dialog inpainter\".")


    args = parser.parse_args()
    local_rank = args.local_rank
    args.local_rank = local_rank
    logger.info(args.local_rank)
    # pytorch parallel gpu
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.local_rank)
    device = "cpu"
    args.device = device
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    logger.info(args.device)
    # pytorch parallel gpu

    args.start_running_time = time.asctime(time.localtime(time.time()))
    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)
    #if dist.get_rank() == 0:
    #    check_dir_exist_or_build([args.output_index_dir_path], force_emptying=args.force_emptying_dir)
    #    json_dumps_arguments(oj(args.output_index_dir_path, "parameters.txt"), args)
            
    return args


if __name__ == "__main__":
    #args = get_args()
    #set_seed(args)
    #dense_indexing(args)
    merge_blocks_to_large_blocks()

# python  -m torch.distributed.launch --nproc_per_node 2 xxx.py
