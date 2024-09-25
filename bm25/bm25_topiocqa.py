import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import os
import sys
sys.path.append('..')
sys.path.append('.')
from os.path import join as oj

import json
import argparse
import numpy as np
from tqdm import tqdm
from pprint import pprint

from pyserini.search.lucene import LuceneSearcher
import pytrec_eval

from utils import check_dir_exist_or_build


def split_to_chunks(data, num_chunk):
    chunk_size = len(data) // num_chunk
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i: i + chunk_size])
    return chunks

def run_bm25(args):
    query_list = []
    qid_list = []
    i = 0
    max_seq_length = 512

    #with open(args.test_file, "r") as f:
    #    data = f.readlines()

    #with open(args.default_file, "r") as f:
    #    data_default = f.readlines()

    with open(args.query_file, "r") as f:
        data_query = f.readlines()

    for i in range(len(data_query)):
        query = ""
        #test = json.loads(data[i])
        #default = json.loads(data_default[i])
        query = json.loads(data_query[i])["query"]
        
        query = query.split(" ")[:max_seq_length]
        query = " ".join(query)

        query_list.append(query)
        qid_list.append(json.loads(data_query[i])["sample_id"])

    qid_chunks = split_to_chunks(qid_list, num_chunk=args.split_num_chunk)
    query_chunks = split_to_chunks(query_list, num_chunk=args.split_num_chunk)
    
    # pyserini search
    searcher = LuceneSearcher(args.index_dir_path)
    searcher.set_bm25(args.bm25_k1, args.bm25_b)
    for chunk_id in range(len(qid_chunks)):
        hits = searcher.batch_search(query_chunks[chunk_id], qid_chunks[chunk_id], k = args.top_n, threads = 20)
        
        with open(oj(args.retrieval_output_path, args.output_file_name), "w") as f:
            for qid in qid_chunks[chunk_id]:
                if qid not in hits:
                    print("{} not in hits".format(qid))
                    continue
                for i, item in enumerate(hits[qid]):
                    f.write("{} {} {} {} {} {}".format(qid,
                                                    "Q0",
                                                    item.docid,
                                                    i + 1,
                                                    -i - 1 + 200,
                                                    item.score,
                                                    "bm25"
                                                    ))
                    f.write('\n')


        if not args.not_perform_evaluation:
            res = print_res(oj(args.retrieval_output_path, args.output_file_name), args.gold_qrel_file_path, args.rel_threshold)

        print("{} chunk search OK!".format(chunk_id))
        
    return

def print_res(run_file, qrel_file, rel_threshold):
    with open(run_file, 'r' )as f:
        run_data = f.readlines()
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = {}
    qrels_ndcg = {}
    runs = {}
    
    for line in qrel_data:
        line = line.strip().split()
        query = line[0]
        passage = line[2]
        rel = int(line[3])
        if query not in qrels:
            qrels[query] = {}
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}

        # for NDCG
        qrels_ndcg[query][passage] = rel
        # for MAP, MRR, Recall
        if rel >= rel_threshold:
            rel = 1
        else:
            rel = 0
        qrels[query][passage] = rel
    
    for line in run_data:
        line = line.split(" ")
        query = line[0]
        passage = line[2]
        rel = int(line[4])
        if query not in runs:
            runs[query] = {}
        runs[query][passage] = rel

    # pytrec_eval eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.100"})
    res = evaluator.evaluate(runs)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]
    recall_20_list = [v['recall_20'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res = evaluator.evaluate(runs)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]

    res = {
            #"MAP": np.average(map_list),
            "MRR": round(np.average(mrr_list)*100, 5),
            "NDCG@3": round(np.average(ndcg_3_list)*100, 5), 
            #"Recall@5": round(np.average(recall_5_list)*100, 5),
            "Recall@10": round(np.average(recall_10_list)*100, 5),
            #"Recall@20": round(np.average(recall_20_list)*100, 5),
            "Recall@100": round(np.average(recall_100_list)*100, 5),
        }

    
    logger.info("---------------------Evaluation results:---------------------")    
    logger.info(res)
    return res


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--index_dir_path", type = str, default="/media/nvme/fengran/index/bm25_topiocqa")
    parser.add_argument("--test_file", type = str, default="/media/nvme/fengran/TopiOCQA/test.json")
    parser.add_argument("--default_file", type = str, default="/media/nvme/abbas/convqa/fengran2abbas/TopiOCQA_test_default_query.json")
    parser.add_argument("--query_file", type = str, default="/media/nvme/fengran/output/combine_topiocqa/mistral_TopiOCQA_test_AD+FT.json")
    parser.add_argument("--gold_qrel_file_path", type=str, default="/media/nvme/fengran/TopiOCQA/topiocqa_qrel.tsv")
    parser.add_argument("--not_perform_evaluation", action="store_true", default=False)
    
    #parser.add_argument("--test_input_type", type=str, default="reasoning")
    parser.add_argument("--top_n", type=int, default=100)
    parser.add_argument("--bm25_k1", type=float, default=0.9)   # 0.82 for qrecc, 0.9 for topiocqa
    parser.add_argument("--bm25_b", type=float, default=0.4)    # 0.68 for qrecc, 0.4 for topiocqa
    parser.add_argument("--rel_threshold", type=int, default=1)
    parser.add_argument("--split_num_chunk", type=int, default=1)

    parser.add_argument("--retrieval_output_path", type=str, default="/media/nvme/fengran/output/bm25_topiocqa")
    parser.add_argument("--output_file_name", type=str, default="mistral_TopiOCQA_test_AD+FT.trec")
    parser.add_argument("--force_emptying_dir", action="store_true", default=False)
    
    args = parser.parse_args()

    check_dir_exist_or_build([args.retrieval_output_path], args.force_emptying_dir)
    #json_dumps_arguments(oj(args.retrieval_output_path, "parameters.txt"), args)
 
    logger.info("---------------------The arguments are:---------------------")
    pprint(args)

    return args

if __name__ == '__main__':
    args = get_args()
    run_bm25(args)
    #print_trec_res("/media/nvme/fengran/output/bm25_qrecc/convgqr_flant5.trec.0", "/media/nvme/fengran/QReCC/qrecc_qrel.tsv", rel_threshold=1)
