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

qid2genquery = {}
qid2validgenquery = {}
query_lenseg = []
valid_qid = []
mrr_list = []
bad_qid = []

def split_to_chunks(data, num_chunk):
    chunk_size = len(data) // num_chunk
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i: i + chunk_size])
    return chunks

def run_bm25(args):
    query_list = []
    qid_list = []
    #max_seq_length = 64

    with open(args.test_file_path, "r") as f:
        data = f.readlines()#[:20000]

    with open(args.search_q_gold_qrel_file_path, "r") as f:
        for line in f:
            qid, _, pid, rel = line.split('\t')
            valid_qid.append(qid)

    for line in tqdm(data):
        line = json.loads(line)
        generated_query = line[args.key_name]
        while '' in generated_query:
            generated_query.remove('')

        query_nums = 0
        for idx in range(len(generated_query)):
            query = generated_query[idx]
            qid = line['sample_id'] + '_' + str(idx + 1)
            if qid not in valid_qid:
                continue
            query_list.append(query)
            qid_list.append(qid)
            qid2genquery[qid] = query
            query_nums += 1

        query_lenseg.append(query_nums) # len = test_set_query, each element indicate the nums of generated query

        #query = query.split(" ")[:max_seq_length]
        #query = " ".join(query)
    qid_chunks = split_to_chunks(qid_list, num_chunk=args.split_num_chunk)
    query_chunks = split_to_chunks(query_list, num_chunk=args.split_num_chunk)

    assert len(query_lenseg) == len(data)
    assert sum(query_lenseg) == len(query_list)
    # pyserini search
    searcher = LuceneSearcher(args.index_dir_path)
    searcher.set_bm25(args.bm25_k1, args.bm25_b)
    for chunk_id in range(len(qid_chunks)):
        hits = searcher.batch_search(query_chunks[chunk_id], qid_chunks[chunk_id], k = args.top_n, threads = 20)
        with open(oj(args.retrieval_output_path, "llama_train_pseudo_doc_0.trec.{}".format(chunk_id)), "w") as f:
            for qid in qid_chunks[chunk_id]:
                if qid not in hits:
                    print("{} not in hits".format(qid))
                    continue
                for i, item in enumerate(hits[qid]):
                    f.write("{} {} {} {} {}".format(qid, "Q0", item.docid, i + 1, -i - 1 + 200, item.score))
                    f.write('\n')
                if qid not in qid2validgenquery:
                    qid2validgenquery[qid] = []
                else:
                    qid2validgenquery[qid].append(qid2genquery[qid])

        res, mrr_list, ndcg_3_list, recall_100_list, bad_qid = print_res(oj(args.retrieval_output_path, "llama_train_pseudo_doc_0.trec.{}".format(chunk_id)), args.search_q_gold_qrel_file_path, args.rel_threshold)            
        #assert sum(query_lenseg) == len(mrr_list)
        reorder_gen_query_search(args, ndcg_3_list, bad_qid)
        print("{} chunk search OK!".format(chunk_id))

def reorder_gen_query_search(args, score_list, bad_qid):
    with open(args.test_file_path, "r") as f:
        data = f.readlines()#[:20000]

    query_list = []
    qid_list = []
    cur, cur_qid = 0, 0
    with open(args.test_reoreder_file_path, "w") as g:
        for line in tqdm(data):
            line = json.loads(line)
            query2score = {}
            reorder_list = []
            generated_query = line[args.key_name]
            qid = line["sample_id"]
            query_nums = len(generated_query)
            scores = score_list[cur: cur + query_nums]
            for i in range(query_nums):
                query2score[generated_query[i]] = scores[i]
            query2score_list = sorted(query2score.items(), key=lambda x: x[1], reverse=True) # list[(key, value)]
            for i in range(len(query2score_list)):
                reorder_list.append(query2score_list[i][0])
            #reorder_list = [key for key in reorder_list if key != '']
            line[args.key_name] = reorder_list
            g.write(json.dumps(line) + '\n')
            cur_qid += 1
            query_list.append(line[args.key_name][0])
            qid_list.append(qid)
            cur += query_nums

    # useful for test set only
    '''
    #assert cur == sum(query_lenseg)
    qid_chunks = split_to_chunks(qid_list, num_chunk=args.split_num_chunk)
    query_chunks = split_to_chunks(query_list, num_chunk=args.split_num_chunk)

    # pyserini search
    searcher = LuceneSearcher(args.index_dir_path)
    searcher.set_bm25(args.bm25_k1, args.bm25_b)
    for chunk_id in range(len(qid_chunks)):
        hits = searcher.batch_search(query_chunks[chunk_id], qid_chunks[chunk_id], k = args.top_n, threads = 20)
        
        with open(oj(args.retrieval_output_path, "best_llama_train_context_gen_query.trec.{}".format(chunk_id)), "w") as f:
            for qid in qid_chunks[chunk_id]:
                if qid not in hits:
                    print("{} not in hits".format(qid))
                    continue
                for i, item in enumerate(hits[qid]):
                    f.write("{} {} {} {} {}".format(qid, "Q0", item.docid, i + 1, -i - 1 + 200, item.score))
                    f.write('\n')

        if not args.not_perform_evaluation:
            res, _, _, _, _ = print_res(oj(args.retrieval_output_path, "best_llama_train_context_gen_query.trec.{}".format(chunk_id)), args.gold_qrel_file_path, args.rel_threshold)
            logger.info("---------------------Evaluation results:---------------------")    
            logger.info(res)
        print("{} chunk search OK!".format(chunk_id))
    '''

def print_res(run_file, qrel_file, rel_threshold):
    with open(run_file, 'r' )as f:
        run_data = f.readlines()
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = {}
    qrels_ndcg = {}
    runs = {}
    
    qrels_q = []
    for line in qrel_data:
        line = line.strip().split()
        query = line[0]
        passage = line[2]
        rel = int(line[3])
        if query not in qrels:
            qrels[query] = {}
            qrels_q.append(query)
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
    
    run_q = []
    for line in run_data:
        line = line.split(" ")
        query = line[0]
        passage = line[2]
        rel = int(line[4])
        if query not in runs:
            runs[query] = {}
            run_q.append(query)
        runs[query][passage] = rel

    for key in qrels_q:
        if key not in runs:
            bad_qid.append(key)

    print(bad_qid)
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
            "MRR": round(np.average(mrr_list)*100, 5),
            "NDCG@3": round(np.average(ndcg_3_list)*100, 5), 
            "Recall@5": round(np.average(recall_5_list)*100, 5),
            "Recall@10": round(np.average(recall_10_list)*100, 5),
            "Recall@20": round(np.average(recall_20_list)*100, 5),
            "Recall@100": round(np.average(recall_100_list)*100, 5),
            "MAP": round(np.average(map_list)*100, 5)
        }
    return res, mrr_list, ndcg_3_list, recall_100_list, bad_qid

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--index_dir_path", type = str, default="/media/nvme/fengran/index/bm25_topiocqa")
    parser.add_argument("--test_file_path", type = str, default="/media/nvme/fengran/TopiOCQA/mistral_test_with_doc_newcontext_search_q_new.json")
    parser.add_argument("--test_reoreder_file_path", type = str, default="/media/nvme/fengran/TopiOCQA/mistral_test_with_doc_newcontext_search_q_reorder.json")
    parser.add_argument("--search_q_gold_qrel_file_path", type=str, default="/media/nvme/fengran/TopiOCQA/mistral_topiocqa_test_qrel_newcontext.tsv")
    parser.add_argument("--gold_qrel_file_path", type=str, default="/media/nvme/fengran/TopiOCQA/topiocqa_test_qrel.tsv")
    parser.add_argument("--not_perform_evaluation", action="store_true", default=False)
    parser.add_argument("--use_context", action="store_true", default=True)
    parser.add_argument("--QR", action="store_true", default=False)
    
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--bm25_k1", type=float, default=0.9)   # 0.82 for qrecc, 0.9 for topiocqa
    parser.add_argument("--bm25_b", type=float, default=0.4)    # 0.68 for qrecc, 0.4 for topiocqa
    parser.add_argument("--rel_threshold", type=int, default=1)
    parser.add_argument("--split_num_chunk", type=int, default=1)

    parser.add_argument("--retrieval_output_path", type=str, default="/media/nvme/fengran/output/bm25_topiocqa")
    parser.add_argument("--force_emptying_dir", action="store_true", default=False)

    args = parser.parse_args()

    model_name = "mistral"
    #model_name = "llama-2-7b-chat-hf"

    if args.use_context:
        if not args.QR:
            args.key_name = model_name + "_context_generated_query"
        else:
            args.key_name = model_name + "_context_rewritten_query"
    else:
        args.skey_name = model_name + "_generated_query"

    check_dir_exist_or_build([args.retrieval_output_path], args.force_emptying_dir)
    #json_dumps_arguments(oj(args.retrieval_output_path, "parameters.txt"), args)
 
    logger.info("---------------------The arguments are:---------------------")
    pprint(args)

    return args

if __name__ == '__main__':
    args = get_args()
    run_bm25(args)
    #print_res("/media/nvme/fengran/output/bm25_topiocqa/best_llama_context_gen_query.trec.0", "/media/nvme/fengran/TopiOCQA/topiocqa_qrel.tsv", 1)

