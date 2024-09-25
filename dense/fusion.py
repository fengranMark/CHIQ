import sys
sys.path.append('..')
sys.path.append('.')

import json,string, re
import pytrec_eval
from collections import Counter

import argparse
from collections import defaultdict
import time
import numpy as np

def print_trec_res(run_file, qrel_file, rel_threshold):
    with open(run_file, 'r' )as f:
        run_data = f.readlines()
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = {}
    qrels_ndcg = {}
    runs = {}
    
    for line in qrel_data:
        if args.dataset == "topiocqa":
            line = line.split()
        else:
            line = line.split('\t')
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
        #query = line[0]
        query = line[0].replace('-','_')
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
    
    print("---------------------Evaluation results:---------------------")    
    print(res)
    return res

# (qid, Q0, docid, rank, rel, score, bm25)
def read_rank_list(args, file):
    qid_docid_list = defaultdict(list)
    qid_score_list = defaultdict(list)
    with open(file, 'r') as f:
        for line in f:
            try:
                qid, _, docid, rank, _, score, _ = line.strip().split(' ')
                qid = qid.replace('-','_')
            except:
                qid, _, docid, rank, _, score = line.strip().split(' ')
                qid = qid.replace('-','_')

            
            

            qid_docid_list[qid].append(docid)
            qid_score_list[qid].append(float(score))
    return qid_docid_list, qid_score_list # qid: [docid] [score]

def fuse(args, docid_list0, docid_list1, doc_score_list0, doc_score_list1, alpha):
    if args.fusion_method == "CQE_hybrid":
        score = defaultdict(float)
        score0 = defaultdict(float)
        for i, docid in enumerate(docid_list0):
            score0[docid]+=doc_score_list0[i]
        #breakpoint()
        min_val0 = min(doc_score_list0)
        min_val1 = min(doc_score_list1)
        for i, docid in enumerate(docid_list1):
            if score0[docid]==0:
                score[docid]+=min_val0 + doc_score_list1[i]*alpha
            else:
                score[docid]+=doc_score_list1[i]*alpha
        for i, docid in enumerate(docid_list0):
            if score[docid]==0:
                score[docid]+=min_val1*alpha
            score[docid]+=doc_score_list0[i]
        score= {k: v for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)}
        return score
    elif args.fusion_method == "combine_max":
        score = {}
        for i, docid in enumerate(docid_list0):
            score[docid] = doc_score_list0[i]
        for i, docid in enumerate(docid_list1):
            if docid not in score:
                score[docid] = doc_score_list1[i]
            else:
                score[docid] = max(score[docid], doc_score_list1[i])
        score= {k: v for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)}
        return score
    elif args.fusion_method == "combine_sum":
        score = {}
        for i, docid in enumerate(docid_list0):
            score[docid] = doc_score_list0[i]
        for i, docid in enumerate(docid_list1):
            if docid not in score:
                score[docid] = doc_score_list1[i]
            else:
                score[docid] = score[docid] + doc_score_list1[i]
        score= {k: v for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)}
        return score

    elif args.fusion_method == "RRF": # bug
        score0, score1, score = {}, {}, {}
        for i, docid in enumerate(docid_list0):
            score0[docid] = doc_score_list0[i]
        for i, docid in enumerate(docid_list1):
            score1[docid] = doc_score_list1[i] 
        
        for key, value in score0.items():
            if key in score1:
                score[key] = 1 / (score0[key] + score1[key] + 60)
            else:
                score[key] = 1 / (score0[key] + 60) 
        for key, value in score1.items():
            if key not in score0:
                score[key] = 1 / (score1[key] + 60) 
        score= {k: v for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)}
        return score


def hybrid_fusion(args, rank_file0, rank_file1, fusion_output, trec_gold):
    print('Read ranked list0...')
    qid_docid_list0, qid_score_list0 = read_rank_list(args, rank_file0)
    print('Read ranked list1...')
    qid_docid_list1, qid_score_list1 = read_rank_list(args, rank_file1)
    #breakpoint()
    qids = qid_docid_list0.keys()
    #print(qids)
    fout = open(fusion_output, 'w')
    for j, qid in enumerate(qids):
        #  pid : score   for each qid
        rank_doc_score = fuse(args, qid_docid_list0[qid], qid_docid_list1[qid], qid_score_list0[qid], qid_score_list1[qid], args.alpha)
        for rank, doc in enumerate(rank_doc_score):
            if rank==args.topk:
                break
            score = rank_doc_score[doc]
            fout.write('{} Q0 {} {} {} {} {}\n'.format(qid, doc, rank + 1, str(-rank - 1 + 200), score, "fusion"))
    print('fusion finish')
        #pbar.update(j + 1)
    #time_per_query = (time.time() - start_time)/len(qids)
    #print('Fusing {} queries ({:0.3f} s/query)'.format(len(qids), time_per_query))
    trec_res = print_trec_res(fusion_output, trec_gold, args.rel_threshold)
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rank list fusion')
    parser.add_argument('--topk', default=100, type=int, help='number of hits to retrieve')
    parser.add_argument("--alpha", type=float, default=0.4) # topiocqa: bm25 KD-FT 5 bm25-dense 13 qrecc: ance bm25 KD-FT 3.5 
    parser.add_argument("--rel_threshold", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="cast19")
    parser.add_argument("--fusion_method", type=str, default="CQE_hybrid")
    args = parser.parse_args()

    if args.dataset == "qrecc":
        rank_file0 = "/media/nvme/fengran/output/ance_qrecc/ftflant5large_searchq_noinstruc.trec"
        #rank_file0 = "/media/nvme/fengran/output/fusion_qrecc/llama_QReCC_test_search_query+single_response_ance.trec"
        rank_file1 = "/media/nvme/fengran/output/ance_qrecc/llama_QReCC_test_search_query+single_response+summary_new.trec"
        hybrid_fusion_output = "/media/nvme/fengran/output/fusion_qrecc/ftflant5+llama_QReCC_test_search_query+single_response+summary_new_ance.trec"
        trec_gold = "/media/nvme/fengran/QReCC/qrecc_qrel.tsv"
        hybrid_fusion(args, rank_file0, rank_file1, hybrid_fusion_output, trec_gold)
    elif args.dataset == "topiocqa":
        rank_file1 = "/media/nvme/fengran/output/bm25_topiocqa/flant5_ft.trec"
        #rank_file1 = "/media/nvme/fengran/output/fusion_topiocqa/llama_TopiOCQA_test_search_query+single_response_ance.trec"
        rank_file0 = "/media/nvme/fengran/output/bm25_topiocqa/llama_TopiOCQA_test_search_query+single_response+explain_question_cur.trec"
        hybrid_fusion_output = "/media/nvme/fengran/output/fusion_topiocqa/flant5ft+llama_TopiOCQA_test_search_query+single_response+explain_question_cur_bm25.trec"
        trec_gold = "/media/nvme/fengran/TopiOCQA/topiocqa_qrel.tsv"
        hybrid_fusion(args, rank_file0, rank_file1, hybrid_fusion_output, trec_gold)
    elif args.dataset == "cast19":
        rank_file1 = "/media/nvme/fengran/output/ance_cast/mistral_cast19_flant5large-noinstruc.trec"
        #rank_file1 = "/media/nvme/fengran/output/fusion_topiocqa/llama_TopiOCQA_test_search_query+single_response_ance.trec"
        rank_file0 = "/media/nvme/fengran/output/ance_cast/mistral_cast19_test_search_query+single_response+explain_question.trec"
        hybrid_fusion_output = "/media/nvme/fengran/output/fusion_cast/flant5ft+mistral_cast19_test_search_query+single_response+explain_question.trec"
        trec_gold = "/media/nvme/fengran/cast19/qrels.tsv"
        hybrid_fusion(args, rank_file0, rank_file1, hybrid_fusion_output, trec_gold)
    elif args.dataset == "cast20":
        rank_file1 = "/media/nvme/fengran/output/ance_cast/mistral_cast20_flant5large-noinstruc.trec"
        #rank_file1 = "/media/nvme/fengran/output/fusion_topiocqa/llama_TopiOCQA_test_search_query+single_response_ance.trec"
        rank_file0 = "/media/nvme/fengran/output/ance_cast/mistral_cast20_test_search_query+single_response+explain_question.trec"
        hybrid_fusion_output = "/media/nvme/fengran/output/fusion_cast/flant5ft+mistral_cast20_test_search_query+single_response+explain_question.trec"
        trec_gold = "/media/nvme/fengran/cast20/qrels.tsv"
        hybrid_fusion(args, rank_file0, rank_file1, hybrid_fusion_output, trec_gold)

