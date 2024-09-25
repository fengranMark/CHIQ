from json.tool import main
import json
from tqdm import tqdm, trange
import csv
import random

def gen_topiocqa_qrel(raw_dev_file_path, output_qrel_file_path):
    '''
    raw_dev_file_path = "gold_dev.json"
    output_qrel_file_path = "topiocqa_qrel.trec"
    '''
    with open(raw_dev_file_path, "r") as f:
        data = json.load(f)
    
    with open(output_qrel_file_path, "w") as f:
        for line in tqdm(data):
            sample_id = "{}_{}_{}".format("TopiOCQA-Dev", line["conv_id"], line["turn_id"])
            for pos in line["positive_ctxs"]:
                #pid = int(pos["passage_id"]) - 1
                pid = int(pos["passage_id"])
                f.write("{} {} {} {}".format(sample_id, 0, pid, 1))
                f.write('\n')


def gen_train_test_files(raw_train_file_path, raw_dev_file_path, output_train_file_path, ouput_test_file_path, collection_file_path):
    '''
    raw_train_file_path = "gold_train.json"
    raw_dev_file_path = "gold_dev.json"
    output_train_file_path = "train.json"
    ouput_test_file_path = "test.json"
    collection_file_path = "full_wiki_segments.tsv"
    '''
    qid2passage = {}
    with open(collection_file_path, 'r') as fin:
        reader = csv.reader(fin, delimiter="\t")
        for i, row in enumerate(tqdm(reader)):
            if row[0] == "id": # ['id', 'text', 'title'] id begin from 1
                continue
            idx, text, title = int(row[0]), row[1], ' '.join(row[2].split(' [SEP] '))
            qid2passage[idx] = " ".join([title, text])

    with open(raw_train_file_path, "r") as f:
        data = json.load(f)
    
    last_conv_id = -1
    last_response = ""
    context_queries_and_answers = []
    context_pos_docs_pids = set()
    random_pid = list(range(25700592))

    with open(output_train_file_path, "w") as f:
        for line in tqdm(data):
            sample_id = "{}_{}_{}".format("TopiOCQA-Train", line["conv_id"], line["turn_id"])
            query = line["question"]
            answers = line["answers"]
            if len(answers) == 0:
                answer = "UNANSWERABLE"
            else:
                answer = answers[0]

            positive_ctxs = line["positive_ctxs"]
            pos_docs = []
            pos_docs_pids = []
            for pos in positive_ctxs:
                passage = pos["title"].rstrip().replace(' [SEP] ', ' ') + ' ' + pos["text"].rstrip()
                pos_docs.append(passage)
                pos_docs_pids.append(int(pos["passage_id"]))            
            # hard_negative_ctxs = line["hard_negative_ctxs"]
            # negative_ctxs = line["negative_ctxs"]

            record = {}
            record["sample_id"] = sample_id
            record["cur_utt_text"] = query
            if int(line["conv_id"]) != last_conv_id:
                context_queries_and_answers = []
                context_pos_docs_pids = set()
                last_response = ""
            #record["ctx_utts_text"] = context_queries_and_answers
            record["last_response"] = last_response
            record["pos_docs"] = pos_docs
            record["pos_docs_pids"] = pos_docs_pids

            prepos_neg_docs_pids = list(context_pos_docs_pids - set(pos_docs_pids))
            neg_docs = []
            neg_docs_pids = []
            if len(prepos_neg_docs_pids):
                neg_docs_pids.append(random.choice(prepos_neg_docs_pids))
            else:
                neg_docs_pids.append(random.choice(random_pid))
            neg_docs.append(qid2passage[neg_docs_pids[0]])

            record["neg_docs"] = neg_docs
            record["neg_docs_pids"] = neg_docs_pids
            record["prepos_neg_docs_pids"] = prepos_neg_docs_pids
            f.write(json.dumps(record))
            f.write('\n')

            last_response = positive_ctxs[0]["title"].rstrip().replace(' [SEP] ', ' ') + ' ' + positive_ctxs[0]["text"].rstrip()
            context_pos_docs_pids |= set(pos_docs_pids)
            #context_queries_and_answers.append(query)
            #context_queries_and_answers.append(answer)
            last_conv_id = int(line["conv_id"])


    with open(raw_dev_file_path, "r") as f:
        data = json.load(f)
    
    last_conv_id = -1
    last_response = ""
    context_queries_and_answers = []
    with open(ouput_test_file_path, "w") as f:
        for line in tqdm(data):
            sample_id = "{}_{}_{}".format("TopiOCQA-Dev", line["conv_id"], line["turn_id"])
            query = line["question"]
            answers = line["answers"]
            if len(answers) == 0:
                answer = "UNANSWERABLE"
            else:
                answer = answers[0]

            positive_ctxs = line["positive_ctxs"]
            pos_docs = []
            pos_docs_pids = []
            for pos in positive_ctxs:
                passage = pos["title"].rstrip().replace(' [SEP] ', ' ') + ' ' + pos["text"].rstrip()
                pos_docs.append(passage)
                pos_docs_pids.append(int(pos["passage_id"]))
            # hard_negative_ctxs = line["hard_negative_ctxs"]
            # negative_ctxs = line["negative_ctxs"]

            record = {}
            record["sample_id"] = sample_id
            record["cur_utt_text"] = query
            if int(line["conv_id"]) != last_conv_id:
                context_queries_and_answers = []
                context_pos_docs_pids = set()
            #record["ctx_utts_text"] = context_queries_and_answers
            record["last_response"] = last_response
            record["pos_docs"] = pos_docs
            record["pos_docs_pids"] = pos_docs_pids

            prepos_neg_docs_pids = list(context_pos_docs_pids - set(pos_docs_pids))
            neg_docs = []
            neg_docs_pids = []
            if len(prepos_neg_docs_pids):
                neg_docs_pids.append(random.choice(prepos_neg_docs_pids))
            else:
                neg_docs_pids.append(random.choice(random_pid))
            neg_docs.append(qid2passage[neg_docs_pids[0]])

            record["neg_docs"] = neg_docs
            record["neg_docs_pids"] = neg_docs_pids
            record["prepos_neg_docs_pids"] = prepos_neg_docs_pids
            f.write(json.dumps(record))
            f.write('\n')

            last_response = positive_ctxs[0]["title"].rstrip().replace(' [SEP] ', ' ') + ' ' + positive_ctxs[0]["text"].rstrip()
            context_pos_docs_pids |= set(pos_docs_pids)
            #context_queries_and_answers.append(query)
            #context_queries_and_answers.append(answer)
            last_conv_id = int(line["conv_id"])

def merge_bm25_neg_info(bm25_run_file, orig_file, new_file):
    qid2bm25_pid = {}
    with open(bm25_run_file, 'r') as f:
        data = f.readlines()

    for line in data:
        line = line.strip().split()
        qid, pid = line[0], int(line[2])
        if qid not in qid2bm25_pid:
            qid2bm25_pid[qid] = [pid]
        else:
            qid2bm25_pid[qid].append(pid)

    with open(orig_file, 'r') as f:
        ori_data = f.readlines()

    with open(new_file, 'w') as g:
        for line in ori_data:
            record = json.loads(line)
            qid = record["sample_id"]
            pos_docs_pids = record["pos_docs_pids"]
            bm25_hard_neg_docs_pids = []
            for pid in qid2bm25_pid[qid]:
                if pid not in pos_docs_pids:
                    bm25_hard_neg_docs_pids.append(pid)
            record["bm25_hard_neg_docs_pids"] = bm25_hard_neg_docs_pids
            g.write(json.dumps(record))
            g.write('\n')
        
            
def extract_doc_content_of_bm25_hard_negs_for_train_file(collection_path, 
                                                         train_inputfile, 
                                                         train_outputfile_with_doc,
                                                         neg_ratio=2):
    '''
    - collection_path = "collection.tsv"
    - train_inputfile = "train.json"
    - train_outputfile_with_doc = "train_with_neg.json"       
    '''
    with open(train_inputfile, 'r') as f:
        data = f.readlines()
    
    pid2passage = {}
    with open(collection_file_path, 'r') as fin:
        reader = csv.reader(fin, delimiter="\t")
        for i, row in enumerate(tqdm(reader)):
            if row[0] == "id": # ['id', 'text', 'title'] id begin from 1
                continue
            idx, text, title = int(row[0]), row[1], ' '.join(row[2].split(' [SEP] '))
            pid2passage[idx] = " ".join([title, text])
    
    # Merge doc content to the train file
    with open(train_outputfile_with_doc, 'w') as fw:
        for line in data:
            line = json.loads(line)
            pos_docs_pids = line["pos_docs_pids"]
            neg_docs_text = []
            for pid in line["bm25_hard_neg_docs_pids"]: #[:neg_ratio]:
                if pid in pid2passage and pid not in pos_docs_pids:
                    neg_docs_text.append(pid2passage[pid])
            
            line["bm25_hard_neg_docs"] = neg_docs_text
            
            fw.write(json.dumps(line))
            fw.write('\n')

         

def modify_pos_docs(conv_sample, pos_docs_text):
    '''
    Modify the pos doc content based on the current conversational sample 
    to avoid simply string-match, enhance model generalization ability
    '''
    return pos_docs_text

def modify_neg_docs(conv_sample, neg_docs_text):
    '''
    Modify the neg doc content based on the current conversational sample 
    to avoid simply string-match, enhance model generalization ability
    '''
    return neg_docs_text
            


if __name__ == "__main__":
    
    raw_train_file_path = "./gold_train.json" # data.gold_passages_info.all_history.train
    raw_dev_file_path = "./gold_dev.json" # data.gold_passages_info.all_history.dev
    output_train_file_path = "./train.json"
    output_test_file_path = "./test.json"
    collection_file_path = "./datasets/topiocqa/full_wiki_segments.tsv"
    gen_train_test_files(raw_train_file_path, raw_dev_file_path, output_train_file_path, output_test_file_path, collection_file_path)

    raw_dev_file_path = "./gold_dev.json"
    output_qrel_file_path = "./topiocqa_qrel.trec"
    gen_topiocqa_qrel(raw_dev_file_path, output_qrel_file_path)


    bm25_run_file = "output/topiocqa/bm25/bm25_train_for_hardneg_res.trec"
    train_inputfile = "train_with_gold_rel_p.json"
    train_outputfile_with_doc = "train_with_gold_rel_p_neg.json"
    merge_bm25_neg_info(bm25_run_file, train_inputfile, train_outputfile_with_doc)
    extract_doc_content_of_bm25_hard_negs_for_train_file(collection_file_path, train_outputfile_with_doc, train_outputfile_with_doc)

