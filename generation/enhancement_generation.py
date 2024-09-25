import sys
[sys.path.append(i) for i in ['.', '..']]

from utils.utils import *
import torch
from transformers import (
    # LlamaForCausalLM,
    MistralForCausalLM,
    AutoTokenizer
)
DS_NAME_LST = ["TopiOCQA", "QReCC", "cast21", "cast20", "cast19"]
DS_NAME_LST += [ds_name+ext for ext in ["RM", "RL"] for ds_name in DS_NAME_LST[2:]]


PORTION_LST = ["train", "test"]

PROMPT_DICT = {

    "multi_response": \
        """
        Give a one-sentence answer for each of the following questions. Put each answer on a new line. 
        Each answer should not contain any pronouns (e.g. it, he, they, etc..) and 
        should not refer to content from a previous answer:
        """,
    "single_response": \
        """
        Given a series of question-and-answer pairs, along with a new question, your task is to give a one 
        sentence response to the new question.
        """,
    "search_query": \
        """
        Given a series of question-and-answer pairs as context, along with a new question, your task is to convert
         the new question into a search engine query that can be used to retrieve relevant documents. The output
          should be placed in a JSON dictionary as follow: {"query": ""}
        """,
    "explain_response": \
        """
        You are given a question-and-answer pair, where the answer is not clear. Your goal is to write a long version of 
        the answer based on its given context. The generated answer should be one sentence only and less than 20 words.
        """,

    "explain_question": \
        """
        You are given a set of question-answers pairs and a new question that is ambiguous. Your goal is to re-write the 
        question so it became clear. Just write the new question without any introduction.
        """,
    
    "summarize_context": \
        """
        You are given a context in the form of question-answer pairs. Your goal is to write a paragraph that summarizes
         the information in the context. The summary should be short with one sentence for each question answer pair.
        """,

    "new_topic": \
    """
    Given a series of question-and-answer pairs, along with a new question, your task is to determine whether the new 
    question continues the discussion on an existing topic or introduces a new topic. Please respond with either 
    "new_topic" or "old_topic" as appropriate.
    """

}
PROMPT_DICT = {k: re.sub('\s+', ' ', v).strip() + "\n\n" for k, v in PROMPT_DICT.items()}

PROMPT_DICT["keyword_alias"] = \




class LMGenDataCollator:

    def __init__(self, tokenizer,
                       max_length= None,
                       pad_to_multiple_of = None,
                       padding = True,
                       prompt_name=None,
                       model_name=None,
                 ):

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding = padding



        assert self.tokenizer.eos_token

    def __call__(self, features):
        idx_lst = [f.get("idx", [])[:1] for f in features]


        input_ids = [f["input_ids"][-2048:] for f in features] #

        batch = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # if idx_lst[0]:
        batch["idx"] = torch.tensor(idx_lst)

        return batch

##########################
### Process Remote Data ##
##########################
def import_remote_data(args):
    _import_remote_input(args)
    # _import_remote_query(args)
    # _import_remote_bm25(args)


def _import_remote_bm25(args):
    bm25_file_mapping = {"llama_direct_prompt.trec.0": "test_with_rewritten_q_new.json",
                        # "llama-2-7b-chat-hf_context_rewritten_query"[0]
                        "llama_reasoningandresponse_expansion.trec.0": "test_with_responseandreasoning.json",
                        # "ResponseAndReasoning_Query" + "ResponseAndReasoning_Response"
                        "llama_reasoningandresponse_useq_expansion.trec.0": "test_with_responseandreasoning_usesearchq.json",
                        # "ResponseAndReasoning_Query" + "ResponseAndReasoning_Response"
                        "llama_reasoningandresponse.trec.0": "test_with_responseandreasoning.json",
                        # "ResponseAndReasoning_Query"
                        "llama_reasoningthenresponse_expansion.trec.0": "test_with_responsethenreasoning.json",
                        # the same as previous three
                        "llama_reasoningthenresponse_useq_expansion.trec.0": "test_with_responsethenreasoning_usesearchq.json",
                        "llama_reasoningthenresponse.trec.0": "test_with_responsethenreasoning.json",
                        }


    for ds_name in DS_NAME_LST:
        group_dict = {}
        for k, v in bm25_file_mapping.items():
            portion = v.split("_")[0]
            method = k.split(".")[0] #v.split("_with_")[1][:-5] 
            filename = os.path.join(args.remote_project_dir, "output", f"bm25_{ds_name.lower()}", k)
            if not os.path.exists(filename): continue
            key = (portion, method)
            #if method != "llama_reasoningandresponse": continue
            if key not in group_dict: group_dict[key] = defaultdict(list)
            for line in open(filename):
                parts = line.split()
                if len(parts) == 6:
                    sample_idx, _, passage_idx, _, rank_idx, score = parts
                else:
                    sample_idx, _, passage_idx, _, rank_idx = parts
                    score = 0.0
                score = float(score)
                group_dict[key][sample_idx].append((passage_idx, rank_idx, score))

        for (portion, method), dico in group_dict.items():
            print(f"{ds_name}_{portion}_{method}")
            fout = open(os.path.join(args.tmp_dir, f"{ds_name}_{portion}_{method}_bm25.json"), 'w')
            json.dump(dico, fout)
            fout.close()



def _import_remote_query(args):

    for ds_name in DS_NAME_LST:
        d_dir = os.path.join(args.remote_project_dir, ds_name)
        for filename in os.listdir(d_dir):
            if "with" not in filename or not filename.endswith(".json"): continue
            portion = filename.split("_")[0]
            method = filename.split("with_")[1][:-5]
            query_key = ""
            group_dict = {}

            for line in tqdm(open(os.path.join(d_dir, filename))):
                dico = json.loads(line.strip())
                if not query_key:

                    lst = [key for key in dico.keys() if key.endswith("_query")]
                    if len(lst) != 0:
                        query_key = lst[-1]
                    else:
                        print(filename)
                        print(dico.keys())
                        break
                    print(filename, query_key)
                group_dict[dico["sample_id"]] = dico[query_key]

            fout = open(os.path.join(args.tmp_dir, f"{ds_name}_{portion}_{method}_query.json"), 'w')
            json.dump(group_dict, fout)
            fout.close()

def _import_remote_input(args):

    for ds_name in DS_NAME_LST:
        for portion in PORTION_LST:
            print(ds_name, portion)
            filename = os.path.join(args.remote_project_dir, ds_name, f"{portion}_with_search_q.json")
            if not os.path.exists(filename): continue
            fout = open(os.path.join(args.tmp_dir, f"{ds_name}_{portion}_input.json"), 'w')

            for line in tqdm(open(filename)):
                dico = json.loads(line.strip())

                if ds_name != "TopiOCQA":
                    if not dico['ctx_utts_text']:
                        history = [dico['cur_utt_text']]
                    else:
                        history.append(dico['cur_utt_text'])
                else:
                    history = dico['cur_utt_text'].strip().split(" [SEP] ")
                assert len(history) % 2 == 1

                response = dico.get("cur_response_text", "")
                pos_doc_text_lst = dico.get("pos_docs", [])
                pos_doc_idx_lst = dico.get("pos_docs_pids", [])
                row = {"sample_id": dico["sample_id"], "history": history, "response": response,
                       "pos_doc_text_lst": pos_doc_text_lst, "pos_doc_idx_lst": pos_doc_idx_lst}
                # print(history)
                assert len(history) % 2 == 1
                history.append(response)

                json.dump(row, fout)
                fout.write("\n")

            fout.close()

            # copy eval file
            ext = "_train" if portion == "train" else ""
            scr_file = os.path.join(args.remote_project_dir, ds_name, f"{ds_name.lower()}{ext}_qrel.tsv")
            dst_file = os.path.join(args.tmp_dir, f"{ds_name}_{portion}_label.json")
            if os.path.exists(scr_file):
                shutil.copy(scr_file, dst_file)



#########################
### Evaluation Methods ##
##########################
def run_eval(eval_data, pred_data):
    import pytrec_eval

    # pytrec_eval eval
    evaluator = pytrec_eval.RelevanceEvaluator(eval_data["qrels"],
                                               {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.100"})
    res = evaluator.evaluate(pred_data)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]
    recall_20_list = [v['recall_20'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(eval_data["qrels_ndcg"], {"ndcg_cut.3"})
    res = evaluator.evaluate(pred_data)
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

def _load_eval_data(filename, rel_threshold=1):
    # prepare eval labels
    qrels = {}
    qrels_ndcg = {}

    for line in open(filename):
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

    return {"qrels": qrels, "qrels_ndcg": qrels_ndcg}

def _load_dummy_pred_data(args, ds_name, portion):
    bm25_file = os.path.join(args.tmp_dir, f"{ds_name}_{portion}_bm25.json")
    method = "llama_reasoningandresponse"
    rows = json.load(open(bm25_file))[method]
    pred_data = defaultdict(dict)
    for sample_idx, lst  in rows.items():
        for passage_idx, rank_idx, _ in lst:
            pred_data[sample_idx][passage_idx] = int(rank_idx)

    return pred_data

def _load_pred_data(pred_file):
     # pred_file = os.path.join(args.tmp_dir, f"{args.ds_name}_{args.portion}_{args.prompt_name}_{args.model_name}_bm25.json")
    pred_data = defaultdict(dict)
    rows = json.load(open(pred_file))
    for sample_idx, lst in rows.items():
        for passage_idx, rank_idx, _ in lst:
            pred_data[sample_idx][passage_idx] = int(rank_idx)

    return pred_data

def evaluate(args):
    label_file = os.path.join(args.tmp_dir, f"{args.ds_name}_{args.portion}_label.json")
    eval_data = _load_eval_data(label_file, rel_threshold=1)

    for filename in os.listdir(args.tmp_dir):
        if not filename.endswith("_bm25.json"): continue
        if "explain_question" not in filename: continue
        pred_data = _load_pred_data(os.path.join(args.tmp_dir, filename))
        result = run_eval(eval_data, pred_data)
        print(filename, result)

####################
### BM25 Methods  ##
####################
def load_bm25_index(index_path, ds_name):
    from pyserini.search.lucene import LuceneSearcher

    k1, b, rm3 = 0.9, 0.4, False
    if ds_name.lower() == "topiocqa":
        k1, b = 0.9, 0.4
    elif ds_name.lower() == "qrecc":
        k1, b = 0.68, 0.68
    searcher = LuceneSearcher(index_path)
    searcher.set_bm25(k1, b)
    if rm3:
        searcher.set_rm3()

    return searcher


def _mp_search_bm25(args):

    args.max_hits = 200
    output_file = os.path.join(args.tmp_dir, f"{args.ds_name}_{args.portion}_{args.prompt_name}_{args.model_name}_bm25.json")
    if os.path.exists(output_file): return

    query_file = os.path.join(args.tmp_dir, f"{args.ds_name}_{args.portion}_{args.prompt_name}_{args.model_name}_query.json")
    if not os.path.exists(query_file): return

    query_dict = json.load(open(query_file))
    query_lst = list(query_dict.items())

    if args.max_process == 1:
        _search_bm25(args, query_lst)

    wc = len(query_lst)
    bs = int(math.ceil(wc / args.max_process))
    l_begin = 0
    attrs_lst = []

    for p_idx in range(args.max_process):
        l_end = l_begin + bs
        attrs = (args, query_lst[l_begin:l_end])
        attrs_lst.append(attrs)
        l_begin = l_end

    p = multiprocessing.Pool(args.max_process)
    results = p.starmap(_search_bm25, attrs_lst)

    output_dict = {}
    for dico in results:
        output_dict.update(dico)


    fout = open(output_file, 'w')
    json.dump(output_dict, fout)
    fout.close()

def _search_bm25(args, query_lst):
    bm25_index_dir = os.path.join(args.remote_project_dir, "index", f"bm25_{args.ds_name.lower()}")
    searcher = load_bm25_index(bm25_index_dir, args.ds_name)
    output_dict = {}

    for sample_idx, query in tqdm(query_lst):
        hits = searcher.search(query, k=args.max_hits)
        output_dict[sample_idx] = [(hits[rank].docid, args.max_hits-rank, hits[rank].score) for rank in range(len(hits))]

    return output_dict

#########################
### Prompt Generation  ##
#########################
def _get_str_prompt(args, prompt_name, prompt_lst):
    pn = "keyword_alias" if  "keyword_alias" in prompt_name else prompt_name

    if "Mistral" in args.model_name:
        chat_prefix = f"<s>[INST] {PROMPT_DICT[pn]}"
    elif "llama" in args.model_name:
        chat_prefix = f"[INST] <<SYS>> {PROMPT_DICT[pn]}  <</SYS>>"
    elif "gemma" in args.model_name:
        chat_prefix = f"<start_of_turn>user\n{PROMPT_DICT[pn]}\n"
    elif "gpt-3.5-turbo" in args.model_name:
        chat_prefix = f"{PROMPT_DICT[pn]}\n"
    else:
        raise ValueError(f"{args.model_name} is unsupported")

    if "gemma" in args.model_name:
        chat_suffix = f"<end_of_turn>\n<start_of_turn>model"
    elif "gpt-3.5-turbo" in args.model_name:
        chat_suffix = "\n"
    else:
        chat_suffix = "[/INST]"

    return "\n".join([chat_prefix] + prompt_lst + [chat_suffix])

def _generate_prompt_1(args):
    filename = os.path.join(args.tmp_dir, "..", f"{args.ds_name}_{args.portion}_input.json")

    prompt_name_lst = ["multi_response", "single_response", "search_query", "explain_response",
                       "explain_question", "summarize_context", "new_topic"]

    fout_dict = {prompt_name: open(os.path.join(args.tmp_dir, f"{args.ds_name}_{args.portion}_{prompt_name}_{args.model_name}_prompt.json"), 'w')
                 for prompt_name in prompt_name_lst}

    for idx, line in enumerate(open(filename)):
        dico = json.loads(line.strip())
        sample_id = dico["sample_id"]
        question_lst = [item for i, item in enumerate(dico["history"]) if i  % 2 == 0]
        answer_lst = [item for i, item in enumerate(dico["history"]) if i % 2 == 1]
        # if not len(dico["history"]) % 2 == 1:
        #     print(args.portion, dico["history"])
        #     continue
        assert len(dico["history"]) % 2 == 1

        s = _get_str_prompt(args, "multi_response", [q + "?" if not q.endswith("?") else q for q in question_lst])
        json.dump({"idx": idx, "sample_id": sample_id, "prompt": s}, fout_dict["multi_response"])
        fout_dict["multi_response"].write("\n")

        question = question_lst[-1] + "?" if not question_lst[-1].endswith("?") else question_lst[-1]
        response = dico["response"]

        prompt = ["### Context\n"]
        for q, a in zip(question_lst[:-1], answer_lst):
            q = q + "?" if not q.endswith("?") else q
            prompt.append(f"{q} {a}")

        s = _get_str_prompt(args, "summarize_context", prompt)
        json.dump({"idx": idx, "sample_id": sample_id, "prompt": s}, fout_dict["summarize_context"])
        fout_dict["summarize_context"].write("\n")

        prompt += ["\n", "### Question", question]
        s = _get_str_prompt(args, "new_topic", prompt)
        json.dump({"idx": idx, "sample_id": sample_id, "prompt": s}, fout_dict["new_topic"])
        fout_dict["new_topic"].write("\n")

        s = _get_str_prompt(args, "single_response", prompt)
        json.dump({"idx": idx, "sample_id": sample_id, "prompt": s}, fout_dict["single_response"])
        fout_dict["single_response"].write("\n")

        s = _get_str_prompt(args, "search_query", prompt)
        json.dump({"idx": idx, "sample_id": sample_id, "prompt": s}, fout_dict["search_query"])
        fout_dict["search_query"].write("\n")

        prompt = prompt[:-3]
        prompt.append("\n")
        prompt.append("### Ambiguous Question")
        prompt.append(question)
        s = _get_str_prompt(args, "explain_question", prompt)
        json.dump({"idx": idx, "sample_id": sample_id, "prompt": s}, fout_dict["explain_question"])
        fout_dict["explain_question"].write("\n")

        # if response.count(" ") < 5:
        prompt = prompt[:-2]
        prompt.append("### Question and Answer to explain")
        prompt.append(f"{question} {response}")
        s = _get_str_prompt(args, "explain_response", prompt)
        json.dump({"idx": idx, "sample_id": sample_id, "prompt": s}, fout_dict["explain_response"])
        fout_dict["explain_response"].write("\n")


    [fout.close() for fout in fout_dict.values()]



def _generate_prompt_2(args):

    # load original history
    filename = os.path.join(args.tmp_dir, f"{args.ds_name}_{args.portion}_input.json")
    history_question_dict = defaultdict(list)
    history_response_dict = defaultdict(list)
    original_data_dict = {}
    for idx, line in enumerate(open(filename)):
        dico = json.loads(line.strip())
        original_data_dict[dico["sample_id"]] = dico
        parts = dico["sample_id"].split("_")
        key = "_".join(parts[:-1])
        q = dico["history"][-1] + "?" if not dico["history"][-1].endswith("?") else dico["history"][-1]
        history_question_dict[key].append((int(parts[-1]), q))
        history_response_dict[key].append((int(parts[-1]), dico["response"]))

    for key, lst in history_question_dict.items():
        lst.sort(key= lambda x:x[0])
        history_question_dict[key] = [item[1] for item in lst]

    for key, lst in history_response_dict.items():
        lst.sort(key= lambda x:x[0])
        history_response_dict[key] = [item[1] for item in lst]

    # combo prompt
    for prompt_name in ["explain_question", "single_response"]:
        filename = os.path.join(args.tmp_dir, f"{args.ds_name}_{args.portion}_{prompt_name}_{args.model_name}_output.json")
        for line in open(filename):
            dico = json.loads(line.strip())
            if not dico["text"]: continue
            parts = dico["sample_id"].split("_")
            key = "_".join(parts[:-1])
            idx = int(parts[-1])
            if idx == len(history_question_dict[key]): continue
            if prompt_name == "explain_question":
                history_question_dict[key][idx] = dico["text"]
            elif prompt_name == "single_response":
                history_response_dict[key][idx] = dico["text"]

    new_summary = {}
    prompt_name = "summarize_context"
    filename = os.path.join(args.tmp_dir, f"{args.ds_name}_{args.portion}_{prompt_name}_{args.model_name}_output.json")
    for line in open(filename):
        dico = json.loads(line.strip())
        new_summary[dico["sample_id"]] = dico["text"]

    new_history = {}
    counter = 0
    prompt_name_lst = ["single_response", "search_query", "explain_response",
                       "explain_question", "summarize_context", "new_topic"]
    extension_lst = ["explain", "summary"]
    fout_dict = {}

    for ext in extension_lst:
        for prompt_name in prompt_name_lst:
            filename = os.path.join(args.tmp_dir, f"{args.ds_name}_{args.portion}_{ext}-{prompt_name}_{args.model_name}_prompt.json")
            fout_dict[f"{ext}-{prompt_name}"] = open(filename, 'w')

    for key, qus_lst in history_question_dict.items():
        res_lst = history_response_dict[key]
        history = new_history.get(key, [])
        if not history: new_history[key] = []

        for i in range(len(qus_lst)):
            sample_id = f"{key}_{i+1}"

            for ext in extension_lst:
                prompt = ["### Context\n"] + new_history[key]
                if ext == "summary" and new_summary[sample_id]:
                    prompt = ["### Context\n"] + [new_summary[sample_id]]

                question = original_data_dict[sample_id]["history"][-1]
                question = question + "?" if not question.endswith("?") else question
                prompt += ["\n### Question", question]

                if ext != "summary":
                    s = _get_str_prompt(args, "summarize_context", prompt)
                    json.dump({"idx": counter, "sample_id": sample_id, "prompt": s}, fout_dict[ext+"-summarize_context"])
                    fout_dict[ext+"-summarize_context"].write("\n")

                s = _get_str_prompt(args, "new_topic", prompt)
                json.dump({"idx": counter, "sample_id": sample_id, "prompt": s}, fout_dict[ext+"-new_topic"])
                fout_dict[ext+"-new_topic"].write("\n")

                s = _get_str_prompt(args, "single_response", prompt)
                json.dump({"idx": counter, "sample_id": sample_id, "prompt": s}, fout_dict[ext+"-single_response"])
                fout_dict[ext+"-single_response"].write("\n")

                s = _get_str_prompt(args, "search_query", prompt)
                json.dump({"idx": counter, "sample_id": sample_id, "prompt": s}, fout_dict[ext+"-search_query"])
                fout_dict[ext+"-search_query"].write("\n")

                # explain_question
                prompt = prompt[:-2]
                prompt += ["\n### Ambiguous Question", question]
                s = _get_str_prompt(args, "explain_question", prompt)
                json.dump({"idx": counter, "sample_id": sample_id, "prompt": s}, fout_dict[ext+"-explain_question"])
                fout_dict[ext+"-explain_question"].write("\n")

            new_history[key].append(f"{qus_lst[i]} {res_lst[i]}")
            counter += 1

def _generate_prompt_keyword(args):
    extension_lst = ["", "explain-", "summary-"]
    prompt_name_lst = ["single_response", "search_query", "explain_response",
                       "explain_question", "summarize_context", "new_topic"]

    for ext_prefix in extension_lst:
        for prompt_name in  prompt_name_lst:
            filename = os.path.join(args.tmp_dir, f"{args.ds_name}_{args.portion}_{ext_prefix}{prompt_name}_{args.model_name}_output.json")
            if not os.path.exists(filename): continue
            file_output = os.path.join(args.tmp_dir, f"{args.ds_name}_{args.portion}_{ext_prefix}{prompt_name}_keyword_alias_{args.model_name}_prompt.json")
            fout = open(file_output, 'w')
            for idx, line in enumerate(open(filename)):
                dico = json.loads(line.strip())
                if not dico["text"]: continue
                s = _get_str_prompt(args, prompt_name+"_keyword_alias", [dico["text"]])
                json.dump({"idx": idx, "sample_id": dico["sample_id"], "prompt": s}, fout)
                fout.write("\n")

            fout.close()

def _clean_gen(prompt_name, text):
    if "[/INST]" in text:
        text = text.split("[/INST]")[1].strip()
    skip_lst = ["context", "given", "mentioned"]

    if prompt_name == "explain_question":
        if text[-1] == ")" and "? (" in text: text = text[:text.index("?")+1].strip()
        if text[-1] == "?" and ":\n" in text: text = text[text.rindex(":\n")+2:].strip()

        if text[-1] != "?" or "\n" in text: return ""
        if any(f" {w} " in text for w in skip_lst): return ""
        # if ", " in text: return ""# and text.count(" ") > 15:
        # if ", " in text: return ""
        #     print(text)
        #     return ""

        return text.strip()
    if prompt_name == "explain_response":
        if any(f" {w} " in text for w in skip_lst): return ""
        # print(text)
        # if text.count(". "): return ""
        # print(text)
        # print("\n==========\n")
        return text

    if prompt_name == "multi_response":
        text = re.sub('\n+', '\n', text)
        lst = []
        for line in text.split("\n"):
            if line[-1] == "?": continue
            for i in range(len(line)):
                if line[i].isalpha():
                    lst.append(line[i:])
                    break

        text = "\n".join(lst)

    if prompt_name == "new_topic":
        if "new_topic" in text: return "yes"
        if "old_topic" in text: return "no"
        return ""

    if prompt_name == "summarize_context":
        if "**Question:**" in text: return ""

    if "keyword_alias" in prompt_name:
        if "### Output:" in text: text = text.split("### Output:")[-1].strip()
        if "},\n]" in text: text = text.replace("},\n]", "}\n]")
        if "[" not in text or "]" not in text: return ""
        b, e = text.index("["), text.rindex("]")
        # print(text[b:e+1])
        try:
            lst = json.loads(text[b:e+1])
            for i, dico in enumerate(lst):
                assert isinstance(dico["keyword"], str) and isinstance(dico["alias"], str)
                dico = {"keyword": dico["keyword"], "alias": dico["alias"]}
                lst[i] = dico
            return lst
        except:
            return ""

    if "search_query" == prompt_name:
        if "{" not in text or "}" not in text: return ""
        b, e = text.index("{"), text.index("}")

        try:
            text = json.loads(text[b:e + 1])["query"]
            if isinstance(text, list):
                text = text[0]
            assert isinstance(text, str)
        except:
            return ""

    # if "single_response" == prompt_name:
    #     print(text)
    #     print("\n==========\n")


    return text

def _get_parse_combo_lst():
    portion_lst = ["train", "test"]
    prompt_name_lst = ["single_response", "search_query", "explain_response",
                       "explain_question", "summarize_context", "new_topic"]
    ext_prefix_lst = ["", "explain-", "summary-"]
    ext_suffix_lst = ["", "_keyword_alias"]

    combo_lst = []

    for ds_name in DS_NAME_LST:
        for portion in portion_lst:
            for ext_prefix in ext_prefix_lst:
                for ext_suffix  in ext_suffix_lst:
                    for prompt_name in prompt_name_lst:
                        key = (ds_name, portion, ext_prefix, prompt_name, ext_suffix)
                        combo_lst.append(key)

    return combo_lst

def _parse_llm_data(args):
    combo_lst = _get_parse_combo_lst()

    for ds_name, portion, ext_prefix, prompt_name, ext_suffix in combo_lst:
        pn = f"{ext_prefix}{prompt_name}{ext_suffix}"
        prompt_file = os.path.join(args.tmp_dir, f"{ds_name}_{portion}_{pn}_{args.model_name}_prompt.json")
        if not os.path.exists(prompt_file): continue
        idx2sample_map = {}
        for idx, line in enumerate(open(prompt_file)):
            dico = json.loads(line.strip())
            idx2sample_map[dico["idx"]] = dico["sample_id"]

        generate_file = os.path.join(args.tmp_dir, f"{ds_name}_{portion}_{pn}_{args.model_name}_generation.json")
        if not os.path.exists(generate_file): continue
        # print(generate_file)
        output_dict = {}
        failed_count = 0
        output_file = os.path.join(args.tmp_dir, f"{ds_name}_{portion}_{pn}_{args.model_name}_output.json")
        print(output_file)

        fout = open(output_file, 'w')
        for line in open(generate_file):
            dico = json.loads(line.strip())
            if dico["idx"] in output_dict: continue
            output_dict[dico["idx"]] =_clean_gen(prompt_name + ext_suffix, dico["text"])
            if not output_dict[dico["idx"]]:
                failed_count += 1
            # if prompt_name == "explain_question":
            #     print(output_dict[dico["idx"]])
            sample_id = idx2sample_map[dico["idx"]]
            json.dump({"sample_id": sample_id, "text": output_dict[dico["idx"]]}, fout)
            fout.write("\n")
        fout.close()
        print(pn, len(idx2sample_map), failed_count)

####################
### llm Methods  ###
####################

def _tokenize_prompt(args, dataset_dir, tokenizer):

    input_file = os.path.join(args.tmp_dir, f"{args.ds_name}_{args.portion}_{args.prompt_name}_{args.model_name}_prompt.json")
    idx_lst, text_lst = [], []
    for line in open(input_file):
        dico = json.loads(line.strip())
        idx_lst.append(dico["idx"])
        text_lst.append(dico["prompt"])

    inputs = tokenizer.batch_encode_plus(text_lst, add_special_tokens=False)

    meta_data_features = {
        'input_ids': Sequence(feature=Value(dtype='int32', id=None), length=-1, id=None),
        'idx': Sequence(feature=Value(dtype='int32', id=None), length=-1, id=None)
    }
    features = Features(meta_data_features)

    dico = {k:v for k, v in inputs.items() if "input_ids" == k}
    dico["idx"] = [[i] for i in idx_lst]

    iterative_dataset_save(args.tmp_dir, dataset_dir, dico, features)

def _load_model(args, model_name):

    if "llama" in model_name:
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path,
                                                 low_cpu_mem_usage=True,
                                                 torch_dtype=torch.bfloat16,
                                                 )

        model.config.pad_token_id = model.config.bos_token_id
    elif "Mistral" in model_name:
        model = MistralForCausalLM.from_pretrained(args.model_name_or_path,
                                                 low_cpu_mem_usage=True,
                                                 torch_dtype=torch.bfloat16
                                                 )

        model.config.pad_token_id = model.config.pad_token_id
    elif "gemma" in model_name:
        from transformers import GemmaForCausalLM
        model = GemmaForCausalLM.from_pretrained(args.model_name_or_path,
                                                   low_cpu_mem_usage=True,
                                                   torch_dtype=torch.bfloat16
                                                   )

        model.config.pad_token_id = model.config.pad_token_id
    else:
        raise ValueError("Model is not supported")
    return model

def _generate_llm_data(args):

    filename = os.path.join(args.tmp_dir, f"{args.ds_name}_{args.portion}_{args.prompt_name}_{args.model_name}_prompt.json")
    if not os.path.exists(filename): return
    wc_prompt = int(subprocess.check_output("wc -l %s" % filename, shell=True).split()[0])
    if not wc_prompt: return

    output_file = os.path.join(args.tmp_dir, f"{args.ds_name}_{args.portion}_{args.prompt_name}_{args.model_name}_generation.json")
    if os.path.exists(output_file):
        wc_generation = int(subprocess.check_output("wc -l %s" % output_file, shell=True).split()[0])
        print(wc_generation, wc_prompt)

        if wc_prompt <= wc_generation:
            print("File", filename, "is done")
            return


    import transformers
    from transformers import (
        AutoTokenizer,
        set_seed
    )
    from torch.utils.data.dataloader import DataLoader

    from accelerate import Accelerator
    args.use_slow_tokenizer = False
    args.seed = 1234
    if args.prompt_name in ["summarize_context", "multi_response"]:
        args.per_device_eval_batch_size = 1
        args.max_new_tokens = 768
    else:
        args.per_device_eval_batch_size = 2
        args.max_new_tokens = 512
    args.model_name_or_path = os.path.join(args.model_dir, args.model_name)

    model_name = os.path.split(args.model_name_or_path)[-1]

    accelerator = Accelerator(mixed_precision="fp16")
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)


    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              use_fast=not args.use_slow_tokenizer,
                                              padding_side="left")


    tokenizer.pad_token = tokenizer.bos_token


    # tokenize data
    dataset_dir = os.path.join(args.tmp_dir, f"dataset_{args.ds_name}_{args.portion}_{args.model_name}_{args.prompt_name}")


    if accelerator.is_local_main_process:
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
        _tokenize_prompt(args, dataset_dir, tokenizer)
    else:
        print("wait main process to tokenize data")
    accelerator.wait_for_everyone()
    eval_dataset = load_from_disk(dataset_dir)
    data_collator = LMGenDataCollator(tokenizer, model_name=args.model_name, prompt_name=args.prompt_name)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    model = _load_model(args, model_name)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    # Prepare everything with our `accelerator`.
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    model.eval()
    gen_kwargs = {
        "top_k": 50,
        "top_p": 0.9,
        # "prompt": "Write a bedtime story about neural networks I can read to my toddler",
        "temperature": 0.6,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": 1.15,
        # "prompt_template": "<s>[INST] {prompt} [/INST] ",
        # "presence_penalty": 0,
        # "frequency_penalty": 0

    }



    fout = open(output_file, 'w')

    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            idx_lst = accelerator.gather(batch["idx"]).cpu().numpy().tolist()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            if accelerator.is_local_main_process:

                new_decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                for idx, text in zip(idx_lst, new_decoded_preds):
                    json.dump({"idx": idx[0], "text": text}, fout)
                    fout.write("\n")

                if step % 10 == 0:
                    fout.close()
                    fout = open(output_file, 'a')

                # decoded_preds_all.extend([(idx[0], p) )
                # for text in new_decoded_preds:
                #     print(generated_tokens.shape)
                #     print("\n==================\n")
                #     print(text)
                #     print("\n==================\n")

def _load_default_queries(args, method= "default"):
    if method == "default":
        filename = f"{args.ds_name}_{args.portion}_{method}_query.json"
        default_query_dict = {}
        for line in open(filename):
            dico = json.loads(line)
            default_query_dict[dico["sample_id"]] = dico["text"]
    else:
        method = "responsethenreasoning"
        filename = os.path.join(args.tmp_dir, f"{args.ds_name}_{args.portion}_{method}_query.json")
        default_query_dict = json.load(open(filename))
        default_query_dict = {sample_id: lst[0] for sample_id, lst in default_query_dict.items()}

    return default_query_dict

def _load_keyword_alias_file(args, prompt_name, mode="alias"):
    filename = os.path.join(args.tmp_dir, f"{args.ds_name}_{args.portion}_{prompt_name}_keyword_alias_output.json")
    output_dict = {}
    if not os.path.exists(filename): return {}
    for line in open(filename):
        dico = json.loads(line.strip())
        keyword_str = "\t".join([item["keyword"] for item in dico["text"] if item["keyword"].strip()])
        alias_str = "\t".join([item["alias"] for item in dico["text"] if item["alias"].strip()])
        if mode == "keyword":
            output_dict[dico["sample_id"]] = keyword_str
        elif mode == "alias":
            output_dict[dico["sample_id"]] = alias_str
        elif mode == "keyword_alias":
            output_dict[dico["sample_id"]] = keyword_str + "\t\t" + alias_str
        else:
            raise ValueError(f"Unrecognized mode={mode}")

    return output_dict

def create_query_files(args, prefix_ext=""):

    # default_query
    default_query_dict = _load_default_queries(args)
    file_output = os.path.join(args.tmp_dir, f"{args.ds_name}_{args.portion}_default_{args.model_name}_query.json")  #
    json.dump(default_query_dict, open(file_output, 'w'))

    # explain_question
    for prompt_name in ["search_query", "explain_question"]:
        filename  = os.path.join(args.tmp_dir, f"{args.ds_name}_{args.portion}_{prefix_ext}{prompt_name}_{args.model_name}_output.json")
        query_dict = copy.deepcopy(default_query_dict)
        if not os.path.exists(filename): continue
        for line in open(filename):
            dico = json.loads(line.strip())
            if not dico["text"]: continue
            query_dict[dico["sample_id"]] = dico["text"]
        file_output = os.path.join(args.tmp_dir, f"{args.ds_name}_{args.portion}_{prefix_ext}{prompt_name}_{args.model_name}_query.json")
        json.dump(query_dict, open(file_output, 'w'))

     # explain_question
    for mode in ["keyword", "alias", "keyword_alias"]:
        keyword_alias_dict = _load_keyword_alias_file(args, prompt_name, mode=mode)
        if not keyword_alias_dict: continue
        tmp_query_dict = copy.deepcopy(query_dict)
        for sample_id, text in keyword_alias_dict.items():
            if mode == "alias":
                tmp_query_dict[sample_id] += f" {text}"
            else:
                tmp_query_dict[sample_id] = text
        file_output = os.path.join(args.tmp_dir, f"{args.ds_name}_{args.portion}_{prefix_ext}{prompt_name}_{mode}_{args.model_name}_query.json")
        json.dump(tmp_query_dict, open(file_output, 'w'))


def main():

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--mode",
                        default="parse_llm_data",
                        choices=["import_remote", "generate_prompt",
                                 "generate_llm_data", "parse_llm_data",
                                 "search_bm25"],
                        help="Choose which functionality to run")


    parser.add_argument(
        "--gpu_lst",
        default="0,1,2,3,5,6,7",
        type=str,
        # required=True,
        help="gpu_lst",
    )

    parser.add_argument("--model_name",
                        type=str,
                        # default="llama-2-7b-chat-hf",
                        # default="Mistral-7B-Instruct-v0.2",
                        # default="gemma-7b-it",
                        default="gpt-3.5-turbo",
                        help="Path to LLAMA model for tokenizer")

    # params for llm data generation
    parser.add_argument("--ds_name",
                        type=str,
                        default=DS_NAME_LST[0],
                        help="Path to LLAMA model for tokenizer")

    parser.add_argument("--portion",
                        type=str,
                        default="test",
                        help="Path to LLAMA model for tokenizer")

    parser.add_argument("--prompt_name",
                        type=str,
                        default="explain_question",
                        help="Path to LLAMA model for tokenizer")

    parser.add_argument(
        "--max_process", type=int, default=32, help="max_process"
    )

    args = parser.parse_args()
    args.model_dir = os.path.join(args.project_dir, "models")
    args.data_dir = os.path.join(args.project_dir, "data")
    args.tmp_dir = os.path.join(args.project_dir, "tmp")
    prompt_name_lst = ["single_response", "search_query", "explain_response",
                       "explain_question", "summarize_context", "new_topic"]
    # search_query, single_response, explain_response, explain_question, summarize_context
    extension_lst = ["explain", "summary"]


    if args.mode == "import_remote":
        import_remote_data(args)
    if args.mode == "generate_prompt":
        for ds_name in DS_NAME_LST[2:]:
            args.ds_name = ds_name
            _generate_prompt_1(args)
        # _generate_prompt_2(args)
        # _generate_prompt_keyword(args)
    if args.mode == "generate_llm_data":

        # for ds_name in DS_NAME_LST[-4:-3]:
        #     args.ds_name = ds_name
        for prompt_name in prompt_name_lst:
            args.prompt_name = prompt_name
            _generate_llm_data(args)

    if args.mode == "parse_llm_data":
        _parse_llm_data(args)
    if args.mode == "search_bm25":
        for prefix_ext in [""]+extension_lst:#
            if prefix_ext: prefix_ext += "-"
            create_query_files(args, prefix_ext=prefix_ext)

        evaluate(args)



    # print(prompt)
if __name__ == "__main__":
    main()
