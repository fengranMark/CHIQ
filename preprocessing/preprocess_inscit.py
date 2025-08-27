from json.tool import main
import json
from tqdm import tqdm, trange
import csv
import random
import os

def generate_collection(input_file_dir, output_file):
    files = os.listdir(input_file_dir)
    with open(output_file, 'w') as g:
        for file in files:
            input_file_path = os.path.join(input_file_dir, file)
            with open(input_file_path, 'r') as f:
                json_reader = json.load(f)
                # Move the data from json format to tsv
                for json_object in json_reader:
                    current_title = json_object['title'].strip()
                    # Parsing the text from each json object in the file
                    for pid, passage in enumerate(json_object['passages']):
                        new_row = ""
                        cur_id = passage['id']
                        new_row += cur_id + '\t'
                        psg_text = ' '.join(passage['titles']) + ' ' + ' '.join(passage['text'].split())
                        #psg_text += ' ' + ' '.join(passage['titles'])
                        new_row += psg_text
                        g.write(new_row + '\n')

def generate_qrel(input_file, output_file):
    cid = 0
    with open(input_file, 'r') as f, open(output_file, 'w') as g:
        json_reader = json.load(f)
        # Move the data from json format to tsv
        for dial_name in json_reader:
            dial = json_reader[dial_name]
            cid += 1
            for tid, turn in enumerate(dial['turns']):
                sample_id = "{}-{}_{}_{}".format("Inscit-Test", dial_name, str(cid), str(tid + 1))
                
                pos_ctxs = []
                added = set()
                for label in turn['labels']:
                    for e in label['evidence']:
                        pid = e['passage_id']
                        if pid in added:
                            continue
                        added.add(pid)
                        g.write("{}\t{}\t{}\t{}".format(sample_id, 0, pid, 1) + "\n")


def generate_test_file(input_file, output_file):
    cid = 0
    with open(input_file, 'r') as f, open(output_file, 'w') as g:
        json_reader = json.load(f)
        # Move the data from json format to tsv
        for dial_name in json_reader:
            dial = json_reader[dial_name]
            cid += 1
            for tid, turn in enumerate(dial['turns']):
                example = {}
                example['sample_id'] = "{}-{}_{}_{}".format("Inscit-Test", dial_name, str(cid), str(tid + 1))
                context = ' [SEP] '.join([' '.join(t.split()) for t in turn['context']])
                example['cur_utt_text'] = context.split(" [SEP] ")[-1]
                example["ctx_utts_text"] = context.split(" [SEP] ")[:-1]
                #example['cur_utt_text'] = ' [SEP] '.join([' '.join(t.split()) for t in turn['context']])
                example['gold_answer'] = [' '.join(l['response'].split()) for l in turn['labels']]
               # breakpoint()
                example["response_type"] = [' '.join(l['responseType'].split()) for l in turn['labels']]#' '.join(l['responseType'].split()) for l in turn['labels']

                pos_ctxs = []
                added = set()
                for label in turn['labels']:
                    example["pos_docs"], example["pos_docs_pids"] = [], []
                    for e in label['evidence']:
                        pos_ctx = {}
                        titles = [t[:-1] if t.endswith('.') else t for t in e['passage_titles']]
                        pos_ctx['title'] = ' [SEP] '.join(titles)
                        pos_ctx['text'] = e["passage_text"]
                        pos_ctx["score"] = 1000
                        pos_ctx["title_score"] = 1
                        pid = e['passage_id']
                        pos_ctx['passage_id'] = pid
                        if pid in added:
                            continue
                        added.add(pid)
                        pos_ctxs += [pos_ctx]
                        example["pos_docs"].append(pos_ctx['text'])
                        example["pos_docs_pids"].append(pos_ctx['passage_id'])
                #example["positive_ctxs"] = pos_ctxs
                #breakpoint()
                g.write(json.dumps(example) + '\n')

if __name__ == "__main__":
    input_file_dir = "./text_0420_processed"
    output_file = "collection.tsv"
    #generate_collection(input_file_dir, output_file)

    input_file = "./test.json"
    output_file = "./new_test.json"
    generate_test_file(input_file, output_file)

    input_file = "./test.json"
    output_file = "./inscit_qrel.trec"
    #generate_qrel(input_file, output_file)
