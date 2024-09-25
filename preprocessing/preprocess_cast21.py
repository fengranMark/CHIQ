import json
from tqdm import tqdm



def gen_test_file(raw_input_test_file, output_test_file):
    with open(raw_input_test_file, "r") as f:
        data = json.load(f)

    with open(output_test_file, "w") as f:
        for line in tqdm(data):
            ctx = []
            dialog_id = line['number']
            for turn in line['turn']:
                turn_id = turn['number']
                cur_utt_text = turn['raw_utterance']
                response = turn['passage']
                oracle_utt_text = turn['manual_rewritten_utterance']
                
                record = {"sample_id":"{}_{}".format(dialog_id, turn_id), 
                        "cur_utt_text": cur_utt_text,
                        "oracle_utt_text": oracle_utt_text,
                        "ctx_utts_text": ctx,
                        "canonical_response": [response]}
                
                f.write(json.dumps(record))
                f.write('\n')
                ctx.append(cur_utt_text)
                #ctx.append(turn['passage'])
    
    
    print("generate CAsT-21 test.json file ok")
                      


def reidx_qrel(raw_qrel_file, output_qrel_file, rawdocid2docid):
    with open(raw_qrel_file, "r") as fr, open(output_qrel_file, "w") as fw:
        for line in fr:
            line = line.strip().split('\t')
            rawdocid = line[2]
            line[2] = rawdocid2docid[rawdocid]
            line = "\t".join(line)
            fw.write(line + '\n')
        
    print("reidx CAsT-21 qrel ok!")

if __name__ == "__main__":
    raw_input_test_file = "cast21/2021_manual_evaluation_topics_v1.0.json"
    output_test_file = "cast21/eval_topics.json"
    raw_qrel_file = "cast21/2021qrels.txt"
    output_qrel_file = "cast21/qrels.tsv"
    gen_test_file(raw_input_test_file, output_test_file)
