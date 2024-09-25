import json
import argparse

def check_query(ori_query, default_query):
    # ori_query = ori_query.strip().replace(' ', '')
    if len(ori_query.strip().replace(' ', '')) == 0:
        return default_query
    return ori_query

def check_token(token, old_list, new_list):
    if token in old_list and token not in new_list:
        return True
    return False

def generate_combine_method_file(args):
    with open(args.default_file, "r") as f:
        data_default = f.readlines()

    with open(args.search_query, "r") as f:
        data_search_query = f.readlines()

    with open(args.explain_question, "r") as f:
        data_explain_question = f.readlines()

    with open(args.single_response, "r") as f:
        data_single_response = f.readlines()

    with open(args.summarize_context, "r") as f:
        data_summarize_context = f.readlines()

    with open(args.new_topic, "r") as f:
        data_new_topic = f.readlines()

    with open(args.explain_response, "r") as f:
        data_explain_response = f.readlines()

    

    method_map = {"ori-search_query": data_search_query, "ori-explain_question": data_explain_question, "ori-single_response": data_single_response,
                "ori-summarize_context": data_summarize_context, "ori-new_topic": data_new_topic, #"ori-explain_question_keywords": data_explain_question_keywords, 
                "ori-explain_response": data_explain_response, "default": data_default, 
                }

    cur = 0
    #with open(args.output_file_1, 'w') as f1, open(args.output_file_2, 'w') as f2, open(args.output_file_3, 'w') as f3, open(args.output_file_4, 'w') as f4, open(args.output_file_5, 'w') as f5:
    with open(args.output_file_1, 'w') as f1:
        for i in range(len(method_map["default"])):
            sample_id = json.loads(method_map["default"][i])["sample_id"]
            turn_id = int(sample_id.split('_')[-1])
            default_query = json.loads(method_map["default"][i])["text"]
            #search_query = json.loads(method_map["ori-search_query"][i])["oracle_utt_text"]

            search_query_id = json.loads(method_map["ori-search_query"][cur])["sample_id"]
            if search_query_id == sample_id:
                search_query = json.loads(method_map["ori-search_query"][cur])["oracle_utt_text"]
                #single_response = json.loads(method_map["ori-single_response"][cur])["ResponseThenReasoning_Response"]
                cur += 1
            else:
                search_query = default_query
                single_response = ""
            single_response = json.loads(method_map["ori-single_response"][i])["text"]
            explain_question = json.loads(method_map["ori-explain_question"][i])["text"]
            #explain_response = json.loads(method_map["ori-explain_response"][i])["text"]
            explain_response = ""
            new_topic = json.loads(method_map["ori-new_topic"][i])["text"].strip()
            summarize_context = json.loads(method_map["ori-summarize_context"][i])["text"]


            if turn_id == 1:
                summarize_context, explain_response = "", ""
            else:
                for idx in range(turn_id - 1):
                    #if new_topic != "yes":
                    explain_question += ' ' + json.loads(method_map["ori-explain_question"][i - idx - 1])["text"]
                    explain_response += ' ' + json.loads(method_map["ori-explain_response"][i - idx - 1])["text"]
            
            ori_session = list(set(default_query.strip().split(' ')))
            enhance_session = list(set((search_query + ' ' + single_response + ' ' + explain_question + ' ' + explain_response).strip().split(' ')))
            keywords =  ' '.join([word for word in ori_session if word in enhance_session])

            query_1 = search_query
            query_2 = explain_question
            query_3 = single_response 
            query_4 = explain_response
            query_5 = summarize_context
            
            query_1, query_2, query_3, query_4, query_5 = check_query(query_1, default_query), check_query(query_2, default_query), check_query(query_3, default_query), check_query(query_4, default_query), check_query(query_5, default_query)


            f1.write(json.dumps({"sample_id": sample_id, "query": query_1}) + '\n')
            f2.write(json.dumps({"sample_id": sample_id, "query": query_2}) + '\n')
            f3.write(json.dumps({"sample_id": sample_id, "query": query_3}) + '\n')
            f4.write(json.dumps({"sample_id": sample_id, "query": query_4}) + '\n')
            f5.write(json.dumps({"sample_id": sample_id, "query": query_5}) + '\n')



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_file", type = str)
    parser.add_argument("--default_file", type = str)
    parser.add_argument("--search_query", type = str)
    parser.add_argument("--explain_question", type = str)
    parser.add_argument("--explain_response", type = str)
    parser.add_argument("--single_response", type = str)
    parser.add_argument("--new_topic", type = str)
    parser.add_argument("--summarize_context")


    parser.add_argument("--extract", type = bool, default=False)
    parser.add_argument("--output_file_1", type = str)
    parser.add_argument("--output_file_2", type = str)
    parser.add_argument("--output_file_3", type = str)
    parser.add_argument("--output_file_4", type = str)
    parser.add_argument("--output_file_5", type = str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    generate_combine_method_file(args)

