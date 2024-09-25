# ConvCHIQ
A code base for the submitted paper - CHIQ: Contextual History Enhancement for Improving Query Rewriting in Conversational Search.

# Environment Dependency

Main packages:
- python 3.8
- torch 1.8.1
- transformer 4.2.0
- numpy 1.22
- faiss-gpu 1.7.2
- pyserini 0.16

# Runing Steps

## 1. Download data and Preprocessing

Four public datasets can be downloaded from TREC Interactive Knowledge Assistance Track (iKAT)(https://www.trecikat.com/). The preprocessing for each dataset is under the "preprocess" folder.

## 2. Enhanced Conversational Context Generation

    python generation/enhancement_generation.py
    python generation/adhoc_query_rewriting.py

## 3. Search Originated Fine-tuning

### 3.1 Generate a Set of Query as Pseudo Supervision Signals

    python generation/generate_pseudo_query_{dataset}.py

### 3.2 Select the Optimal Query with Relevance Judgments

    python bm25/evaluate_{dataset}_generated_query.py

### 3.3 Search Originated LM Fine-tuning

    python train_QR.py --pretrained_query_encoder="checkpoints/Flant5-base" \ 
      --train_file_path=$train_file_path \ 
      --log_dir_path=$log_dir_path \
      --model_output_path=$model_output_path \ 
      --collate_fn_type="flat_concat_for_train" \ 
      --decode_type=$search_q \ 
      --per_gpu_train_batch_size=8 \ 
      --num_train_epochs=10 \
      --max_query_length=32 \
      --max_doc_length=384 \ 
      --max_response_length=32 \
      --max_concat_length=512 

### 3.4 Generate Search Query for Retrieval

    python test_QR.py --model_checkpoint_path=$model_checkpoint_path \
      --test_file_path=$test_file_path \
      --output_file_path=$output_file_path \
      --collate_fn_type="flat_concat_for_test" \ 
      --decode_type=$search_q \ 
      --per_gpu_eval_batch_size=32 \ 
      --max_query_length=32 \
      --max_doc_length=384 \ 
      --max_response_length=32 \
      --max_concat_length=512 \ 

## 4. Indexing
To evaluate the variants of our CHIQ, we should first establish indexes for both dense and sparse retrievers.

### 4.1 Dense
For dense retrieval, we use the pre-trained ad-hoc search model ANCE to generate passage embeddings. Two scripts for each dataset are provided by running:

    python indexing/dense/dense_index.py

### 4.2 Sparse

For sparse retrieval, we first run the format conversion script as:

    python indexing/bm25/convert_to_pyserini_format.py
    
Then create the index for the collection by running

    bash create_index.sh

## 5. Evaluation

To evaluate CHIQ-AD and CHIQ-FT, run the script with corresponding query files.

    python dense/test_{dataset}.py
    python bm25/bm25_{dataset}.py

To evaluate CHIQ-fusion, run:

    python dense/fusion.py
