import pytorch_lightning as pl
import transformers, datasets
from datasets import load_dataset
import torch
import torchmetrics
import numpy as np
import json
from torchmetrics.text.rouge import ROUGEScore
from pprint import pprint
import pandas as pd
from transformers import AutoModel, AutoTokenizer



def json_data_loader(data_path):
    with open(data_path, "rb") as f:
        legal_data = json.load(f)
    return legal_data

def tokenization(text, tokenizer, max_seq_len=512, truncation=True):
    text = "summarize: {}".format(text)
    text_encoding = tokenizer(text, 
                        max_length=max_seq_len,
                        padding="max_length",
                        truncation=truncation,
                        return_attention_mask=True,
                        add_special_tokens=True,
                        return_tensors="pt"
                    )
    return text_encoding


def legal(legal_data_path, filtered_legal_data_path, min_seq_len=128, max_seq_len=512):
    legal_data = json_data_loader(legal_data_path)
    tokenizer = AutoTokenizer.from_pretrained("t5-large")

    count =0
    fitered_dict = {}
    for k, v in legal_data.items():
        input_article = v['original_text']
        ref_summary = v['reference_summary']
        tokenized_input = tokenization(input_article, tokenizer, max_seq_len=max_seq_len)
        input_ids, attention_mask = tokenized_input['input_ids'], tokenized_input['attention_mask']
        input_seq_len = torch.sum(attention_mask)
        text_len = len(input_article.split(' '))
        # print(input_seq_len, len(input_article.split(' ')))
        if input_seq_len >= min_seq_len and input_seq_len <= max_seq_len:
            fitered_dict[k] = v
            count +=1
    print(f"Number of filtered samples: {count}")

    with open(filtered_legal_data_path, 'w') as fp:
        json.dump(fitered_dict, fp)

def edt_data(data_path, filtered_data_path, min_seq_len=128, max_seq_len=512):
    data = json_data_loader(data_path)
    tokenizer = AutoTokenizer.from_pretrained("t5-large")
    count =0
    fitered_dict = {}
    print(f"Number of orignal dataset size {len(data)}") #303893
    key_count = []
    for idx, v in enumerate(data):
        input_article = v['text']
        ref_summary = v['title']
        v['index_in_original_data'] = idx
        k = idx
        key_count.append(k)
        tokenized_input = tokenization(input_article, tokenizer, max_seq_len=max_seq_len, truncation=False)
        input_ids, attention_mask = tokenized_input['input_ids'], tokenized_input['attention_mask']
        input_seq_len = torch.sum(attention_mask)
        text_len = len(input_article.split(' '))
        # print(input_seq_len, len(input_article.split(' ')))
        
        if input_seq_len >= min_seq_len and input_seq_len <= max_seq_len:
            fitered_dict[k] = v
            count +=1
            if count >= 300: 
                print(f"looping data samples {idx}")
                break
    print(f"Number of filtered samples: {count}")
    
    # key_count = [v['pub_time'] for v in data]
    # print(f"Number of unique key {len(set(key_count))}") 
    # 120538; pub_time cannot be used as key to map sample
    with open(filtered_data_path, 'w') as fp:
        json.dump(fitered_dict, fp)
    

def privacy_data(data_path, filtered_data_path, min_seq_len=128, max_seq_len=512):
    data = pd.read_csv(data_path, header=0)
    tokenizer = AutoTokenizer.from_pretrained("t5-large")
    count =0
    fitered_dict = {}
    print(f"Number of orignal dataset size {len(data)}") #303893
    key_count = []
    input_seq_lens = []
    for idx, v in enumerate(data['QouteText']):
        if type(v) != str:
            continue
        input_article = v
        k = idx
        tokenized_input = tokenization(input_article, tokenizer, max_seq_len=max_seq_len, truncation=False)
        input_ids, attention_mask = tokenized_input['input_ids'], tokenized_input['attention_mask']
        input_seq_len = torch.sum(attention_mask)
        text_len = len(input_article.split(' '))
        input_seq_lens.append(input_seq_len)
        if input_seq_len >= min_seq_len and input_seq_len <= max_seq_len:
            fitered_dict[k] = {"text": input_article, "uid": k}
            count +=1
    
    print(f"Number of filtered samples: {count}")
    print(f'max seq len {max(input_seq_lens)}, min seq len {min(input_seq_lens)}, mean {np.mean(input_seq_lens)}')
    with open(filtered_data_path, 'w') as fp:
        json.dump(fitered_dict, fp)

def merge_policy_summaries(summary_path_1, summary_path_2, merged_data_path):
    summaries1 = json_data_loader(summary_path_1)
    summaries2 = json_data_loader(summary_path_2)
    fitered_dict = {}
    for k in summaries1.keys():
        v1 = summaries1[k]
        v2 = summaries2[k]
        summaries_list = v1["summaries"] + v2["summaries"]
        merged_list = []
        for item in summaries_list:
            if item.get("policy") != "reference_summary":
                merged_list.append(item)
        fitered_dict[k] = v1
        fitered_dict[k]["summaries"] = merged_list
    
    with open(merged_data_path, 'w') as fp:
        json.dump(fitered_dict, fp)


if __name__ == "__main__":
    path_prefix = "/hdd1/data/Summarization-of-Privacy-Policies/generated_summaries/"
    summary_path_1 = path_prefix + "TOSDR_full_content_au_labeled_v2_128_512.json"
    summary_path_2 = path_prefix + "TOSDR_full_content_au_labeled_v2_128_512_nbeams1_rep2.json"
    merged_data_path = path_prefix + "TOSDR_full_content_au_labeled_v2_128_512_merged.json"
    merge_policy_summaries(summary_path_1, summary_path_2, merged_data_path)
    
    data_path = "/hdd1/data/Summarization-of-Privacy-Policies/TOSDR_full_content_au_labeled_v2.csv"
    filtered_data_path = "/hdd1/data/Summarization-of-Privacy-Policies/TOSDR_full_content_au_labeled_v2_128_512.json"
    # privacy_data(data_path, filtered_data_path)
    
    data_path = "/hdd1/data/EDT_dataset/Trading_benchmark/evaluate_news.json"
    filtered_data_path = "/hdd1/data/EDT_dataset/Trading_benchmark/evaluate_news_128_512.json"
    # edt_data(data_path, filtered_data_path)
    
    legal_data_path = "/hdd1/data/legal_summarization/all_v1.json"
    filtered_legal_data_path = "/hdd1/data/legal_summarization/all_128_512.json"
    # legal(legal_data_path, filtered_legal_data_path)
    
    