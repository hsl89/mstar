import json

def compute_indices(world_size, current_rank, dataset_length):
    num_test_samples_per_rank = int(dataset_length / world_size)
    if current_rank < world_size-1:
        indices = list(range(num_test_samples_per_rank*current_rank, num_test_samples_per_rank*(current_rank+1)))
    else:
        indices = list(range(num_test_samples_per_rank*current_rank, dataset_length))
    return indices

def save_json(data_dict, data_path):
    with open(data_path, 'w') as fp:
        json.dump(data_dict, fp)
        
def load_jsonl(gcs_path):
    with open(gcs_path, "rb") as f:
        datas = [json.loads(l) for l in f.readlines()]
    return datas

def load_humanfeedback_data(path):
    batch_ids = list(range(3,21)) 
    batch_ids.append(22)
    hf_data = []
    for bid in batch_ids:
        hf_batch_file = f"{path}/batch{bid}.json"
        hf_data.extend([json.loads(line) for line in open(hf_batch_file,'r')])
    return hf_data