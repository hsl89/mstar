NOTE: GluonNLP uses `/dev/shm/gluonnlp` shared memory filesystem to share
datasets among multi-process workloads. At this time, `/dev/shm/gluonnlp` is not
cleaned up automatically after the workload completes and manual deletion is
needed to free up memory. Sometimes you may not want to delete
`/dev/shm/gluonnlp` after running a workload, as you intend to run a workload
based on same dataset later and it's useful to keep the dataset in shared
memory.

# BERT

-1. p4 instance preparation

```bash
sudo mkfs.btrfs /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1 /dev/nvme5n1 /dev/nvme6n1 /dev/nvme7n1 /dev/nvme8n1
sudo mount /dev/nvme1n1 /mnt
sudo chown ubuntu:ubuntu /mnt/
```

1. Get the dataset (with help of https://github.com/dmlc/gluon-nlp nlp_data tool)

```bash
nlp_data prepare_bookcorpus --segment_sentences --segment_num_worker 16
nlp_data prepare_wikipedia --mode download_prepared --segment_sentences --segment_num_worker 16
find wikicorpus/one_sentence_per_line BookCorpus/one_sentence_per_line -type f > input_reference
```


2. Phase 1 training with sequence length 128 (~20min + ~16 hours)

```bash
python3 prepare_quickthought.py \
    --input-reference input_reference \
    --output /mnt/out_quickthought_128 \
    --model-name google_en_uncased_bert_base \
    --max-seq-length 128 --dupe-factor 5
```

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 run_pretraining.py \
  --model_name google_en_uncased_bert_base \
  --lr 2e-4 \
  --log_interval 100 \
  --batch_size 256 \
  --num_accumulated 1 \
  --num_dataloader_workers 4 \
  --num_steps 225000 \
  --input-files /mnt/out_quickthought_128/*feather \
  --mmap-folder /mnt/mstar_mmap \
  --ckpt_dir /mnt/ckpt_dir \
  --ckpt_interval 1000 2>&1| tee train.log;
```

3. Phase 2 training with sequence length 512 (~20min + ~12 hours)

```bash
python3 prepare_quickthought.py \
    --input-reference input_reference \
    --output /mnt/out_quickthought_512 \
    --model-name google_en_uncased_bert_base \
    --max-seq-length 512 --dupe-factor 5
```

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 run_pretraining.py \
  --model_name google_en_uncased_bert_base \
  --lr 2e-4 \
  --log_interval 100 \
  --batch_size 32 \
  --num_accumulated 4 \
  --num_dataloader_workers 4 \
  --num_steps 25000 \
  --start_step 225000 \
  --phase2 \
  --phase1_num_steps 225000 \
  --input-files /mnt/out_quickthought_512/*feather \
  --mmap-folder /mnt/mstar_mmap \
  --ckpt_dir /mnt/ckpt_dir | tee train.log;
```

Finally we obtain a folder of structure as followed,

```
google_en_uncased_bert_base
├── vocab-{short_hash}.json
├── model-{short_hash}.params
├── model-{short_hash}.yml
```


# Sequence Tagging Data

To generate the label vocabulary, run this command in the outermost data directory. 
```bash
find ./ -name 'train.txt' -exec cat {} \; | sed "/^$/d" | awk '{print $NF}' | sort | uniq > label_vocab
```

To create the batches run the following command:
```bash
python3 prepare_seqtagging.py \
    --input-directory <path_to_input_directory> \
    --label-vocab <path_to_label_vocab> \
    --output-directory <path_to_output_directory> \
    --model-name google_en_uncased_bert_base \
    --max-seq-length 128 --dupe-factor 5
```