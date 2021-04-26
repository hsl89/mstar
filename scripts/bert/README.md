# Data

p4 instance preparation

```bash
sudo mkfs.btrfs /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1 /dev/nvme5n1 /dev/nvme6n1 /dev/nvme7n1 /dev/nvme8n1
sudo mount /dev/nvme1n1 /mnt
sudo chown ubuntu:ubuntu /mnt/
```

## Masked Language Modeling (MLM) + QuickThought (QT)

Preprocess the subset of CC100 stored at `s3://mstar-data/cc100.txt.split.zst/`.
(See Appendix for instructions to reproduce the subset.)

```bash
# Follow the setup guide in ./batch/README.md first
aws --profile mstar batch submit-job --job-queue lausen-mstar-c5 --job-name prepare-cc100 --job-definition lausen-mstar-c5 --array-properties size=566 --container-overrides '{"command": ["/usr/local/bin/job.sh", "128"] }'
```

To create the batches for phase 2 pre-training, change 128 to 512.


## Sequence Tagging (NER)

```bash
set -k
# Get the data from S3
aws s3 sync s3://mstar-data/ner-utf8 ner-utf8

# Generate the label vocabulary
find ./ner-utf8 -name 'train.txt' -exec cat {} \; | sed "/^$/d" | awk '{print $NF}' | sort | uniq > label_vocab

# Create the batches for phase 1 pre-training
python3 prepare_seqtagging.py \
    --input-directory ./ner-utf8 \
    --label-vocab label_vocab \
    --output-directory ./ner_processed \
    --max-seq-length 128 --dupe-factor 50

# Copy the training files
mkdir ner_feather
find ner_processed -name 'train*feather' | xargs -I'{}' cp '{}' ner_feather   
```

To create the batches for phase 2 pre-training, change `--max-seq-length` to 512.


# Pre-training


```bash
python3 ./run_pretraining.py --trainer.gpus 8 --trainer.plugin ddp_sharded --trainer.max_steps 225000 --data.qt_dir /mnt/cc100_feather --data.ner_dir ner_feather --data.batch_size 128 --data.mmap_folder /mnt/mstar_mmap/ --trainer.replace_sampler_ddp=False --seed 1  --trainer.default_root_dir /mnt/default_root_dir --trainer.precision 16
```


# Appendix: Splitting the CC-100


```bash
set -k

# Obtain relevant lanugage splits from CC-100
aws s3 sync --exclude="*" --include en.txt.xz --include de.txt.xz --include fr.txt.xz --include ko.txt.xz --include ja.txt.xz --include es.txt.xz --include pt.txt.xz --include zh-Hans.txt.xz --include th.txt.xz --include it.txt.xz --include ar.txt.xz --include zh-Hant.txt.xz --include ca.txt.xz --include hi.txt.xz s3://mstar-data/cc100.txt.xz /mnt/cc100.txt.xz

# Unpack
cd /mnt/cc100.txt.xz
ls {en,de,fr,ko,ja,es,pt,zh-Hans,th,it,ar,zh-Hant,ca,hi}*.txt.xz | xargs -P0 -n1 xz -d

# Split files into smaller chunks, compress and store on S3
cat <<EOF >> langs
en
de
fr
ko
ja
es
pt
zh-Hans
th
it
ar
zh-Hant
ca
hi
EOF
cat langs | xargs -P0 -n1 -I'{}' split -l 10000000 -d -a3 '{}'.txt '{}'/'{}'.txt.
ls */*txt* | xargs -P$(nproc) -n1 zstd -T1 -1
aws s3 sync --exclude="*" --include="*zst" /mnt/cc100 s3://mstar-data/cc100.txt.split.zst
```
