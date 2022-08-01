# Goal

This repository assumes that you want to process the pile (or a subset) and produce a `.arrow` file, likely for pretraining. The scripts in this repository will produce a `.arrow` file that contains text data. It is STRONGLY RECOMMENDED to do everything from `us-east-1`. This is where the s3 bucket all scripts rely on right now. Since the data is in `mstar-data` this is also replicated to other regions, but you will need to change the reading paths.  

The pile is described here: https://quip-amazon.com/elDGAoLZlZXb/Pile-Subsets

# Setup

This requires a large instance because we put the entire dataset into memory before writing. Attach to an instance with lots of RAM. This script has only been tested on `u-9tb1.112xlarge` with 9TB RAM.


This repository provides two main capabilities. 

(1) Producing the `.arrow` file from `.jsonl.zst` files in s3.
(2) Filtering out unwanted subsets of the pile.

You will need install the requirements to run the scripts. These are installed through
```
conda create -n python=3.9 #clean conda environment
cd ../../ #same directory as mstar/setup.py
pip install -U .[pile] #install mstar with additional requirements from pile_require in setup.py
```

# Basic Usage and Filtering
## Mount an EBS Volume 

A 5TB mounted drive is large enough to store the files. We will stream in the data (see `pile_data_preprocessing.py`) and then write one large arrow file.

See [this linked page](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html) for setting up the EBS volumne and mounting.

The bash scripts in this folder assume that you have mounted the drive to `/mnt/pile_mnt`. For ease-of-use I recommend the following mount command:
```
mkdir /mnt/pile_mnt
sudo mount /dev/YOUR_EBS_NAME /mnt/pile_mnt
```

You can use `lsblk` to find your EBS device name after you have attached it to your instance.

Your user `ubuntu` may not have access to the /mnt fodler. You can use
```
sudo chmod 777 /mnt
```
to make this directory accessible to all users.


## Create Arrow File

Now you will filter the pile to create a `.arrow` dataset that contains only the subsets you want. List the files you want to EXCLUDE in `configs/subsets_to_exclude.txt`. By default `YoutubeSubtitles` is excluded. A full list of subset names is in `pile_subset_names.txt`. To process all the training data (approx 4hrs for the full pile) run
```
bash scripts/process_pretrain_data.sh 
``` 
This will create an arrow file on your mounted drive. You can unmount and shutdown the instance.

# Upload

You can unmount the drive, remount to a cheaper instance, and do the upload from a cheaper instance.
