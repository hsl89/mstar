"""Utility functions for using HuggingFace"""
import boto3
import botocore
import os
from pathlib import Path
import logging
import hashlib
 

mstar_cache_home = os.path.expanduser(
    os.getenv("MSTAR_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "mstar"))
)

_default_endpoint = "mstar-models"
MSTAR_RESOLVE_ENDPOINT = os.environ.get("MSTAR_RESOLVE_ENDPOINT", _default_endpoint)

logger = logging.getLogger(__name__)

def get_model_file_from_s3(key, revision="main", bucket_name=MSTAR_RESOLVE_ENDPOINT, force_download=False, cache_dir=None):    
    cache_dir = cache_dir or mstar_cache_home
    cache_dir = os.path.join(cache_dir,  "transformers")
    downloaded_folder = os.path.join(cache_dir, key, revision)
    downloaded_model = os.path.join(downloaded_folder, 'pytorch_model.bin')
    downloaded_config = os.path.join(downloaded_folder, 'config.json')

    model_key = f'models/{key}/{revision}/pytorch_model.bin'
    config_key = f'models/{key}/{revision}/config.json'
    path = Path(downloaded_folder)
    path.mkdir(parents=True, exist_ok=True)
    if all([os.path.exists(downloaded_model), os.path.exists(downloaded_config), not force_download]):
        logger.warning("Load from cache %s", downloaded_folder)  
        return downloaded_folder
    s3 = boto3.resource('s3')
    try:
        s3.Bucket(bucket_name).download_file(model_key, downloaded_model)
        s3.Bucket(bucket_name).download_file(config_key, downloaded_config)
        logger.warning("Load from s3 bucket %s", bucket_name) 
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            logger.error("The object does not exist.")
        else:
            raise
    return downloaded_folder


def download_directory_from_s3(bucket_name, remote_folder_name, local_folder):
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucket_name) 
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)
    for obj in bucket.objects.filter(Prefix = remote_folder_name):
        local_folder = local_folder.replace("tokenizers", "")
        obj_location = f'{local_folder}/{obj.key}'
        if not os.path.exists(os.path.dirname(obj_location)):
            os.makedirs(os.path.dirname(obj_location))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, obj_location)


def get_tokenizer_file_from_s3(key, revision="main", bucket_name=MSTAR_RESOLVE_ENDPOINT, cache_dir = None):
    cache_dir = cache_dir or mstar_cache_home
    download_folder = os.path.join(cache_dir,  "tokenizers") 
    bucket_name = _default_endpoint
    tokenizer_folder = os.path.join(download_folder, key, revision)
    download_directory_from_s3(bucket_name, f"tokenizers/{key}/{revision}/", download_folder)
    return tokenizer_folder
    

def get_md5sum(file_name):
    with open(file_name, 'rb') as file_to_check:
        # read contents of the file
        data = file_to_check.read()    
        # pipe contents of the file through
        md5_returned = hashlib.md5(data).hexdigest()
        return md5_returned