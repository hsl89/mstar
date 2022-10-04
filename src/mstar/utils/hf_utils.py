"""Utility functions for using HuggingFace"""
import boto3
import botocore
import os
import shutil
from pathlib import Path
from filelock import FileLock
from filelock._error import Timeout
import logging
import hashlib
 

mstar_cache_home = os.path.expanduser(
    os.getenv("MSTAR_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "mstar"))
)

# pylint: disable=invalid-envvar-default
lock_hold_timeout_tokenizer = os.getenv("LOCK_ACQUIRE_TIMEOUT_TOKENIZER", 120)  
lock_hold_timeout_model = os.getenv("LOCK_ACQUIRE_TIMEOUT_MODEL", 600)
# pylint: enable=invalid-envvar-default

_default_endpoint = "mstar-models"
MSTAR_RESOLVE_ENDPOINT = os.environ.get("MSTAR_RESOLVE_ENDPOINT", _default_endpoint)

logger = logging.getLogger(__name__)


def _get_file_from_s3(key, file_names, revision="main", bucket_name=MSTAR_RESOLVE_ENDPOINT, force_download=False, cache_dir=None):    
    cache_dir = cache_dir or mstar_cache_home
    cache_dir = os.path.join(cache_dir,  "transformers")
    downloaded_folder = os.path.join(cache_dir, key, revision)
    files_to_download = []
    for file in file_names:
        files_to_download.append((os.path.join(downloaded_folder, file), f'models/{key}/{revision}/{file}'))

    path = Path(downloaded_folder)
    path.mkdir(parents=True, exist_ok=True)

    if all([os.path.exists(download_file) for download_file, _ in files_to_download] + [not force_download]):
        logger.warning("Load from cache %s", downloaded_folder)  
        return downloaded_folder
    s3 = boto3.resource('s3')
    s3_cli = boto3.client('s3')
    lock_path = downloaded_folder + ".lock"
    logger.warning("Using lock file %s", lock_path)
    lock = FileLock(lock_path, timeout=lock_hold_timeout_model)
    try:
        with lock:
            for download_file, download_key in files_to_download:
                if os.path.exists(download_file):
                    # Temporarily disable the check because of chunk_size is unknown for different models
                    # if os.path.isfile(download_file):
                    #     assert _validate_md5sum(download_file, s3_cli.head_object(Bucket=bucket_name, Key=download_key)), \
                    #         f"The checksum of local file {download_file} doesn't match" \
                    #         "remote object's ETag in S3 bucket. Validation failed!"
                    continue
                s3.Bucket(bucket_name).download_file(download_key, download_file)
        logger.warning("Load from s3 bucket %s", bucket_name) 
    except botocore.exceptions.ClientError as e:
        shutil.rmtree(downloaded_folder)
        if e.response['Error']['Code'] == "404":
            logger.error("The object does not exist.")
        else:
            raise
    except Timeout:
        print(f"Failed to acquire the lock within {lock_hold_timeout_model} seconds.", 
               "Please extend LOCK_ACQUIRE_TIMEOUT_MODEL and try again")
        shutil.rmtree(downloaded_folder)
        raise
    except FileExistsError as e:
        logger.warning("There is possible lock acquisition failure happening, please cleanup \
            the stale lock file %s first.", lock_path)
        shutil.rmtree(downloaded_folder)
        raise FileExistsError from e
    finally:
        lock.release()
        if os.path.exists(lock_path):
            os.remove(lock_path)

    return downloaded_folder


def get_model_file_from_s3(key, revision="main", bucket_name=MSTAR_RESOLVE_ENDPOINT, force_download=False, cache_dir=None):    
    downloaded_folder = _get_file_from_s3(key, ['pytorch_model.bin', 'config.json'], revision, bucket_name, force_download, cache_dir)
    return downloaded_folder


def get_config_file_from_s3(key, revision="main", bucket_name=MSTAR_RESOLVE_ENDPOINT, force_download=False, cache_dir=None):
    downloaded_folder = _get_file_from_s3(key, ['config.json'], revision, bucket_name, force_download, cache_dir)
    return downloaded_folder

def get_etag_checksum(file_name, chunk_size=8 * 1024 * 1024):
    if os.path.getsize(file_name) >= chunk_size:
        md5s = []
        with open(file_name, 'rb') as f:
            for data in iter(lambda: f.read(chunk_size), b''):
                md5s.append(hashlib.md5(data).digest())
        m = hashlib.md5(b"".join(md5s))
        return '{}-{}'.format(m.hexdigest(), len(md5s))
    else:
        with open(file_name, 'rb') as f:
            m = hashlib.md5(f.read())
        return '{}-{}'.format(m.hexdigest(), 1)


def get_md5sum(file_name):
    with open(file_name, 'rb') as file_to_check:
        # read contents of the file
        data = file_to_check.read()    
        # pipe contents of the file through
        md5_returned = hashlib.md5(data).hexdigest()
        return md5_returned


def _validate_md5sum(local_file, remote_file):
    if "ETag" in remote_file:
        etag = remote_file["ETag"].replace('"', '')
        if '-' in etag and etag == get_etag_checksum(local_file):
            return True
        if '-' not in etag and etag == get_md5sum(local_file):
            return True
    return False


def _check_folder_exist(bucket_name, remote_folder_name, local_folder, download=False):
    s3_resource = boto3.resource('s3')
    s3_cli = boto3.client('s3')
    bucket = s3_resource.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix = remote_folder_name):
        local_folder = local_folder.replace("tokenizers", "")
        obj_location = f'{local_folder}/{obj.key}'
        if os.path.exists(obj_location):
            if os.path.isfile(obj_location):
                # Temporarily disable the check because of chunk_size is unknown for different tokenizers
                # assert _validate_md5sum(obj_location, s3_cli.head_object(Bucket=bucket_name, Key=obj.key)), \
                #        f"The checksum of local file {obj_location} doesn't match" \
                #        "remote object's ETag in S3 bucket. Validation failed!"
                continue
        elif not download:
            return False
        if download:
            if not os.path.exists(os.path.dirname(obj_location)):
                os.makedirs(os.path.dirname(obj_location))
            if obj.key[-1] == '/':
                continue
            bucket.download_file(obj.key, obj_location)
    return True


def _download_directory_from_s3(bucket_name, remote_folder_name, local_folder, tokenizer_folder):
    if not os.path.exists(local_folder):
        os.makedirs(local_folder, exist_ok=True)
    else:
        # Check if the tokenizer exists in the local folder
        if _check_folder_exist(bucket_name, remote_folder_name, local_folder):
            return
    lock_path = local_folder + ".lock"
    logger.warning("Using lock file %s", lock_path)
    lock = FileLock(lock_path, timeout=lock_hold_timeout_tokenizer)
    try:
        with lock:
            _check_folder_exist(bucket_name, remote_folder_name, local_folder, download=True)
    except Timeout:
        print(f"Failed to acquire the lock within {lock_hold_timeout_tokenizer} seconds.", 
               "Please extend LOCK_ACQUIRE_TIMEOUT_TOKENIZER and try again")
        shutil.rmtree(tokenizer_folder)
        raise
    except FileExistsError as e:
        logger.warning("There is possible lock acquisition failure happening, please cleanup \
            the stale lock file %s first.", lock_path)
        shutil.rmtree(tokenizer_folder)
        raise FileExistsError from e
    finally:
        lock.release()
        if os.path.exists(lock_path):
            os.remove(lock_path)


def get_tokenizer_file_from_s3(key, revision="main", bucket_name=MSTAR_RESOLVE_ENDPOINT, cache_dir = None):
    cache_dir = cache_dir or mstar_cache_home
    download_folder = os.path.join(cache_dir,  "tokenizers") 
    bucket_name = _default_endpoint
    tokenizer_folder = os.path.join(download_folder, key, revision)
    _download_directory_from_s3(bucket_name, f"tokenizers/{key}/{revision}/", download_folder, tokenizer_folder)
    return tokenizer_folder
