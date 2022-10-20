import argparse
import fnmatch
import functools
import hashlib
import inspect
import itertools
import logging
import math
import os
import psutil
import random
import sys
import time
import uuid
import warnings
from typing import Optional

import boto3
from botocore.handlers import disable_signing
import requests
import tqdm

S3_PREFIX = 's3://'

if not sys.platform.startswith('win32'):
    # refer to https://github.com/untitaker/python-atomicwrites
    def replace_file(src, dst):
        """Implement atomic os.replace with linux and OSX.
        Parameters
        ----------
        src : source file path
        dst : destination file path
        """
        try:
            os.rename(src, dst)
        except OSError as os_err:
            try:
                os.remove(src)
            except OSError:
                pass
            finally:
                raise OSError from os_err
else:
    import ctypes

    _MOVEFILE_REPLACE_EXISTING = 0x1
    # Setting this value guarantees that a move performed as a copy
    # and delete operation is flushed to disk before the function returns.
    # The flush occurs at the end of the copy operation.
    _MOVEFILE_WRITE_THROUGH = 0x8
    _windows_default_flags = _MOVEFILE_WRITE_THROUGH

    def _str_to_unicode(x):
        """Handle text decoding. Internal use only"""
        if not isinstance(x, str):
            return x.decode(sys.getfilesystemencoding())
        return x

    def _handle_errors(rv, src):
        """Handle WinError. Internal use only"""
        if not rv:
            msg = ctypes.FormatError(ctypes.GetLastError())
            # if the MoveFileExW fails(e.g. fail to acquire file lock), removes the tempfile
            try:
                os.remove(src)
            except OSError:
                pass
            finally:
                raise OSError(msg)

    def replace_file(src, dst):
        """Implement atomic os.replace with windows.
        refer to https://docs.microsoft.com/en-us/windows/desktop/api/winbase/nf-winbase-movefileexw
        The function fails when one of the process(copy, flush, delete) fails.
        Parameters
        ----------
        src : source file path
        dst : destination file path
        """
        _handle_errors(
            ctypes.windll.kernel32.MoveFileExW(_str_to_unicode(src), _str_to_unicode(dst),
                                               _windows_default_flags | _MOVEFILE_REPLACE_EXISTING),
            src)


def sha1sum(filename):
    """Calculate the sha1sum of a file
    Parameters
    ----------
    filename
        Name of the file
    Returns
    -------
    ret
        The sha1sum
    """
    with open(filename, mode='rb') as f:
        d = hashlib.sha1()
        for buf in iter(functools.partial(f.read, 1024 * 100), b''):
            d.update(buf)
    return d.hexdigest()


def download(url: str,  # noqa: MC0001
             path: Optional[str] = None,
             overwrite: Optional[bool] = False,
             sha1_hash: Optional[str] = None,
             retries: Optional[int] = 5,
             verify_ssl: Optional[bool] = True) -> str:
    # pylint: disable=too-many-arguments, too-many-locals, too-many-statements
    """Download a given URL

    Parameters
    ----------
    url
        URL to download
    path
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite
        Whether to overwrite destination file if already exists.
    sha1_hash
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    retries
        The number of times to attempt the download in case of failure or non 200 return codes
    verify_ssl
        Verify SSL certificates.
    Returns
    -------
    fname
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
        # Empty filenames are invalid
        assert fname, 'Can\'t construct file-name from this URL. ' \
            'Please set the `path` option manually.'
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path

    if not verify_ssl:
        warnings.warn('Unverified HTTPS request is being made (verify_ssl=False). '
                      'Adding certificate verification is strongly advised.')

    assert retries >= 0, f"Number of retries should be at least 0, currently it's {retries}"
    if overwrite or not os.path.exists(fname) or (sha1_hash and not sha1sum(fname) == sha1_hash):
        is_s3 = url.startswith(S3_PREFIX)
        if is_s3:
            s3 = boto3.resource('s3')
            if boto3.session.Session().get_credentials() is None:
                s3.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
            components = url[len(S3_PREFIX):].split('/')
            if len(components) < 2:
                raise ValueError('Invalid S3 url. Received url={}'.format(url))
            s3_bucket_name = components[0]
            s3_key = '/'.join(components[1:])

        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        while retries + 1 > 0:
            # pylint: disable=broad-except
            try:
                print('Downloading {} from {}...'.format(fname, url))
                if is_s3:
                    response = s3.meta.client.head_object(Bucket=s3_bucket_name, Key=s3_key)
                    total_size = int(response.get('ContentLength', 0))
                    random_uuid = str(uuid.uuid4())
                    tmp_path = '{}.{}'.format(fname, random_uuid)
                    if tqdm is not None:

                        def hook(t_obj):
                            def inner(bytes_amount):
                                t_obj.update(bytes_amount)

                            return inner

                        with tqdm.tqdm(total=total_size, unit='iB', unit_scale=True) as t:
                            s3.meta.client.download_file(s3_bucket_name, s3_key, tmp_path,
                                                         Callback=hook(t))
                    else:
                        s3.meta.client.download_file(s3_bucket_name, s3_key, tmp_path)
                else:
                    r = requests.get(url, stream=True, verify=verify_ssl)  # pylint: disable=missing-timeout
                    if r.status_code != 200:
                        raise RuntimeError('Failed downloading url {}'.format(url))
                    # create uuid for temporary files
                    random_uuid = str(uuid.uuid4())
                    total_size = int(r.headers.get('content-length', 0))
                    chunk_size = 1024
                    if tqdm is not None:
                        t = tqdm.tqdm(total=total_size, unit='iB', unit_scale=True)
                    with open('{}.{}'.format(fname, random_uuid), 'wb') as f:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:  # filter out keep-alive new chunks
                                if tqdm is not None:
                                    t.update(len(chunk))
                                f.write(chunk)
                    if tqdm is not None:
                        t.close()
                # if the target file exists(created by other processes)
                # and have the same hash with target file
                # delete the temporary file
                if not os.path.exists(fname) or (sha1_hash and not sha1sum(fname) == sha1_hash):
                    # atomic operation in the same file system
                    replace_file('{}.{}'.format(fname, random_uuid), fname)
                else:
                    try:
                        os.remove('{}.{}'.format(fname, random_uuid))
                    except OSError:
                        pass
                    finally:
                        warnings.warn(
                            'File %s exists in file system so the downloaded file is deleted',
                            fname)
                if sha1_hash and not sha1sum(fname) == sha1_hash:
                    raise UserWarning(
                        'File {} is downloaded but the content hash does not match.'
                        ' The repo may be outdated or download may be incomplete. '
                        'If the "repo_url" is overridden, consider switching to '
                        'the default repo.'.format(fname))
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e

                print('download failed due to {}, retrying, {} attempt{} left'
                      .format(repr(e), retries, 's' if retries > 1 else ''))

    return fname


def sort_multipart_file_path(path_l, prefix=None):
    """Sort the multipart file path

    Parameters
    ----------
    path_l
    prefix

    Returns
    -------
    sorted_path_l
    """
    assert isinstance(path_l, (list, tuple))
    if prefix is not None:
        path_l = [ele for ele in path_l if ele.startswith(prefix)]
    if len(path_l) <= 1:
        return path_l
    prefix = os.path.commonprefix(path_l)
    postfix = os.path.commonprefix([ele[::-1] for ele in path_l])[::-1]
    part_id_l = [ele[len(prefix):-len(postfix)] for ele in path_l]
    part_id_int_l = [int(ele) for ele in part_id_l]
    assert len(set(part_id_l)) == len(part_id_l)
    sorted_path_l = sorted(zip(path_l, part_id_int_l), key=lambda ele: ele[1])
    sorted_path_l = [ele[0] for ele in sorted_path_l]
    return sorted_path_l


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def repeat(iterable, count=None, *, set_epoch=False):
    if count is None:
        i = 0
        while True:
            if set_epoch:
                iterable.sampler.set_epoch(i)
            for sample in iterable:
                yield sample
            i += 1
    else:
        for i in range(count):
            if set_epoch:
                iterable.sampler.set_epoch(i)
            for sample in iterable:
                yield sample


# pylint: disable=too-many-arguments, too-many-locals
def logging_config(folder: Optional[str] = None, name: Optional[str] = None,
                   logger: logging.Logger = logging.root, level: int = logging.INFO,
                   console_level: int = logging.INFO, console: bool = True,
                   local_rank: Optional[int] = None, num_workers: Optional[int] = None,
                   overwrite_handler: bool = False) -> str:
    """Config the logging module. It will set the logger to save to the specified file path.
    Parameters
    ----------
    folder
        The folder to save the log
    name
        Name of the saved
    logger
        The logger
    level
        Logging level
    console_level
        Logging level of the console log
    local_rank
        The local rank
    num_workers
        The number of workers
    console
        Whether to also log to console
    overwrite_handler
        Whether to overwrite the existing handlers in the logger

    Returns
    -------
    folder
        The folder to save the log file.
    """
    if name is None:
        name = inspect.stack()[-1][1].split('.')[0]
    if folder is None:
        folder = os.path.join(os.getcwd(), name)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    need_file_handler = True
    need_console_handler = True
    # Check all loggers.
    if overwrite_handler:
        logger.handlers = []
    else:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                need_console_handler = False
    logpath = os.path.join(folder, name + ".log")
    print("All Logs will be saved to {}".format(logpath))
    logger.setLevel(level)
    if local_rank is None:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        formatter = logging.Formatter(f'[{local_rank}/{num_workers}] %(asctime)s - %(name)s'
                                      f' - %(levelname)s - %(message)s')
    if need_file_handler:
        logfile = logging.FileHandler(logpath)
        logfile.setLevel(level)
        logfile.setFormatter(formatter)
        logger.addHandler(logfile)
    if console and need_console_handler:
        # Initialze the console logging
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logger.addHandler(logconsole)
    return folder


def naming_convention(file_dir, file_name):
    """Rename files with 8-character hash"""
    long_hash = sha1sum(os.path.join(file_dir, file_name))
    file_prefix, file_sufix = file_name.split('.')
    new_name = '{file_prefix}-{short_hash}.{file_sufix}'.format(file_prefix=file_prefix,
                                                                short_hash=long_hash[:8],
                                                                file_sufix=file_sufix)
    return new_name, long_hash


# Python 3.9 feature backport https://github.com/python/cpython/pull/11478
class BooleanOptionalAction(argparse.Action):
    def __init__(self, option_strings, dest, default=None, type=None, choices=None, required=False,
                 help=None, metavar=None):
        # pylint: disable=redefined-builtin

        _option_strings = []
        for option_string in option_strings:
            _option_strings.append(option_string)

            if option_string.startswith('--'):
                option_string = '--no-' + option_string[2:]
                _option_strings.append(option_string)

        if help is not None and default is not None:
            help += f" (default: {default})"

        super().__init__(option_strings=_option_strings, dest=dest, nargs=0, default=default,
                         type=type, choices=choices, required=required, help=help, metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self.option_strings:
            setattr(namespace, self.dest, not option_string.startswith('--no-'))

    def format_usage(self):
        return ' | '.join(self.option_strings)


def wait_if_busy_fn(max_cpu, jitter, max_time, fn, *args, **kwargs):
    """Higher-order function for waiting up to a max time if cpu utilization exceeds set limit
    before executing the function.

    Parameters
    ----------
    max_cpu : int or float
        Utilization percentage limit that triggers wait.
    jitter : int or float
        The maximum random sleep time in seconds when waiting for the CPU to become less busy.
    max_time : int or float or None
        The maximum wait time in seconds this decorator allows. If None, wait indefinitely.
    """
    assert 0 <= max_cpu <= 100, f'max_cpu must be between 0 and 100, got: {max_cpu}'
    assert jitter > 0, f'jitter must be greater than 0, got: {jitter}'
    assert max_time is None or max_time > 0, \
        f'max_time, if set, must be greater than 0, got: {max_time}'
    time_waited = 0

    # Sleep if high CPU util
    while not max_time or time_waited < max_time:
        if psutil.cpu_percent() > max_cpu:
            to_wait = random.uniform(0, jitter)
            time_waited += to_wait
            time.sleep(to_wait)
        else:
            break
    return fn(*args, **kwargs)


def wait_if_busy(max_cpu, jitter, max_time=None):
    """Higher-order function for waiting up to a max time if cpu utilization exceeds set limit
    before executing the function.

    Parameters
    ----------
    max_cpu : int or float
        Utilization percentage limit that triggers wait.
    jitter : int or float
        The maximum random sleep time in seconds when waiting for the CPU to become less busy.
    max_time : int or float or None
        The maximum wait time in seconds this decorator allows.
    """
    assert 0 <= max_cpu <= 100, f'max_cpu must be between 0 and 100, got: {max_cpu}'
    assert jitter > 0, f'jitter must be greater than 0, got: {jitter}'
    assert max_time is None or max_time > 0, \
        f'max_time, if set, must be greater than 0, got: {max_time}'

    def decorator(func):
        return functools.wraps(func)(functools.partial(wait_if_busy_fn, max_cpu, jitter, max_time,
                                                       func))

    return decorator


def with_env_fn(env_dict, fn, *args, **kwargs):
    """Higher-order function for Override environment variables when executing."""

    assert all(isinstance(v, str) for v in env_dict.values()), \
        'All environment variables should be passed as strings.'

    before = {k: os.environ[k] for k in env_dict if k in os.environ}
    os.environ.update(env_dict)
    out = fn(*args, **kwargs)
    for k in env_dict:
        del os.environ[k]
    os.environ.update(before)
    return out


def with_env(env_dict):
    """Decorator for overriding environment variables when executing."""

    assert all(isinstance(v, str) for v in env_dict.values()), \
        'All environment variables should be passed as strings.'

    def decorator(func):
        return functools.wraps(func)(functools.partial(with_env_fn, env_dict, func))

    return decorator


def list_matched_s3_objects(bucket, prefix, pattern=None):
    """List objects in the specified bucket with the provided prefix and pattern.

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    prefix : str
        Desired object prefix.
    pattern : str or None
        Object key pattern to match, if provided.
    """

    client = boto3.client('s3')
    object_list = client.list_objects(Bucket=bucket, Prefix=prefix)
    object_keys = [obj['Key'] for obj in object_list.get('Contents', [])]
    if pattern:
        matched_keys = sorted(k for k in object_keys if fnmatch.fnmatch(k, pattern))
    else:
        matched_keys = sorted(object_keys)

    return matched_keys


def generate_shards(num_objs, num_partitions):
    """Partitions a list of objects across a specified number of partitions,
    so that each partition has the same number of shards, all items appear
    the same number of times, and no two partition will have the same shard
    at each index.

    Parameters
    ----------
    num_objs : int
        Number of objects to shard.
    num_partitions : int
        Number of total partitions.

    Returns
    -------
    list of lists
    """
    assert num_partitions <= num_objs
    n, p = num_objs, num_partitions
    total_shards = n * p // math.gcd(n, p)  # lcm(n, p)
    repeats = total_shards // n

    perm_parts = {}
    perm_indices = [i // 2 if i % 2 == 0 else (n + i + 1) // 2 for i in range(repeats)]
    perm_indices_set = set(perm_indices)
    largest = max(perm_indices_set)
    for i, perm in enumerate(itertools.permutations(range(n))):
        if i in perm_indices_set:
            perm_parts[i] = perm
            if i == largest:
                break
        else:
            continue

    perm_parts = [k for p in perm_indices for k in perm_parts[p]]

    parts = []
    for i in range(num_partitions):
        parts.append(perm_parts[i::p])
    return parts
