import collections
import pickle
import boto3
import moto
import pytest
import lorem
import gzip
import datetime
import random
import json

import mstar


def generate_content(file_type):
    if file_type == 'txt':
        return '\n\n'.join([lorem.text() for _ in range(5)])
    elif file_type == 'jsonl':
        payload = []
        for _ in range(128):
            timestamp = datetime.datetime.now()
            timestamp -= datetime.timedelta(minutes=random.randrange(525600))
            payload.append(
                json.dumps({'text': lorem.text(), 'timestamp': str(timestamp), 'url': 'N/A'}))
        payload = '\n'.join(payload)
        return payload


@pytest.fixture(autouse=True)
def bucket_fixture():
    bucket = 'test_bucket'
    file_range = range(100)

    mocks3 = moto.mock_s3()
    mocks3.start()
    conn = boto3.resource('s3', region_name='us-east-1')
    conn.create_bucket(Bucket=bucket)
    client = boto3.client('s3')
    for f in file_range:
        content = generate_content('txt')
        client.put_object(Bucket=bucket, Key=f'test/part{f}.txt.gz',
                          Body=gzip.compress(content.encode()))
        content = generate_content('jsonl')
        client.put_object(Bucket=bucket, Key=f'test/part{f}.jsonl.gz',
                          Body=gzip.compress(content.encode()))
    yield
    mocks3.stop()


def map_mock(obj, fn, *iterables, **kwargs):
    """
    Mocking out the map function of multiprocessing because moto
    doesn't play well with multiprocessing
    """
    fn_obj = pickle.dumps(fn)
    fn = pickle.loads(fn_obj)
    res = []
    for i in iterables[0]:
        temp_res = fn(i)
        res.append(temp_res)
    return res


@pytest.mark.parametrize('infinite,eager', [
    (False, True),
    (False, False),
])
@pytest.mark.parametrize('shuffle', [True, False])
@pytest.mark.parametrize('keep_last_batch', [True, False])
@pytest.mark.parametrize('file_type', ['jsonl', 'txt'])
def test_DistributedMLM_local(mocker, file_type, infinite, eager, shuffle, keep_last_batch):
    mocker.patch('concurrent.futures.ProcessPoolExecutor.map', map_mock)
    mocker.patch('mstar.utils.executors.LazyProcessPoolExecutor.map', map_mock)

    cfg = mstar.utils.data.MLMS3DataConfig()
    cfg.bucket = 'test_bucket'
    cfg.eval_splits = 1
    cfg.prefix = 'test'
    cfg.pattern = f'*/part*.{file_type}.gz'
    cfg.file_type = file_type

    _, paths = mstar.utils.data.get_train_and_eval_paths(cfg)
    for i, _ in enumerate(mstar.utils.data.DistributedMLM(
            paths, cfg, infinite, eager, shuffle, keep_last_batch)):
        # loop through to make sure things work
        if infinite and i > 1000:
            break
        continue


@pytest.mark.parametrize('shuffle', [True, False])
@pytest.mark.parametrize('keep_last_batch', [True, False])
@pytest.mark.parametrize('world', [
    6, 7, 10
])
def test_DistributedMLM_sharded(mocker, shuffle, keep_last_batch, world):
    mocker.patch('concurrent.futures.ProcessPoolExecutor.map', map_mock)
    mocker.patch('mstar.utils.executors.LazyProcessPoolExecutor.map', map_mock)
    mocker.patch('torch.distributed.get_world_size', lambda: world)
    mocker.patch('torch.distributed.is_initialized', lambda: True)

    def dummy_broadcast(*args, **kwargs):
        pass
    mocker.patch('torch.distributed.broadcast_object_list', dummy_broadcast)
    mocker.patch('torch.cuda.device_count', lambda: 4)

    cfg = mstar.utils.data.MLMS3DataConfig()
    cfg.bucket = 'test_bucket'
    cfg.eval_splits = 0
    cfg.prefix = 'test'
    cfg.pattern = '*/part*.txt.gz'
    cfg.file_type = 'txt'

    paths, _ = mstar.utils.data.get_train_and_eval_paths(cfg)

    worker_shards = []
    for i in range(world):
        mocker.patch('torch.distributed.get_rank')
        dataset = mstar.utils.data.DistributedMLM(
            paths, cfg, False, False, shuffle, keep_last_batch)
        worker_shards.append(dataset.shard_paths)

    all_shards_count = collections.Counter(i for p in worker_shards for i in p)
    assert list(sorted(all_shards_count.keys())) == sorted(paths)
    assert min(all_shards_count.values()) == max(all_shards_count.values())


def test_get_train_and_eval_paths():
    s3_cfg = mstar.utils.data.S3DataConfig()
    s3_cfg.bucket = 'test_bucket'
    s3_cfg.eval_splits = 37
    s3_cfg.prefix = 'test'
    s3_cfg.pattern = '*/part*.txt.gz'

    train_paths, eval_paths = mstar.utils.data.get_train_and_eval_paths(s3_cfg)
    assert len(train_paths), len(eval_paths) == (63, 37)
