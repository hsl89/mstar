import collections
import math
import os
import time
import boto3
import pytest
import moto

import mstar


@pytest.mark.parametrize('max_cpu,jitter,max_time,current_cpu,expected_max_wait', [
    (90, 0.5, 2, 95, 2.7),  # wait if busy
    (90, 10, 20, 5, 0.3),   # no wait if not busy
])
def test_wait_if_busy(mocker, max_cpu, jitter, max_time, current_cpu, expected_max_wait):
    mocker.patch('psutil.cpu_percent', return_value=current_cpu)

    @mstar.utils.misc.wait_if_busy(max_cpu, jitter, max_time)
    def func_call(*args, **kwargs):
        print(f'called with {args}, {kwargs}')

    tic = time.time()
    func_call(1, b=2)
    toc = time.time()

    assert toc-tic <= expected_max_wait


@pytest.mark.parametrize('env_vars', [
    {'test_var': 'test_val'}
])
def test_with_env(env_vars):
    @mstar.utils.misc.with_env(env_vars)
    def func_call():
        for k, v in env_vars.items():
            assert os.environ[k] == v

    for k in env_vars:
        assert k not in os.environ


@pytest.mark.parametrize('all_keys,prefix,pattern,expected', [
    ([], 'abc', 'a', set()),  # no result if not object in bucket
    (['aa', 'ab', 'ac', 'bc'], 'a', None, {'aa', 'ab', 'ac'}),  # prefix match
    (['aa', 'ab', 'ac', 'bc'], 'a', '*b', {'ab'})  # prefix and pattern match
])
@moto.mock_s3
def test_list_matched_s3_objects(mocker, all_keys,
                                 prefix, pattern, expected):
    bucket = 'test_bucket'
    conn = boto3.resource('s3', region_name='us-east-1')
    conn.create_bucket(Bucket=bucket)
    client = boto3.client('s3')
    for k in all_keys:
        client.put_object(Bucket=bucket, Key=k, Body='test content.')

    got = set(mstar.utils.misc.list_matched_s3_objects(bucket, prefix, pattern))
    assert got == expected


@pytest.mark.parametrize('num_objs,num_parts,num_expected', [
    (6, 2, 3),
    (6, 4, 3),
    (5, 3, 5),
    (3, 3, 1),
    (3, 1, 3),
    (100000, 4, 25000),
    (1000000, 250000, 4),
])
def test_generate_shards(num_objs, num_parts, num_expected):
    parts = mstar.utils.misc.generate_shards(num_objs, num_parts)
    assert [len(p) for p in parts] == [num_expected] * num_parts
    for i in range(len(parts[0])):
        shards = {p[i] for p in parts}
        assert len(shards) == len(parts)

    all_shards_count = collections.Counter(i for p in parts for i in p)
    assert list(sorted(all_shards_count.keys())) == list(range(num_objs))
    assert min(all_shards_count.values()) == max(all_shards_count.values())
