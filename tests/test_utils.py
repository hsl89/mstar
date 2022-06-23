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
def test_wait_if_busy(mocker, max_cpu, jitter, max_time,
                      current_cpu, expected_max_wait):
    mocker.patch('psutil.cpu_percent', return_value=current_cpu)

    @mstar.utils.misc.wait_if_busy(max_cpu, jitter, max_time)
    def func_call(*args, **kwargs):
        print(f'called with {args}, {kwargs}')

    tic = time.time()
    func_call(1, b=2)
    toc = time.time()

    assert toc - tic <= expected_max_wait


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

    got = set(
        mstar.utils.misc.list_matched_s3_objects(
            bucket, prefix, pattern))
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


def test_mstar_logger(mocker):
    client = mocker.patch('mstar.utils.lightning.MlflowClient')

    test_run = mocker.MagicMock()
    test_run.info.run_id = 'run-id-1'

    client().get_experiment_by_name = mocker.MagicMock(return_value=None)
    client().create_experiment = mocker.MagicMock(return_value='experiment-id')
    client().create_run = mocker.MagicMock(return_value=test_run)

    # Create new experiment with first run
    mstar_logger = mstar.utils.lightning.MStarEKSLogger(
        experiment_name='my-experiment', run_name='first-run')
    mstar_logger.get_client()
    assert mstar_logger.name == 'experiment-id'
    assert mstar_logger.run_id == 'run-id-1'

    test_run2 = mocker.MagicMock()
    test_run2.info.run_id = 'run-id-2'

    client().create_experiment = mocker.MagicMock(return_value='experiment-id')
    client().create_run = mocker.MagicMock(return_value=test_run2)

    # Create second run in this experiment
    mstar_logger2 = mstar.utils.lightning.MStarEKSLogger(
        experiment_name='my-experiment', run_name='second-run')
    mstar_logger2.get_client()
    assert mstar_logger2.name == 'experiment-id'
    assert mstar_logger2.run_id == 'run-id-2'

    # Test behavior of log_hyperparams
    mstar_logger2.log_hyperparams({"key": 1.0})
    mstar_logger2.experiment.log_param.assert_called_once()

    # Test behavior of log_metrics
    mstar_logger2.log_metrics({"loss": 0.1})
    mstar_logger2.experiment.log_metric.assert_called_once()

    # Test behavior of log_artifact
    mstar_logger2.log_artifact("/mnt/pytorch_lightning/bert.pt")
    mstar_logger2.experiment.log_artifact.assert_called_once()

    # Raise TypeError if metric value is string
    with pytest.raises(TypeError):
        mstar_logger2.log_metrics({"loss": 0.1, "loss2": "0.2"})


def test_flops_calculator():

    test_decoder_config = {
        'model_type': 'decoder',
        'activation_checkpointing': True,
        'vocab_size': 50257,
        'hidden_size': 2048,
        'decoder_num_layers': 24,
        'decoder_seq_len': 2048,
        'micro_batchsize': 16,
        'sec_per_step': 4.4,
    }

    decoder_tflops = 88.56

    decoder_with_checkpointing_tflops = mstar.utils.flops_calc.compute_tflops_per_gpu(
        **test_decoder_config)
    decoder_diff = math.sqrt(
        (decoder_with_checkpointing_tflops -
            decoder_tflops)**2)

    assert decoder_diff < 1e-1

    test_decoder_config['activation_checkpointing'] = False
    decoder_no_checkpointing_tflops = mstar.utils.flops_calc.compute_tflops_per_gpu(
        **test_decoder_config)
    # no checkpointing requires fewer operations
    assert decoder_no_checkpointing_tflops < decoder_with_checkpointing_tflops

    test_encoder_decoder_config = {
        'model_type': 'encoder_decoder',
        'activation_checkpointing': True,
        'vocab_size': 50257,
        'hidden_size': 2048,
        'decoder_num_layers': 12,
        'encoder_num_layers': 12,
        'encoder_seq_len': 2048,
        'decoder_seq_len': 2048,
        'micro_batchsize': 16,
        'sec_per_step': 4.4,
    }

    encoder_decoder_tflops = mstar.utils.flops_calc.compute_tflops_per_gpu(
        **test_encoder_decoder_config)

    # should exceed decoder-only with same num_layers,hidden size,seq_len
    # due to additional cross-attention
    assert encoder_decoder_tflops > decoder_tflops
