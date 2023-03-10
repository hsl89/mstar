# Licensed to the GluonNLP team under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Helper functions."""

import os
import numpy as np


INT_TYPES = (int, np.int32, np.int64)
FLOAT_TYPES = (float, np.float16, np.float32, np.float64)


_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))


def use_einsum_optimization():
    """Whether to use einsum for attention. This will potentially accelerate the attention cell

    Returns
    -------
    flag
        The use einsum flag
    """
    flag = os.environ.get('MSTAR_USE_EINSUM', False)
    return flag


def get_home_dir():
    """Get home directory"""
    _home_dir = os.environ.get('MSTAR_HOME', os.path.join('~', '.mstar'))
    # expand ~ to actual path
    _home_dir = os.path.expanduser(_home_dir)
    return _home_dir


def get_data_home_dir():
    """Get home directory for storing the datasets"""
    home_dir = get_home_dir()
    return os.path.join(home_dir, 'datasets')


def get_model_zoo_home_dir():
    """Get the local directory for storing pretrained models"""
    home_dir = get_home_dir()
    return os.path.join(home_dir, 'models')


def get_model_zoo_checksum_dir():
    """Get the directory that stores the checksums of the artifacts in the model zoo """
    curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
    check_sum_dir = os.path.join(curr_dir, 'models', 'model_zoo_checksums')
    return check_sum_dir


def get_repo_url():
    """Return the base URL for Gluon dataset and model repository """
    default_repo = 's3://mstar-data'
    repo_url = os.environ.get('MSTAR_REPO', default_repo)
    if repo_url[-1] != '/':
        repo_url = repo_url + '/'
    return repo_url


def get_repo_model_zoo_url():
    """Return the base URL for Mstar Model Zoo"""
    repo_url = get_repo_url()
    model_zoo_url = repo_url + 'models/'
    return model_zoo_url


def get_third_party_dir():
    return os.path.realpath(os.path.join(_CURR_DIR, '..', '..', 'third_party'))
