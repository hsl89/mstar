#!/usr/bin/env python
import io
import os
import re
from datetime import datetime
import warnings

import torch
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version("src", "mstar", "__init__.py")

if VERSION.endswith("dev"):
    VERSION = VERSION + datetime.today().strftime("%Y%m%d")

requirements = [
    "boto3",
    "s3fs",
    "smart_open",
    "h5py>=2.10.0",
    "yacs>=0.1.8",
    "sentencepiece",
    "tqdm",
    "regex",
    "requests",
    "pyarrow>=3",
    "transformers@git+https://github.com/barry-jin/transformers@8bf3a601a764c4d74e6488be8bc7a9efacaf1ddc",  # transformers that support balanced strategies.
    "tokenizers>=0.10.2,<=0.12.1",  # 0.10.1 is buggy; Set upper bound to avoid breaking changes in tokenizers.
    "accelerate>=0.9.0,<=0.12.0",
    "tensorboard",
    "pandas",
    'contextvars;python_version<"3.7"',  # Contextvars for python <= 3.6
    'dataclasses;python_version<"3.7"',  # Dataclass for python <= 3.6
    'pickle5;python_version<"3.8"',  # pickle protocol 5 for python <= 3.8
    "graphviz",
    # We can also use patched pytorch-lightning branch to add bfloat16 training support
    "pytorch_lightning==1.6.3",
    "jsonargparse[signatures]@git+https://github.com/leezu/jsonargparse@cf8a40fe2a2d91542d1b2798f065be327f29fcad",  # v3.19 + workaround for https://github.com/PyTorchLightning/pytorch-lightning/issues/9207
    "torchmetrics@git+https://github.com/leezu/metrics@d9ad0ac1ee875cc410fd21b49804b65d592459e3",  # v0.5 + workaround for https://github.com/PyTorchLightning/metrics/issues/484
    "fairscale",
    "asttokens",
    "psutil",
    "ujson",
    "seqeval",
    "datasets",
    "mlflow==1.22.0",
    "protobuf<=3.20.1",
]

tests_require = ["pylint==2.15.6", "prospector", "pytest", "pytest-mock", "moto", "lorem"]

rlfh_require = [
    "transformers==4.18.0",
    "datasets==2.1.0",
    "numpy",
    "dataclasses",
    "typing",
    "absl-py",
    "hydra-core",
    "rouge-score",
    "sacrebleu<2.0",
    "nltk==3.6.5",
    "protobuf==3.20.1",
]

pile_require = ["numpy", "pyarrow==8.0.0", "smart-open[s3]", "zstandard==0.18"]

bedrock_encoder_decoder_tests_require = [
    "hydra-core>=1.2",
    "transformers>=4.20",
    "datasets>=2.0.0",
    "tqdm>=4.64.0",
    "ninja>=1.10.2",
    "s3fs>=2022.3.0",
]

extras = {
    "test": tests_require,
    "rlfh": rlfh_require,
    "pile": pile_require,
    "bedrock_encoder_decoder_tests": bedrock_encoder_decoder_tests_require,
}

force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
extensions = []
cmdclass = {}
if (torch.cuda.is_available() and CUDA_HOME is not None) or force_cuda:
    optimizer_include_dirs = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "src/mstar/clib")
    ]
    megatron_include_dirs = [
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "src/mstar/clib/megatron"
        )
    ]

    fused_softmax_nvcc = [
        "-O3",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]
    extensions.extend(
        [
            CUDAExtension(
                name="mstar.fused_optimizers",
                include_dirs=optimizer_include_dirs,
                sources=[
                    "src/mstar/clib/amp_C_frontend.cpp",
                    "src/mstar/clib/multi_tensor_lans.cu",
                    "src/mstar/clib/multi_tensor_adam.cu",
                    "src/mstar/clib/multi_tensor_l2norm_kernel.cu",
                ],
                extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]},
            ),
            # Megatron kernels imported from
            # https://github.com/NVIDIA/Megatron-LM/tree/7a77abd9b6267dc0020a60b424b4748fc22790bb/megatron/fused_kernels
            CUDAExtension(
                name="mstar.scaled_upper_triang_masked_softmax_cuda",
                include_dirs=megatron_include_dirs,
                sources=[
                    "src/mstar/clib/megatron/scaled_upper_triang_masked_softmax.cpp",
                    "src/mstar/clib/megatron/scaled_upper_triang_masked_softmax_cuda.cu",
                ],
                extra_compile_args={"cxx": ["-O3"], "nvcc": fused_softmax_nvcc},
            ),
            CUDAExtension(
                name="mstar.scaled_masked_softmax_cuda",
                include_dirs=megatron_include_dirs,
                sources=[
                    "src/mstar/clib/megatron/scaled_masked_softmax.cpp",
                    "src/mstar/clib/megatron/scaled_masked_softmax_cuda.cu",
                ],
                extra_compile_args={"cxx": ["-O3"], "nvcc": fused_softmax_nvcc},
            ),
            CUDAExtension(
                name="mstar.scaled_softmax_cuda",
                include_dirs=megatron_include_dirs,
                sources=[
                    "src/mstar/clib/megatron/scaled_softmax.cpp",
                    "src/mstar/clib/megatron/scaled_softmax_cuda.cu",
                ],
                extra_compile_args={"cxx": ["-O3"], "nvcc": fused_softmax_nvcc},
            ),
            CUDAExtension(
                name="mstar.fused_mix_prec_layer_norm_cuda",
                include_dirs=megatron_include_dirs,
                sources=[
                    "src/mstar/clib/megatron/layer_norm_cuda.cpp",
                    "src/mstar/clib/megatron/layer_norm_cuda_kernel.cu",
                ],
                extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": ["-O3", "--use_fast_math", "-maxrregcount=50"],
                },
            ),
        ]
    )

    cmdclass["build_ext"] = BuildExtension
else:
    warnings.warn("Cannot install optimized CUDA kernels.")
setup(
    # Metadata
    name="mstar",
    version=VERSION,
    python_requires=">=3.6",
    description="M*",
    long_description_content_type="text/markdown",
    # Package info
    packages=find_packages(where="src", exclude=("tests", "scripts")),
    package_dir={"": "src"},
    package_data={"": [os.path.join("datasets", "dataset_checksums", "*.txt")]},
    zip_safe=True,
    include_package_data=True,
    install_requires=requirements,
    tests_require=tests_require,
    rlfh_require=rlfh_require,
    pile_require=pile_require,
    bedrock_encoder_decoder_tests_require=bedrock_encoder_decoder_tests_require,
    extras_require=extras,
    ext_modules=extensions,
    cmdclass=cmdclass,
)
