import pytest
import argparse



def pytest_addoption(parser):
    parser.addoption("--model-type", action="store", default="default name")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.model_type
    if 'model_type' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("model_type", [option_value])
