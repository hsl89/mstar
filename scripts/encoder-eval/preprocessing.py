from typing import List
from loguru import logger

def lower_case(sentences: List[str]) -> List[str]:
    # logger.info("********* Converting sentences to lower case *********\n\n")
    return [x.lower() for x in sentences]
