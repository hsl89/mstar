import gzip
import logging
from typing import Iterable

from .document import Document

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class BaseDataFormat(object):
    reader_registry = dict()

    def read(self, file_path: str) -> Iterable[Document]:
        raise NotImplementedError

    @classmethod
    def register(cls, key: str):
        def add_to_registry(subclass):
            if issubclass(subclass, BaseDataFormat):
                if key in BaseDataFormat.reader_registry:
                    logger.warning("Replace registration for key {0} with {1}" % key, subclass.__name__)
                    BaseDataFormat.reader_registry[key] = subclass
                setattr(BaseDataFormat, key, subclass)
                return subclass
            else:
                raise ValueError("Trying to register non DataFormat "
                                 "class %s to DataFormat" % subclass.__name__)

        return add_to_registry

    @staticmethod
    def compression_aware_open(fp):
        """
        Support reading from plain file OR gzip file
        """
        if fp.endswith(".gz"):
            return gzip.open(fp, 'rb')
        return open(fp, 'r', encoding='utf8')
