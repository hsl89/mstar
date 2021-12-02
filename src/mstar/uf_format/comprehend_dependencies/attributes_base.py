from typing import Dict


class AttributesBase(object):
    """
    Base class to declare a class has an 'immutable' attributes
    """

    def __init__(self,
                 attributes: Dict[str, object]):
        if attributes is None:
            self._attributes = dict()
        else:
            self._attributes = attributes

    def get(self, key):
        return self._attributes.get(key, None)

    def set(self, key, value):
        self._attributes[key] = value

    @property
    def attributes(self):
        return self._attributes
