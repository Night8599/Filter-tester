'''
A base class for all filters.
'''

from data import FilterData

class Filter:
    def __init__(self, name=str, config=None):
        self.name = name
        self.config = config
    def iterate(self, data=None) -> FilterData:
        raise NotImplementedError("Subclasses must implement this method.")
    