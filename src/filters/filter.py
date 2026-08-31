'''
A base class for all filters.
'''

class Filter:
    def __init__(self, name=str, config=dict):
        self.name = name
        self.config = config
        self.state_estimate = None
        self.check_config(config)
    def iterate(self, data=None) -> None:
        raise NotImplementedError("Subclasses must implement this method.")
    def check_config(self):
        # Implement configuration validation logic here
        pass
    