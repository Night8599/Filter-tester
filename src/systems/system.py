'''
A base class for all systems.
'''

class System:
    def __init__(self, name=str, config=dict):
        self.name = name
        self.config = config
        self.state_data = {}
        self.check_config(config)

    def iterate(self, t) -> None:
        raise NotImplementedError("Subclasses must implement this method.")

    def check_config(self):
        # Implement configuration validation logic here
        pass