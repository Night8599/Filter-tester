from state import State
from data import Data
class System():
    def __init__(self, istate:State = State()) -> None:
        self.istate = istate
        self.state = istate
    def config(self,) -> None:
        pass
    def update(self, t) -> State:
        pass
    