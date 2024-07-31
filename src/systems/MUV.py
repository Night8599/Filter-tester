from utils.state import State
from .. utils.system import System
class MUV(System):
    def __init__(self, istate: State = ...) -> None:
        super().__init__(istate)
    def config(self) -> None:
        return super().config()
    def update(self, t) -> State:
        self.state.pos = self.istate.pos + self.istate.vel * t + self.istate.acc * t ** 2 / 2
        self.state.vel = self.istate.vel + self.istate.acc * t
        self.state.acc = self.istate.acc
        return self.state