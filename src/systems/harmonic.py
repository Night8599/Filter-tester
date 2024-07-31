from utils.data import Data
from utils.state import State
from ..utils.system import System

import math
class Harmonic(System):
    def __init__(self, istate: State = ...) -> None:
        super().__init__(istate)
        self.config()
    def config(self, frequency = 1, phase = 0):
        self.frequency = frequency
        self.phase = phase
    def update(self, t) -> State:
        self.state.pos = math.sin(self.frequency * t + self.phase) + self.istate.pos + self.istate.vel * t + self.istate.acc * t**2 / 2
        self.state.vel = self.frequency * math.cos(self.frequency * t + self.phase) + self.istate.vel + self.istate.acc * t
        self.state.acc = -self.frequency ** 2 * math.sin(self.frequency * t + self.phase) + self.istate.acc
        return self.state