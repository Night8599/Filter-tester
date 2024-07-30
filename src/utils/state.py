from data import Data

import numpy as np
class State:
    def __init__(self, pos = Data(0,0), vel = Data(0,0), acc = Data(0,0)) -> None:
        self.pos = pos
        self.vel = vel
        self.acc = acc
    def state_array(self):
        return np.array([self.pos.value],
                        [self.vel.value],
                        [self.acc.value])
    def deviation_array(self):
        return np.array([self.pos.deviation],
                        [self.vel.deviation],
                        [self.acc.deviation])