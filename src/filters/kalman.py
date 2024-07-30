from utils.data import Data
from ..utils.filter import Filter
from ..utils.state import State

import numpy as np
class Kalman(Filter):
    def __init__(self, pos:Data=..., vel:Data=..., acc:Data=..., A = lambda t: np.array([1, t, t**2/2],
                                                                                        [0, 1, t],
                                                                                        [0, 0, 1]),
                Q = lambda t: np.zeros((3,3)))-> None:
        super().__init__(pos, vel, acc)
        self.state = State(pos, vel, acc)
        self.X = np.array([pos.value],
                          [vel.value],
                          [acc.value])
        self.P = np.array([pos.deviation, 0, 0],
                          [0, vel.deviation, 0],
                          [0, 0, acc.deviation])
        self.A = A
        self.Q = Q
        self.Z = np.zeros((1,3))
        self.H = np.zeros((1,3))
        self.R = np.zeros((3,3))
    def predict(self, t:float) -> None:
        self.X = np.matmul(self.A(t), self.X)
        self.P = np.matmul(np.matmul(self.A(t), self.X), self.A(t)) + self.Q(t)
    
    def update(self,t:float) -> None:
        self.predict(t)
        
        Y = self.Z - np.matmul(self.H,self.X)
        S = np.matmul(np.matmul(self.H, self.P), np.transpose(self.H)) + self.R
        K = np.matmul(np.matmul(self.P,np.transpose(self.H)),np.linalg.inv(S))
        self.X = self.X + np.matmul(K,Y)
        self.P = self.P - np.matmul(np.matmul(K, self.H), self.P)
        
        self.state.pos.value = self.X[0][0]
        self.state.vel.value = self.X[1][0]
        self.state.acc.value = self.X[2][0]
        self.state.pos.deviation = self.P[0][0]
        self.state.vel.deviation = self.P[1][1]
        self.state.acc.deviation = self.P[2][2]
        