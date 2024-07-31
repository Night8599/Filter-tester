import numpy as np
import matplotlib.pyplot as plt

# Definição do modelo harmônico com ruído
def harmonic_motion(t, amplitude, frequency, phase, noise_std=0.5):
    true_pos = amplitude * np.sin(frequency * t + phase)
    measured_pos = true_pos + np.random.normal(0, noise_std, size=true_pos.shape)
    return true_pos, measured_pos

# Implementação do filtro de Kalman
class KalmanFilter:
    def __init__(self, dt, process_var, measurement_var):
        self.dt = dt
        self.process_var = process_var
        self.measurement_var = measurement_var

        self.A = np.array([[1, dt], [0, 1]])  # Matriz de transição de estado
        self.H = np.array([[1, 0]])           # Matriz de observação
        self.Q = process_var * np.eye(2)      # Matriz de covariância do processo
        self.R = measurement_var              # Variância da medição

        self.x = np.zeros((2, 1))  # Vetor de estado inicial [posição, velocidade]
        self.P = np.eye(2)         # Matriz de covariância de erro inicial

    def predict(self):
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    def get_state(self):
        return self.x.flatten()

# Parâmetros do modelo harmônico
amplitude = 1.0
frequency = 2.0
phase = 0.0
dt = 0.1
time = np.arange(0, 10, dt)

# Gerar movimento harmônico e medições ruidosas
true_pos, measured_pos = harmonic_motion(time, amplitude, frequency, phase)

# Inicializar filtro de Kalman
kf = KalmanFilter(dt, process_var=0.1, measurement_var=0.5)

# Vetores para armazenar estimativas do filtro de Kalman
kalman_pos = []
kalman_vel = []

# Aplicar filtro de Kalman às medições
for z in measured_pos:
    kf.predict()
    kf.update(np.array([[z]]))
    state = kf.get_state()
    kalman_pos.append(state[0])
    kalman_vel.append(state[1])

# Converter listas para arrays
kalman_pos = np.array(kalman_pos)
kalman_vel = np.array(kalman_vel)

# Plotar resultados
plt.figure(figsize=(10, 8))

# Plotar posição
plt.subplot(2, 1, 1)
plt.plot(time, true_pos, label='Posição Verdadeira')
plt.plot(time, measured_pos, label='Medições Ruidosas', linestyle='dotted')
plt.plot(time, kalman_pos, label='Filtro de Kalman', linestyle='dashed')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição')
plt.legend()
plt.title('Filtro de Kalman aplicado ao Movimento Harmônico')

# Plotar velocidade
plt.subplot(2, 1, 2)
plt.plot(time, frequency * amplitude * np.cos(frequency * time + phase), label='Velocidade Verdadeira')
plt.plot(time, kalman_vel, label='Filtro de Kalman', linestyle='dashed')
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade')
plt.legend()

plt.tight_layout()
plt.show()
