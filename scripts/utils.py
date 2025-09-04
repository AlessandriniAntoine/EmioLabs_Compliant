import os
from posix import stat
import numpy as np

data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

class Observer:

    def __init__(self, observer_path):
        data_obs = np.load(os.path.join(data_path, observer_path))
        self.A = data_obs['stateMatrix']
        self.B = data_obs['inputMatrix']
        self.C = data_obs['outputMatrix']
        self.L = data_obs['observerGain']

        self.state = np.zeros((self.A.shape[0], 1))
        self.output = np.zeros((self.C.shape[0], 1))

    def update(self, u, y):
        self.state = self.A @ self.state + self.B @ u + self.L @ (y - self.output)
        self.output = self.C @ self.state


class StateFeedback:
    def __init__(self, state_feedback_path):
        self.load_data(state_feedback_path)
        self.command = np.zeros((self.K.shape[0], 1))

    def update(self, x, r):
        self.command = self.G @ r - self.K @ x

    def load_data(self, state_feedback_path):
        data_fb = np.load(os.path.join(data_path, state_feedback_path))
        self.K = data_fb['statefeedbackGain']
        self.G = data_fb['feedforwardGain']


class StateFeedbackIntegral:
    def __init__(self, state_feedback_path):
        self.load_data(state_feedback_path)
        self.command = np.zeros((self.K.shape[0], 1))
        self.integral = np.zeros((self.Ki.shape[0], 1))

    def update(self, x, r, y):
        self.command = - self.K @ x - self.Ki @ self.integral
        self.integral += r - y


    def load_data(self, state_feedback_path):
        data_fb = np.load(os.path.join(data_path, state_feedback_path))
        self.K = data_fb['statefeedbackGain']
        self.Ki = data_fb['integralfeedbackGain']
