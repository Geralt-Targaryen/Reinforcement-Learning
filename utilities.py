import random
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


class Buffer:
    def __init__(self, n):
        self.buffer = []
        self.n = n

    def store(self, sars):
        if len(self.buffer) > self.n:
            self.buffer.pop(0)
        self.buffer.append(sars)

    def sample(self, m):
        return random.sample(self.buffer, min(m, len(self.buffer)))


def moving_average(x, x_smoothed, m=10):
    alpha = 1 / (2 * m + 1)
    for i, r in enumerate(x):
        if i < m:
            x_smoothed.append(np.sum(x[:i + m + 1]) * alpha + x[i] * (1 - alpha * (i + m + 1)))
        elif i >= len(x) - m:
            x_smoothed.append(np.sum(x[i - m:]) * alpha + x[i] * (1 - alpha * (len(x) - i + m)))
        else:
            x_smoothed.append(np.sum(x[i - m:i + m + 1]) * alpha)
