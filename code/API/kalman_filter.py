import numpy as np

class KalmanFilter(object):
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1))  # state
        self.P = np.eye(dim_x)         # uncertainty covariance
        self.Q = np.eye(dim_x)         # process uncertainty
        self.B = 0
        self.F = np.eye(dim_x)         # state transition matrix
        self.H = np.zeros((dim_z, dim_x))  # Measurement function
        self.R = np.eye(dim_z)         # measurement uncertainty
        self.z = np.array([[None]*self.dim_z]).T

    def predict(self, u=0):
        if np.isscalar(u):
            self.x = np.dot(self.F, self.x) + self.B*u
        else:
            self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x.copy()

    def update(self, z):
        self.z = np.reshape(z, (self.dim_z, 1))
        y = self.z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x += np.dot(K, y)
        I = np.eye(self.dim_x)
        self.P = np.dot(I - np.dot(K, self.H), self.P) 