import cv2
import numpy as np
from os import listdir
from os.path import isfile, join, exists

class Affine_Transform:

    def __init__(self):
        self.arr = {}
        self.arr['singapore_25.jpg'] = 'blue_singapore'
        self.arr['singapore.jpg'] = 'demo_singapore'
        self.arr['felicity.jpeg'] = 'demo_feicity'
        self.arr['iiit2.jpeg'] = 'demo_iiit2'
        self.arr['city.jpg'] = 'city'
        self.arr['minar.jpg'] = 'minar'
        self.arr['paris.jpg'] = 'demo_paris'
        self.arr['hampi_4.jpg'] = 'demo_hampi'

    def Vk(self, I):
        return I.reshape((-1, 3)).T

    def Vdk(self, I):
        vk = self.Vk(I)
        return vk, np.vstack([vk, np.ones((1, vk.shape[-1]))])

    def affine_transform_k(self, Ik, Mk, Tk, G, epsilon, gamma):
        vIk, vdIk = self.Vdk(Ik)
        vMk, vdMk = self.Vdk(Mk)
        vTk, vdTk = self.Vdk(Tk)
        
        B = np.linalg.pinv(vdIk @ vdIk.T + epsilon * vdMk @ vdMk.T + gamma * np.eye(4))
        M = np.eye(vIk.shape[-1]) - vdIk.T @ B @ vdIk
        u = (epsilon * vTk @ vdMk.T + gamma * G) @ B @ vdIk
        return np.clip((u @ np.linalg.pinv(M)).astype('uint8'), 0, 255).T.reshape(Ik.shape)

    def affine_G(self, M, T):
        return self.Vk(T) @ np.linalg.pinv(self.Vdk(M)[1])
        
    def affine_transform(self, I, M, T, window=(5, 5), epsilon=1e-10, gamma=1):
        G = self.affine_G(M, T)
        O = np.zeros(I.shape, dtype='uint8')
        for i in range(0, I.shape[0], window[0]):
            for j in range(0, I.shape[1], window[1]):
                O[i:i+window[0], j:j+window[1]] = self.affine_transform_k(I[i:i+window[0], j:j+window[1]], M[i:i+window[0], j:j+window[1]], T[i:i+window[0], j:j+window[1]], G, epsilon, gamma)
        return O[:I.shape[0] - I.shape[0] % window[0], :I.shape[1] - I.shape[1] % window[1]]    
    
    def output(self, name):
        img = cv2.imread('../results/' + self.arr[name] + "/output.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img