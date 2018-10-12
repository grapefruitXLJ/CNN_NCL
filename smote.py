import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
import scipy.io as sio


class Smote:
    def __init__(self, samples, N=10, k=5):
        self.n_samples, self.n_attrs = samples.shape
        self.N = N
        self.k = k
        self.samples = samples
        self.newindex = 0
        # self.synthetic = no.zeros((self.n_samples*N, self.n_atters))

    def over_sample(self):
        N = int(self.N/100)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        print('neighbors', neighbors)
        for i in range(len(self.samples)):
            nnarray = neighbors.kneighbors(self.samples[i].reshape(1, -1), return_distance=False)[0]
            # print(nnarry)
            self._populate(N, i, nnarray)
        return self.synthetic

    # for each minority class samples, choose N of the K nearest neighbors and generate N synthetic samples.
    def _populate(self, N, i, nnarry):
        for j in range(N):
            nn = random.randint(0, self.k-1)
            dif = self.samples[nnarry[nn]]-self.samples[i]
            gap = random.random()
            self.synthetic[self.newindex] = self.samples[i] + gap*dif
            self.newindex += 1


if __name__ == "__main__":
    sample = sio.loadmat("E:\\算法相关\\cornData_1\\cornsamplenew.mat")
    sample = sample['cornsamplenew']
    # a = np.array([[[1, 2, 3], [5, 6]], [[4, 5, 6], [7, 8]], [[2, 3, 1], [3, 2]], [[2, 1, 2], [5, 9]], [[2, 3, 4], [6, 5]], [[2, 3, 4], [8, 6]]])
    s = Smote(samples=sample, N=1000, k=5)
    out = s.over_sample()
    # print(out)
    sio.savemat("E:\\算法相关\\cornData_9.27\\Samplegather10.mat", {'sample': out[:, :700], 'propval': out[:, 700:]})




