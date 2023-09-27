import numpy as np


class Calculator:
    def __init__(self, num_points, R, block_size):
        self.num_points = num_points
        self.R = R
        self.block_size = block_size
        self.counter = 0

    def thr(self, a, b):
        value = 0
        if a > b:
            value = 1
        return value

    def distribute_val(self, val, a, b):
        ratio = (val - a) / (b - a)
        val_a = round(val * ratio)
        val_b = val - val_a
        return val_a, val_b

    def block_hist(self, data):
        data = np.asarray(data)
        data = data.flatten()
        bins = [0, 8, 16, 24, 32, 40, 48, 56, 64]
        #histbin = {'0': 0, '8': 0, '16': 0, '24': 0, '32': 0, '40': 0, '48': 0, '56': 0, '64': 0}
        hgram, _ = np.histogram(data, bins=bins)
        #for v in range(len(data)):
         #   for b in range(8):
          #      if bins[b] <= data[v] < bins[b+1]:
           #         p, q = self.distribute_val(data[v], bins[b], bins[b+1])
            #        histbin[str(bins[b])] += p
             #       histbin[str(bins[b + 1])] += q
        #hgram = histbin.values()
        return hgram

    def calc_hist(self, img):
        hist = []
        (m, n) = img.shape
        img = img.astype("float")
        code_val = np.zeros((m - 4, n - 4), np.uint8)
        for i in range(2, m - 2):
            for j in range(2, n - 2):
                grad_0 = img[i, j] - img[i + 1, j]
                grad_45 = img[i, j] - img[i + 1, j + 1]
                grad_90 = img[i, j] - img[i, j + 1]
                grad_135 = img[i, j] - img[i - 1, j + 1]
                vector = [self.thr(grad_0, grad_45), self.thr(grad_0, grad_90), self.thr(grad_0, grad_135),
                          self.thr(grad_45, grad_90), self.thr(grad_45, grad_135), self.thr(grad_90, grad_135)]
                val = 32 * vector[0] + 16 * vector[1] + 8 * vector[2] + 4 * vector[3] + 2 * vector[4] + vector[5]
                code_val[i - 2, j - 2] = val

        r = 0
        while r + 8 < m - 3:
            c = 0
            while c + 8 < n - 3:
                block = code_val[r:r + 8, c:c + 8]
                h = self.block_hist(block)
                hist.append(h)
                c += 8
            r += 8
        hist = [v for ele in hist for v in ele]
        return hist
