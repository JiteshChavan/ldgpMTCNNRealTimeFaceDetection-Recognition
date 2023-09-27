import numpy as np
import math


class Calculator:
    def __init__(self, block_size, stride):
        self.block_size = block_size
        self.stride = stride

    def thr(self, a, b):
        value = 0
        if a > b:
            value = 1
        return value

    def block_hist(self, data):
        data = np.asarray(data)
        data = data.flatten()
        bins = [0, 8, 16, 24, 32, 40, 48, 56]
        hgram, _ = np.histogram(data, bins=bins)
        return hgram

    def calc_hist(self, img):
        hist = []
        m = img.shape[0]
        n = img.shape[1]
        img = img.astype("float")
        s1 = math.floor(m/self.stride)
        s2 = math.floor(n/self.stride)
        indexi = 0
        code_val = np.zeros((s1, s2), np.uint8)
        for i in range(2, m - 1 - self.stride, self.stride):
            indexj = 0
            for j in range(2, n - 1 - self.stride, self.stride):
                grad_0 = img[i, j] - img[i + 1, j]
                grad_45 = img[i, j] - img[i + 1, j + 1]
                grad_90 = img[i, j] - img[i, j + 1]
                grad_135 = img[i, j] - img[i - 1, j + 1]
                vector = [self.thr(grad_0, grad_45), self.thr(grad_0, grad_90), self.thr(grad_0, grad_135),
                          self.thr(grad_45, grad_90), self.thr(grad_45, grad_135), self.thr(grad_90, grad_135)]
                val = 32 * vector[0] + 16 * vector[1] + 8 * vector[2] + 4 * vector[3] + 2 * vector[4] + vector[5]
                code_val[indexi, indexj] = val
                indexj += 1
                # try:
                #     code_val[indexi, indexj] = val
                #     indexi+=1
                #     indexj+=1
                #
                # except:
                #     print("(i, j)=("+str(i)+")+("+str(j)+")")
                #     print("No. of codes : "+str(count))
                #     print("Shape of code_val: "+str(code_val.shape))
                #     stop=input("Stopped")
            indexi += 1

        r = 0
        while r + self.block_size < m - 3:
            c = 0
            while c + self.block_size < n - 3:
                block = code_val[r:r + self.block_size, c:c + self.block_size]
                h = self.block_hist(block)
                hist.append(h)
                c += self.block_size
            r += self.block_size
        hist = [v for ele in hist for v in ele]
        return hist
