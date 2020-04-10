import numpy as np
class StrideFill2D:
    def __init__(self, stride=None):
        if stride is None:
            self.stride = None
        elif type(stride) == int:
            self.stride = (stride, stride)
        else:
            self.stride = tuple(stride)
    def __call__(self, arr):
        return self.forward(arr)
    def forward(self, arr):
        if self.stride is not None:
            orow, ocol = arr.shape
            ncol = ocol + self.stride[1] * 2
            if self.stride[1] == 0:
                nr = arr.copy()
            else:
                nr = np.c_[np.zeros((orow, self.stride[1])), arr, np.zeros((orow, self.stride[1]))]
            if self.stride[0] == 0:
                a = nr
            else:
                a = np.r_[
                    np.zeros((self.stride[0], ncol)), 
                    nr,
                    np.zeros((self.stride[0], ncol)), 
                ]
        else:
            a = arr.copy()
        return a

class Pool2D:
    def __init__(self, size, stride=None):
        if type(size) == int:
            self.size = (size, size)
        else:
            self.size = tuple(size)
        self.fill = StrideFill2D(stride)
    def __call__(self, arr):
        return self.forward(arr)
    def forward(self, arr):
        a = self.fill(arr)
        assert (a.shape[0] % self.size[0] == 0) and (a.shape[1] % self.size[1] == 0)
        rsize, csize = self.size
        res = np.empty((int(a.shape[0] / self.size[0]), int(a.shape[1] / self.size[1])))
        for ir, r in enumerate(range(0, a.shape[0], self.size[0])):
            for ic, c in enumerate(range(0, a.shape[1], self.size[1])):
                res[ir][ic] = self.op(a[r:r+rsize, c:c+csize])
        return res
    def op(self, sub):
        return 0

class MaxPool2D(Pool2D):
    def op(self, sub):
        return np.max(sub)

class AvgPool2D(Pool2D):
    def op(self, sub):
        return np.mean(sub)

class Conv2D:
    def __init__(self, size, stride=None, kernal=None):
        if type(size) == int:
            self.size = (size, size)
        else:
            self.size = tuple(size)
        self.fill = StrideFill2D(stride)
        if kernal is None:
            self.kernal = np.empty(self.size)
        else:
            assert kernal.shape == size
            self.kernal = kernal
    def __call__(self, arr):
        return self.forward(arr)
    def forward(self, arr):
        a = self.fill(arr)
        rsize, csize = self.size
        assert rsize <= a.shape[0] and csize <= a.shape[1]
        res = np.empty((a.shape[0] - rsize + 1, a.shape[1] - rsize + 1))
        for r in range(0, a.shape[0] - rsize):
            for c in range(0, a.shape[1] - csize):
                res[r][c] = self.op(a[r:r+rsize, c:c+csize])
        return res
    def op(self, sub):
        return np.sum(sub*self.kernal)
    def getParameter(self):
        return self.kernal.copy()

if __name__ == "__main__":
    a = np.random.randn(2, 4)
    print(a)
    pool = MaxPool2D(2, stride=(1, 0))
    print(pool(a))
    pool2 = AvgPool2D((2, 4))
    print(pool2(a))
    conv = Conv2D(3, stride=(1, 0))
    print(conv(a))