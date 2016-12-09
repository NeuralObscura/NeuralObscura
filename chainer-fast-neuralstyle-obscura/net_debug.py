import math

import numpy as np
np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True)
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable

class ResidualBlock(chainer.Chain):
    def __init__(self, n_in, n_out, stride=1, ksize=3):
        w = math.sqrt(2)
        super(ResidualBlock, self).__init__(
            c1=L.Convolution2D(n_in, n_out, ksize, stride, 1, w),
            c2=L.Convolution2D(n_out, n_out, ksize, 1, 1, w),
            b1=L.BatchNormalization(n_out),
            b2=L.BatchNormalization(n_out)
        )

    def __call__(self, x, test):
        h = F.relu(self.b1(self.c1(x), test=test))
        h = self.b2(self.c2(h), test=test)
        if x.data.shape != h.data.shape:
            xp = chainer.cuda.get_array_module(x.data)
            n, c, hh, ww = x.data.shape
            pad_c = h.data.shape[1] - c
            p = xp.zeros((n, pad_c, hh, ww), dtype=xp.float32)
            p = chainer.Variable(p, volatile=test)
            x = F.concat((p, x))
            if x.data.shape[2:] != h.data.shape[2:]:
                x = F.average_pooling_2d(x, 1, 2)
        return h + x

class FastStyleNett(chainer.Chain):
    def __init__(self):
        super(FastStyleNett, self).__init__(
            c1=L.Convolution2D(3, 32, 9, stride=1, pad=4),
            c2=L.Convolution2D(32, 64, 4, stride=2, pad=1),
            c3=L.Convolution2D(64, 128, 4,stride=2, pad=1),
            r1=ResidualBlock(128, 128),
            r2=ResidualBlock(128, 128),
            r3=ResidualBlock(128, 128),
            r4=ResidualBlock(128, 128),
            r5=ResidualBlock(128, 128),
            d1=L.Deconvolution2D(128, 64, 4, stride=2, pad=1),
            d2=L.Deconvolution2D(64, 32, 4, stride=2, pad=1),
            d3=L.Deconvolution2D(32, 3, 9, stride=1, pad=4),
            b1=L.BatchNormalization(32),
            b2=L.BatchNormalization(64),
            b3=L.BatchNormalization(128),
            b4=L.BatchNormalization(64),
            b5=L.BatchNormalization(32),
        )

    def four_corners(self, name, x):
        print name
        print x.shape
        shape = x.shape
        depth = 2
        y = x.data.copy()
        print y
        print "---------------"

    def __call__(self, x, test=False):
        # self.four_corners("x",x)
        self.four_corners("self.c1(x)",self.c1(x))
        # self.four_corners("F.relu(self.c1(x))", F.relu(self.c1(x)))
        # self.four_corners("self.b1(F.relu(self.c1(x)))", self.b1(F.relu(self.c1(x))))
        h = self.b1(F.relu(self.c1(x)), test=test)

        # print "h = self.b1(F.relu(self.c1(x)), test=test)"
        # self.four_corners("h",h)
        # self.four_corners("self.c2(h)",self.c2(h))
        # self.four_corners("F.relu(self.c2(h))", F.relu(self.c2(h)))
        # self.four_corners("self.b2(F.relu(self.c2(h)))", self.b2(F.relu(self.c2(h))))
        h = self.b2(F.relu(self.c2(h)), test=test)

        # print "h = self.b2(F.relu(self.c2(h)), test=test)"
        # self.four_corners("h",h)
        # self.four_corners("self.c3(h)",self.c3(h))
        # self.four_corners("F.relu(self.c3(h))", F.relu(self.c3(h)))
        # self.four_corners("self.b3(F.relu(self.c3(h)))", self.b3(F.relu(self.c3(h))))
        h = self.b3(F.relu(self.c3(h)), test=test)

        # print "h = self.b3(F.relu(self.c3(h)), test=test)"
        # self.four_corners("h",h)
        # self.four_corners("self.r1(h, test=test)",self.r1(h, test=test))
        h = self.r1(h, test=test)

        # print "h = self.r1(h, test=test)"
        # self.four_corners("h",h)
        # self.four_corners("h = self.r2(h, test=test)",self.r2(h, test=test))
        h = self.r2(h, test=test)

        # print "h = self.r2(h, test=test)"
        # self.four_corners("h",h)
        # self.four_corners("h = self.r3(h, test=test)",self.r3(h, test=test))
        h = self.r3(h, test=test)

        print "h = self.r3(h, test=test)"
        self.four_corners("h",h)
        self.four_corners("h = self.r4(h, test=test)",self.r4(h, test=test))
        h = self.r4(h, test=test)

        print "h = self.r4(h, test=test)"
        self.four_corners("h",h)
        self.four_corners("h = self.r5(h, test=test)",self.r5(h, test=test))
        h = self.r5(h, test=test)

        print "h = self.r5(h, test=test)"
        self.four_corners("h",h)
        self.four_corners("self.d1(h)",self.d1(h))
        self.four_corners("F.relu(self.d1(h)",F.relu(self.d1(h)))
        self.four_corners("self.b4(F.relu(self.d1(h))",self.b4(F.relu(self.d1(h))))
        h = self.b4(F.relu(self.d1(h)), test=test)

        print "h = self.b4(F.relu(self.d1(h)), test=test)"
        self.four_corners("h",h)
        self.four_corners("self.d2(h)",self.d2(h))
        self.four_corners("F.relu(self.d2(h)",F.relu(self.d2(h)))
        self.four_corners("self.b5(F.relu(self.d2(h))",self.b5(F.relu(self.d2(h))))
        h = self.b5(F.relu(self.d2(h)), test=test)

        print "h = self.b5(F.relu(self.d2(h)), test=test)"
        self.four_corners("h",h)
        self.four_corners("self.d3(h)",self.d3(h))
        y = self.d3(h)

        print "y = self.d3(h)"
        self.four_corners("y",y)
        self.four_corners("(F.tanh(y)+1)*127.5",(F.tanh(y)+1)*127.5)

        return (F.tanh(y)+1)*127.5
