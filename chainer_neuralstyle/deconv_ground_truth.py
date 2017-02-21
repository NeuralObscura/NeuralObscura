#!/usr/bin/env python

import math

import numpy as np
import argparse
from PIL import Image
import time

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable, cuda, serializers


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

class DeconvolutionNet(chainer.Chain):
    def __init__(self):
        super(DeconvolutionNet, self).__init__(
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
            b3=L.BatchNormalization(128)
        )

    def __call__(self, x, test=False):
        h = self.b1(F.elu(self.c1(x)), test=test)
        h = self.b2(F.elu(self.c2(h)), test=test)
        h = self.b3(F.elu(self.c3(h)), test=test)
        h = self.r1(h, test=test)
        h = self.r2(h, test=test)
        h = self.r3(h, test=test)
        h = self.r4(h, test=test)
        td = self.r5(h, test=test)
        gt = self.d1(td)
        return (td, gt)

parser = argparse.ArgumentParser(description='Generate input and ground truth data for iOS ML framework tests')
parser.add_argument('input')
# parser.add_argument('--model', '-m', type=str, help="Model classname to use for generating test data")
parser.add_argument('--tdout', type=str, help="Output path for test input data")
parser.add_argument('--gtout', type=str, help="Output path for ground truth output data")
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--params', '-p', default='models/style.model', type=str)
args = parser.parse_args()

model = DeconvolutionNet()
serializers.load_npz(args.params, model)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

image = xp.asarray(Image.open(args.input).convert('RGB'), dtype=xp.float32).transpose(2, 0, 1)
image = image.reshape((1,) + image.shape)
x = Variable(image)
(td, gt) = model(x)
td_result = cuda.to_cpu(td.data)
td_shape = td_result.shape
assert td_shape[0] == 1
td_shape_new = td_shape[1:]
    
gt_result = cuda.to_cpu(gt.data)
gt_shape = gt_result.shape
assert gt_shape[0] == 1
gt_shape_new = gt_shape[1:]
np.save(args.tdout, td_result.reshape(td_shape_new))
np.save(args.gtout, gt_result.reshape(gt_shape_new))

# Run me:
# python chainer_neuralstyle/deconv_ground_truth.py NeuralObscura/debug.png --gtout NeuralObscuraTests/testdata/deconv-ground-truth.npy --tdout NeuralObscuraTests/testdata/deconv-test-data.npy --params chainer_neuralstyle/models/composition.model
