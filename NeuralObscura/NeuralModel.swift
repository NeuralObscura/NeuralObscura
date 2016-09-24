//
//  NeuralModel.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 9/23/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

private func makeConv(device: MTLDevice,
                      inDepth: Int,
                      outDepth: Int,
                      weights: UnsafePointer<Float>,
                      bias: UnsafePointer<Float>,
                      stride: Int) -> MPSCNNConvolution {

    // All VGGNet conv layers use a 3x3 kernel with stride 1.
    let desc = MPSCNNConvolutionDescriptor(kernelWidth: 3,
                                           kernelHeight: 3,
                                           inputFeatureChannels: inDepth,
                                           outputFeatureChannels: outDepth,
                                           neuronFilter: nil)
    desc.strideInPixelsX = stride
    desc.strideInPixelsY = stride

    let conv = MPSCNNConvolution(device: device,
                                 convolutionDescriptor: desc,
                                 kernelWeights: weights,
                                 biasTerms: bias,
                                 flags: MPSCNNConvolutionFlags.none)

    // To preserve the width and height between conv layers, VGGNet assumes one
    // pixel of padding around the edges. Metal apparently has no problem reading
    // outside the source image, so we don't have to do anything special here.
    conv.edgeMode = .zero

    return conv
}

/*
 class FastStyleNet(chainer.Chain):
 def __init__(self):
 super(FastStyleNet, self).__init__(
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

 def __call__(self, x, test=False):
 h = self.b1(F.elu(self.c1(x)), test=test)
 h = self.b2(F.elu(self.c2(h)), test=test)
 h = self.b3(F.elu(self.c3(h)), test=test)
 h = self.r1(h, test=test)
 h = self.r2(h, test=test)
 h = self.r3(h, test=test)
 h = self.r4(h, test=test)
 h = self.r5(h, test=test)
 h = self.b4(F.elu(self.d1(h)), test=test)
 h = self.b5(F.elu(self.d2(h)), test=test)
 y = self.d3(h)
 return (F.tanh(y)+1)*127.5
 */

class NeuralModel {
    var layerData = [String: StyleModelData]()
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var c1, c2, c3: MPSCNNConvolution?

    init(device: MTLDevice) {
        self.device = device
        commandQueue = device.makeCommandQueue()
        c1 = nil
        c2 = nil
        c3 = nil
    }

    func setup_model() {
        // c1=L.Convolution2D(3, 32, 9, stride=1, pad=4),
        c1 = makeConv(device: device,
                      inDepth: 3,
                      outDepth: 32,
                      weights: layerData["c1_W"]!.pointer(),
                      bias: layerData["c1_b"]!.pointer(),
                      stride: 1)
        // c2=L.Convolution2D(32, 64, 4, stride=2, pad=1),
        c2 = makeConv(device: device,
                      inDepth: 32,
                      outDepth: 64,
                      weights: layerData["c2_W"]!.pointer(),
                      bias: layerData["c2_b"]!.pointer(),
                      stride: 2)
        // c3=L.Convolution2D(64, 128, 4,stride=2, pad=1),
        c3 = makeConv(device: device,
                      inDepth: 64,
                      outDepth: 128,
                      weights: layerData["c3_W"]!.pointer(),
                      bias: layerData["c3_b"]!.pointer(),
                      stride: 2)

        // r1=ResidualBlock(128, 128),
        // r2=ResidualBlock(128, 128),
        // r3=ResidualBlock(128, 128),
        // r4=ResidualBlock(128, 128),
        // r5=ResidualBlock(128, 128),
        // d1=L.Deconvolution2D(128, 64, 4, stride=2, pad=1),
        // d2=L.Deconvolution2D(64, 32, 4, stride=2, pad=1),
        // d3=L.Deconvolution2D(32, 3, 9, stride=1, pad=4),
        // b1=L.BatchNormalization(32),
        // b2=L.BatchNormalization(64),
        // b3=L.BatchNormalization(128),
        // b4=L.BatchNormalization(64),
        // b5=L.BatchNormalization(32),
    }

    //    func transform(texture: MTLTexture) -> MTLTexture {
    //
    //    }
}
