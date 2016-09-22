//
//  CompositionModel.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 9/20/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

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
private func makeConv(device: MTLDevice,
                      inDepth: Int,
                      outDepth: Int,
                      weights: UnsafePointer<Float>,
                      bias: UnsafePointer<Float>,
                      stride: Int) -> MPSCNNConvolution {

    // This should apply the equivlent of "no activation function"
    let neuronFilter = MPSCNNNeuronLinear(device: device, a: 1, b:0)

    // All VGGNet conv layers use a 3x3 kernel with stride 1.
    let desc = MPSCNNConvolutionDescriptor(kernelWidth: 3,
                                           kernelHeight: 3,
                                           inputFeatureChannels: inDepth,
                                           outputFeatureChannels: outDepth,
                                           neuronFilter: neuronFilter)
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


class CompositionModel {
    var layerData = [String: StyleModelData]()
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let c1, c2, c3: MPSCNNConvolution

    init(device: MTLDevice) {
        // PASTE INIT DATA HERE
        layerData["r4_c2_W"] = StyleModelData(modelName: "composition", rawFileName: "r4_c2_W")
        //r4_c2_W shape = (128, 128, 3, 3)
        layerData["r4_c2_b"] = StyleModelData(modelName: "composition", rawFileName: "r4_c2_b")
        //r4_c2_b shape = (128,)
        layerData["r4_c1_W"] = StyleModelData(modelName: "composition", rawFileName: "r4_c1_W")
        //r4_c1_W shape = (128, 128, 3, 3)
        layerData["r4_c1_b"] = StyleModelData(modelName: "composition", rawFileName: "r4_c1_b")
        //r4_c1_b shape = (128,)
        layerData["r4_b1_gamma"] = StyleModelData(modelName: "composition", rawFileName: "r4_b1_gamma")
        //r4_b1_gamma shape = (128,)
        layerData["r4_b1_beta"] = StyleModelData(modelName: "composition", rawFileName: "r4_b1_beta")
        //r4_b1_beta shape = (128,)
        layerData["r4_b2_gamma"] = StyleModelData(modelName: "composition", rawFileName: "r4_b2_gamma")
        //r4_b2_gamma shape = (128,)
        layerData["r4_b2_beta"] = StyleModelData(modelName: "composition", rawFileName: "r4_b2_beta")
        //r4_b2_beta shape = (128,)
        layerData["r5_c2_W"] = StyleModelData(modelName: "composition", rawFileName: "r5_c2_W")
        //r5_c2_W shape = (128, 128, 3, 3)
        layerData["r5_c2_b"] = StyleModelData(modelName: "composition", rawFileName: "r5_c2_b")
        //r5_c2_b shape = (128,)
        layerData["r5_c1_W"] = StyleModelData(modelName: "composition", rawFileName: "r5_c1_W")
        //r5_c1_W shape = (128, 128, 3, 3)
        layerData["r5_c1_b"] = StyleModelData(modelName: "composition", rawFileName: "r5_c1_b")
        //r5_c1_b shape = (128,)
        layerData["r5_b1_gamma"] = StyleModelData(modelName: "composition", rawFileName: "r5_b1_gamma")
        //r5_b1_gamma shape = (128,)
        layerData["r5_b1_beta"] = StyleModelData(modelName: "composition", rawFileName: "r5_b1_beta")
        //r5_b1_beta shape = (128,)
        layerData["r5_b2_gamma"] = StyleModelData(modelName: "composition", rawFileName: "r5_b2_gamma")
        //r5_b2_gamma shape = (128,)
        layerData["r5_b2_beta"] = StyleModelData(modelName: "composition", rawFileName: "r5_b2_beta")
        //r5_b2_beta shape = (128,)
        layerData["r1_c2_W"] = StyleModelData(modelName: "composition", rawFileName: "r1_c2_W")
        //r1_c2_W shape = (128, 128, 3, 3)
        layerData["r1_c2_b"] = StyleModelData(modelName: "composition", rawFileName: "r1_c2_b")
        //r1_c2_b shape = (128,)
        layerData["r1_c1_W"] = StyleModelData(modelName: "composition", rawFileName: "r1_c1_W")
        //r1_c1_W shape = (128, 128, 3, 3)
        layerData["r1_c1_b"] = StyleModelData(modelName: "composition", rawFileName: "r1_c1_b")
        //r1_c1_b shape = (128,)
        layerData["r1_b1_gamma"] = StyleModelData(modelName: "composition", rawFileName: "r1_b1_gamma")
        //r1_b1_gamma shape = (128,)
        layerData["r1_b1_beta"] = StyleModelData(modelName: "composition", rawFileName: "r1_b1_beta")
        //r1_b1_beta shape = (128,)
        layerData["r1_b2_gamma"] = StyleModelData(modelName: "composition", rawFileName: "r1_b2_gamma")
        //r1_b2_gamma shape = (128,)
        layerData["r1_b2_beta"] = StyleModelData(modelName: "composition", rawFileName: "r1_b2_beta")
        //r1_b2_beta shape = (128,)
        layerData["r2_c2_W"] = StyleModelData(modelName: "composition", rawFileName: "r2_c2_W")
        //r2_c2_W shape = (128, 128, 3, 3)
        layerData["r2_c2_b"] = StyleModelData(modelName: "composition", rawFileName: "r2_c2_b")
        //r2_c2_b shape = (128,)
        layerData["r2_c1_W"] = StyleModelData(modelName: "composition", rawFileName: "r2_c1_W")
        //r2_c1_W shape = (128, 128, 3, 3)
        layerData["r2_c1_b"] = StyleModelData(modelName: "composition", rawFileName: "r2_c1_b")
        //r2_c1_b shape = (128,)
        layerData["r2_b1_gamma"] = StyleModelData(modelName: "composition", rawFileName: "r2_b1_gamma")
        //r2_b1_gamma shape = (128,)
        layerData["r2_b1_beta"] = StyleModelData(modelName: "composition", rawFileName: "r2_b1_beta")
        //r2_b1_beta shape = (128,)
        layerData["r2_b2_gamma"] = StyleModelData(modelName: "composition", rawFileName: "r2_b2_gamma")
        //r2_b2_gamma shape = (128,)
        layerData["r2_b2_beta"] = StyleModelData(modelName: "composition", rawFileName: "r2_b2_beta")
        //r2_b2_beta shape = (128,)
        layerData["r3_c2_W"] = StyleModelData(modelName: "composition", rawFileName: "r3_c2_W")
        //r3_c2_W shape = (128, 128, 3, 3)
        layerData["r3_c2_b"] = StyleModelData(modelName: "composition", rawFileName: "r3_c2_b")
        //r3_c2_b shape = (128,)
        layerData["r3_c1_W"] = StyleModelData(modelName: "composition", rawFileName: "r3_c1_W")
        //r3_c1_W shape = (128, 128, 3, 3)
        layerData["r3_c1_b"] = StyleModelData(modelName: "composition", rawFileName: "r3_c1_b")
        //r3_c1_b shape = (128,)
        layerData["r3_b1_gamma"] = StyleModelData(modelName: "composition", rawFileName: "r3_b1_gamma")
        //r3_b1_gamma shape = (128,)
        layerData["r3_b1_beta"] = StyleModelData(modelName: "composition", rawFileName: "r3_b1_beta")
        //r3_b1_beta shape = (128,)
        layerData["r3_b2_gamma"] = StyleModelData(modelName: "composition", rawFileName: "r3_b2_gamma")
        //r3_b2_gamma shape = (128,)
        layerData["r3_b2_beta"] = StyleModelData(modelName: "composition", rawFileName: "r3_b2_beta")
        //r3_b2_beta shape = (128,)
        layerData["b4_gamma"] = StyleModelData(modelName: "composition", rawFileName: "b4_gamma")
        //b4_gamma shape = (64,)
        layerData["b4_beta"] = StyleModelData(modelName: "composition", rawFileName: "b4_beta")
        //b4_beta shape = (64,)
        layerData["b5_gamma"] = StyleModelData(modelName: "composition", rawFileName: "b5_gamma")
        //b5_gamma shape = (32,)
        layerData["b5_beta"] = StyleModelData(modelName: "composition", rawFileName: "b5_beta")
        //b5_beta shape = (32,)
        layerData["b1_gamma"] = StyleModelData(modelName: "composition", rawFileName: "b1_gamma")
        //b1_gamma shape = (32,)
        layerData["b1_beta"] = StyleModelData(modelName: "composition", rawFileName: "b1_beta")
        //b1_beta shape = (32,)
        layerData["b2_gamma"] = StyleModelData(modelName: "composition", rawFileName: "b2_gamma")
        //b2_gamma shape = (64,)
        layerData["b2_beta"] = StyleModelData(modelName: "composition", rawFileName: "b2_beta")
        //b2_beta shape = (64,)
        layerData["b3_gamma"] = StyleModelData(modelName: "composition", rawFileName: "b3_gamma")
        //b3_gamma shape = (128,)
        layerData["b3_beta"] = StyleModelData(modelName: "composition", rawFileName: "b3_beta")
        //b3_beta shape = (128,)
        layerData["c3_W"] = StyleModelData(modelName: "composition", rawFileName: "c3_W")
        //c3_W shape = (128, 64, 4, 4)
        layerData["c3_b"] = StyleModelData(modelName: "composition", rawFileName: "c3_b")
        //c3_b shape = (128,)
        layerData["c2_W"] = StyleModelData(modelName: "composition", rawFileName: "c2_W")
        //c2_W shape = (64, 32, 4, 4)
        layerData["c2_b"] = StyleModelData(modelName: "composition", rawFileName: "c2_b")
        //c2_b shape = (64,)
        layerData["c1_W"] = StyleModelData(modelName: "composition", rawFileName: "c1_W")
        //c1_W shape = (32, 3, 9, 9)
        layerData["c1_b"] = StyleModelData(modelName: "composition", rawFileName: "c1_b")
        //c1_b shape = (32,)
        layerData["d2_W"] = StyleModelData(modelName: "composition", rawFileName: "d2_W")
        //d2_W shape = (64, 32, 4, 4)
        layerData["d2_b"] = StyleModelData(modelName: "composition", rawFileName: "d2_b")
        //d2_b shape = (32,)
        layerData["d3_W"] = StyleModelData(modelName: "composition", rawFileName: "d3_W")
        //d3_W shape = (32, 3, 9, 9)
        layerData["d3_b"] = StyleModelData(modelName: "composition", rawFileName: "d3_b")
        //d3_b shape = (3,)
        layerData["d1_W"] = StyleModelData(modelName: "composition", rawFileName: "d1_W")
        //d1_W shape = (128, 64, 4, 4)
        layerData["d1_b"] = StyleModelData(modelName: "composition", rawFileName: "d1_b")
        //d1_b shape = (64,)
        // END PASTE

        self.device = device
        commandQueue = device.makeCommandQueue()

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
