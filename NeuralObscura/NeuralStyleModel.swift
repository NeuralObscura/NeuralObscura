//
//  CompositionModel.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 9/20/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders
import MetalKit


class NeuralStyleModel {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let modelName: String
    var c1, c2, c3: ConvolutionLayer?
    var b1, b2, b3, b4, b5: BatchNormalizationLayer?
    var r1, r2, r3, r4, r5: ResidualBlock?
    var d1, d2, d3: DeconvolutionLayer?

    init(device: MTLDevice, modelName: String = "composition") {
        self.device = device
        commandQueue = device.makeCommandQueue()
        self.modelName = modelName
        
        // c1=L.Convolution2D(3, 32, 9, stride=1, pad=4),
        c1 = ConvolutionLayer(
            device: device,
            layerPrefix: modelName + "/c1",
            inputFeatureChannels: 3,
            outputFeatureChannels: 32,
            stride: 1)
        
        // c2=L.Convolution2D(32, 64, 4, stride=2, pad=1),
        c2 = ConvolutionLayer(
            device: device,
            layerPrefix: modelName + "/c2",
            inputFeatureChannels: 32,
            outputFeatureChannels: 64,
            stride: 2)
        
        // c3=L.Convolution2D(64, 128, 4,stride=2, pad=1),
        c3 = ConvolutionLayer(
            device: device,
            layerPrefix: modelName + "/c2",
            inputFeatureChannels: 64,
            outputFeatureChannels: 128,
            stride: 2)
        
        // r1=ResidualBlock(128, 128),
        r1 = ResidualBlock(
            device: device,
            layerPrefix: "r1",
            channelsIn: 128,
            channelsOut: 128)
        
        // r2=ResidualBlock(128, 128),
        r2 = ResidualBlock(
            device: device,
            layerPrefix: "r2",
            channelsIn: 128,
            channelsOut: 128)
        
        // r3=ResidualBlock(128, 128),
        r3 = ResidualBlock(
            device: device,
            layerPrefix: "r3",
            channelsIn: 128,
            channelsOut: 128)
        
        // r4=ResidualBlock(128, 128),
        r4 = ResidualBlock(
            device: device,
            layerPrefix: "r4",
            channelsIn: 128,
            channelsOut: 128)
        
        // r5=ResidualBlock(128, 128),
        r5 = ResidualBlock(
            device: device,
            layerPrefix: "r5",
            channelsIn: 128,
            channelsOut: 128)
        
        // d1=L.Deconvolution2D(128, 64, 4, stride=2, pad=1),
        // d2=L.Deconvolution2D(64, 32, 4, stride=2, pad=1),
        // d3=L.Deconvolution2D(32, 3, 9, stride=1, pad=4),
        // b1=L.BatchNormalization(32),
        // b2=L.BatchNormalization(64),
        // b3=L.BatchNormalization(128),
        // b4=L.BatchNormalization(64),
        // b5=L.BatchNormalization(32),
        
        // Keeping the following comment around for now
        
        // layerData["r4_c2_W"] = StyleModelData(modelName: "composition", rawFileName: "r4_c2_W")
        // //r4_c2_W shape = (128, 128, 3, 3)
        // layerData["r4_c2_b"] = StyleModelData(modelName: "composition", rawFileName: "r4_c2_b")
        // //r4_c2_b shape = (128,)
        // layerData["r4_c1_W"] = StyleModelData(modelName: "composition", rawFileName: "r4_c1_W")
        // //r4_c1_W shape = (128, 128, 3, 3)
        // layerData["r4_c1_b"] = StyleModelData(modelName: "composition", rawFileName: "r4_c1_b")
        // //r4_c1_b shape = (128,)
        // layerData["r4_b1_gamma"] = StyleModelData(modelName: "composition", rawFileName: "r4_b1_gamma")
        // //r4_b1_gamma shape = (128,)
        // layerData["r4_b1_beta"] = StyleModelData(modelName: "composition", rawFileName: "r4_b1_beta")
        // //r4_b1_beta shape = (128,)
        // layerData["r4_b2_gamma"] = StyleModelData(modelName: "composition", rawFileName: "r4_b2_gamma")
        // //r4_b2_gamma shape = (128,)
        // layerData["r4_b2_beta"] = StyleModelData(modelName: "composition", rawFileName: "r4_b2_beta")
        // //r4_b2_beta shape = (128,)
        // layerData["r5_c2_W"] = StyleModelData(modelName: "composition", rawFileName: "r5_c2_W")
        // //r5_c2_W shape = (128, 128, 3, 3)
        // layerData["r5_c2_b"] = StyleModelData(modelName: "composition", rawFileName: "r5_c2_b")
        // //r5_c2_b shape = (128,)
        // layerData["r5_c1_W"] = StyleModelData(modelName: "composition", rawFileName: "r5_c1_W")
        // //r5_c1_W shape = (128, 128, 3, 3)
        // layerData["r5_c1_b"] = StyleModelData(modelName: "composition", rawFileName: "r5_c1_b")
        // //r5_c1_b shape = (128,)
        // layerData["r5_b1_gamma"] = StyleModelData(modelName: "composition", rawFileName: "r5_b1_gamma")
        // //r5_b1_gamma shape = (128,)
        // layerData["r5_b1_beta"] = StyleModelData(modelName: "composition", rawFileName: "r5_b1_beta")
        // //r5_b1_beta shape = (128,)
        // layerData["r5_b2_gamma"] = StyleModelData(modelName: "composition", rawFileName: "r5_b2_gamma")
        // //r5_b2_gamma shape = (128,)
        // layerData["r5_b2_beta"] = StyleModelData(modelName: "composition", rawFileName: "r5_b2_beta")
        // //r5_b2_beta shape = (128,)
        // layerData["r1_c2_W"] = StyleModelData(modelName: "composition", rawFileName: "r1_c2_W")
        // //r1_c2_W shape = (128, 128, 3, 3)
        // layerData["r1_c2_b"] = StyleModelData(modelName: "composition", rawFileName: "r1_c2_b")
        // //r1_c2_b shape = (128,)
        // layerData["r1_c1_W"] = StyleModelData(modelName: "composition", rawFileName: "r1_c1_W")
        // //r1_c1_W shape = (128, 128, 3, 3)
        // layerData["r1_c1_b"] = StyleModelData(modelName: "composition", rawFileName: "r1_c1_b")
        // //r1_c1_b shape = (128,)
        // layerData["r1_b1_gamma"] = StyleModelData(modelName: "composition", rawFileName: "r1_b1_gamma")
        // //r1_b1_gamma shape = (128,)
        // layerData["r1_b1_beta"] = StyleModelData(modelName: "composition", rawFileName: "r1_b1_beta")
        // //r1_b1_beta shape = (128,)
        // layerData["r1_b2_gamma"] = StyleModelData(modelName: "composition", rawFileName: "r1_b2_gamma")
        // //r1_b2_gamma shape = (128,)
        // layerData["r1_b2_beta"] = StyleModelData(modelName: "composition", rawFileName: "r1_b2_beta")
        // //r1_b2_beta shape = (128,)
        // layerData["r2_c2_W"] = StyleModelData(modelName: "composition", rawFileName: "r2_c2_W")
        // //r2_c2_W shape = (128, 128, 3, 3)
        // layerData["r2_c2_b"] = StyleModelData(modelName: "composition", rawFileName: "r2_c2_b")
        // //r2_c2_b shape = (128,)
        // layerData["r2_c1_W"] = StyleModelData(modelName: "composition", rawFileName: "r2_c1_W")
        // //r2_c1_W shape = (128, 128, 3, 3)
        // layerData["r2_c1_b"] = StyleModelData(modelName: "composition", rawFileName: "r2_c1_b")
        // //r2_c1_b shape = (128,)
        // layerData["r2_b1_gamma"] = StyleModelData(modelName: "composition", rawFileName: "r2_b1_gamma")
        // //r2_b1_gamma shape = (128,)
        // layerData["r2_b1_beta"] = StyleModelData(modelName: "composition", rawFileName: "r2_b1_beta")
        // //r2_b1_beta shape = (128,)
        // layerData["r2_b2_gamma"] = StyleModelData(modelName: "composition", rawFileName: "r2_b2_gamma")
        // //r2_b2_gamma shape = (128,)
        // layerData["r2_b2_beta"] = StyleModelData(modelName: "composition", rawFileName: "r2_b2_beta")
        // //r2_b2_beta shape = (128,)
        // layerData["r3_c2_W"] = StyleModelData(modelName: "composition", rawFileName: "r3_c2_W")
        // //r3_c2_W shape = (128, 128, 3, 3)
        // layerData["r3_c2_b"] = StyleModelData(modelName: "composition", rawFileName: "r3_c2_b")
        // //r3_c2_b shape = (128,)
        // layerData["r3_c1_W"] = StyleModelData(modelName: "composition", rawFileName: "r3_c1_W")
        // //r3_c1_W shape = (128, 128, 3, 3)
        // layerData["r3_c1_b"] = StyleModelData(modelName: "composition", rawFileName: "r3_c1_b")
        // //r3_c1_b shape = (128,)
        // layerData["r3_b1_gamma"] = StyleModelData(modelName: "composition", rawFileName: "r3_b1_gamma")
        // //r3_b1_gamma shape = (128,)
        // layerData["r3_b1_beta"] = StyleModelData(modelName: "composition", rawFileName: "r3_b1_beta")
        // //r3_b1_beta shape = (128,)
        // layerData["r3_b2_gamma"] = StyleModelData(modelName: "composition", rawFileName: "r3_b2_gamma")
        // //r3_b2_gamma shape = (128,)
        // layerData["r3_b2_beta"] = StyleModelData(modelName: "composition", rawFileName: "r3_b2_beta")
        // //r3_b2_beta shape = (128,)
        // layerData["b4_gamma"] = StyleModelData(modelName: "composition", rawFileName: "b4_gamma")
        // //b4_gamma shape = (64,)
        // layerData["b4_beta"] = StyleModelData(modelName: "composition", rawFileName: "b4_beta")
        // //b4_beta shape = (64,)
        // layerData["b5_gamma"] = StyleModelData(modelName: "composition", rawFileName: "b5_gamma")
        // //b5_gamma shape = (32,)
        // layerData["b5_beta"] = StyleModelData(modelName: "composition", rawFileName: "b5_beta")
        // //b5_beta shape = (32,)
        // layerData["b1_gamma"] = StyleModelData(modelName: "composition", rawFileName: "b1_gamma")
        // //b1_gamma shape = (32,)
        // layerData["b1_beta"] = StyleModelData(modelName: "composition", rawFileName: "b1_beta")
        // //b1_beta shape = (32,)
        // layerData["b2_gamma"] = StyleModelData(modelName: "composition", rawFileName: "b2_gamma")
        // //b2_gamma shape = (64,)
        // layerData["b2_beta"] = StyleModelData(modelName: "composition", rawFileName: "b2_beta")
        // //b2_beta shape = (64,)
        // layerData["b3_gamma"] = StyleModelData(modelName: "composition", rawFileName: "b3_gamma")
        // //b3_gamma shape = (128,)
        // layerData["b3_beta"] = StyleModelData(modelName: "composition", rawFileName: "b3_beta")
        // //b3_beta shape = (128,)
        // layerData["c3_W"] = StyleModelData(modelName: "composition", rawFileName: "c3_W")
        // //c3_W shape = (128, 64, 4, 4)
        // layerData["c3_b"] = StyleModelData(modelName: "composition", rawFileName: "c3_b")
        // //c3_b shape = (128,)
        // layerData["c2_W"] = StyleModelData(modelName: "composition", rawFileName: "c2_W")
        // //c2_W shape = (64, 32, 4, 4)
        // layerData["c2_b"] = StyleModelData(modelName: "composition", rawFileName: "c2_b")
        // //c2_b shape = (64,)
        // layerData["c1_W"] = StyleModelData(modelName: "composition", rawFileName: "c1_W")
        // //c1_W shape = (32, 3, 9, 9)
        // layerData["c1_b"] = StyleModelData(modelName: "composition", rawFileName: "c1_b")
        // //c1_b shape = (32,)
        // layerData["d2_W"] = StyleModelData(modelName: "composition", rawFileName: "d2_W")
        // //d2_W shape = (64, 32, 4, 4)
        // layerData["d2_b"] = StyleModelData(modelName: "composition", rawFileName: "d2_b")
        // //d2_b shape = (32,)
        // layerData["d3_W"] = StyleModelData(modelName: "composition", rawFileName: "d3_W")
        // //d3_W shape = (32, 3, 9, 9)
        // layerData["d3_b"] = StyleModelData(modelName: "composition", rawFileName: "d3_b")
        // //d3_b shape = (3,)
        // layerData["d1_W"] = StyleModelData(modelName: "composition", rawFileName: "d1_W")
        // //d1_W shape = (128, 64, 4, 4)
        // layerData["d1_b"] = StyleModelData(modelName: "composition", rawFileName: "d1_b")
        // //d1_b shape = (64,)
    }
    
    func forward(inputCgImage: CGImage) -> CGImage {
        // note the configurable options
        let inputMtlImage = MTKTextureLoader.newTexture(with: inputCgImage, options: nil)
        // h = self.b1(F.elu(self.c1(x)), test=test)
        // h = self.b2(F.elu(self.c2(h)), test=test)
        // h = self.b3(F.elu(self.c3(h)), test=test)
        // h = self.r1(h, test=test)
        // h = self.r2(h, test=test)
        // h = self.r3(h, test=test)
        // h = self.r4(h, test=test)
        // h = self.r5(h, test=test)
        // h = self.b4(F.elu(self.d1(h)), test=test)
        // h = self.b5(F.elu(self.d2(h)), test=test)
        // y = self.d3(h)
        // return (F.tanh(y)+1)*127.5
    }
}
