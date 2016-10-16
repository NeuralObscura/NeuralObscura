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
    let useTemporary: Bool
    let debug: Bool
    var modelParams = [String: ParameterBuffer]()
    
    let c1, c2, c3: ConvolutionLayer
    let b1, b2, b3, b4, b5: BatchNormalizationLayer
    let r1, r2, r3, r4, r5: ResidualBlock
    let d1, d2, d3: DeconvolutionLayer
    let modelHandle: CommandEncoder

    init(device: MTLDevice,
         modelName: String,
         useTemporary: Bool = true,
         debug: Bool = true) {
        self.device = device
        self.useTemporary = useTemporary
        self.debug = debug

        /* Load model parameters */
        modelParams["r4_c2_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "r4_c2_W")
        //r4_c2_W shape = (128, 3, 3, 128)
        modelParams["r4_c2_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "r4_c2_b")
        //r4_c2_b shape = (128,)
        modelParams["r4_c1_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "r4_c1_W")
        //r4_c1_W shape = (128, 3, 3, 128)
        modelParams["r4_c1_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "r4_c1_b")
        //r4_c1_b shape = (128,)
        modelParams["r5_c2_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "r5_c2_W")
        //r5_c2_W shape = (128, 3, 3, 128)
        modelParams["r5_c2_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "r5_c2_b")
        //r5_c2_b shape = (128,)
        modelParams["r5_c1_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "r5_c1_W")
        //r5_c1_W shape = (128, 3, 3, 128)
        modelParams["r5_c1_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "r5_c1_b")
        //r5_c1_b shape = (128,)
        modelParams["r1_c2_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "r1_c2_W")
        //r1_c2_W shape = (128, 3, 3, 128)
        modelParams["r1_c2_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "r1_c2_b")
        //r1_c2_b shape = (128,)
        modelParams["r1_c1_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "r1_c1_W")
        //r1_c1_W shape = (128, 3, 3, 128)
        modelParams["r1_c1_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "r1_c1_b")
        //r1_c1_b shape = (128,)
        modelParams["r2_c2_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "r2_c2_W")
        //r2_c2_W shape = (128, 3, 3, 128)
        modelParams["r2_c2_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "r2_c2_b")
        //r2_c2_b shape = (128,)
        modelParams["r2_c1_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "r2_c1_W")
        //r2_c1_W shape = (128, 3, 3, 128)
        modelParams["r2_c1_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "r2_c1_b")
        //r2_c1_b shape = (128,)
        modelParams["r3_c2_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "r3_c2_W")
        //r3_c2_W shape = (128, 3, 3, 128)
        modelParams["r3_c2_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "r3_c2_b")
        //r3_c2_b shape = (128,)
        modelParams["r3_c1_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "r3_c1_W")
        //r3_c1_W shape = (128, 3, 3, 128)
        modelParams["r3_c1_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "r3_c1_b")
        //r3_c1_b shape = (128,)
        modelParams["c3_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "c3_W")
        //c3_W shape = (128, 4, 4, 64)
        modelParams["c3_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "c3_b")
        //c3_b shape = (128,)
        modelParams["c2_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "c2_W")
        //c2_W shape = (64, 4, 4, 32)
        modelParams["c2_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "c2_b")
        //c2_b shape = (64,)
        modelParams["c1_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "c1_W")
        //c1_W shape = (32, 9, 9, 3)
        modelParams["c1_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "c1_b")
        //c1_b shape = (32,)
        modelParams["d2_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "d2_W")
        //d2_W shape = (64, 4, 4, 32)
        modelParams["d2_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "d2_b")
        //d2_b shape = (32,)
        modelParams["d3_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "d3_W")
        //d3_W shape = (32, 9, 9, 3)
        modelParams["d3_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "d3_b")
        //d3_b shape = (3,)
        modelParams["d1_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "d1_W")
        //d1_W shape = (128, 4, 4, 64)
        modelParams["d1_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "d1_b")
        //d1_b shape = (64,)


        /* Init model encoders */
        // c1=L.Convolution2D(3, 32, 9, stride=1, pad=4),
        c1 = ConvolutionLayer(
            device: device,
            kernelSize: 9,
            channelsIn: 3,
            channelsOut: 32,
            w: modelParams["c1_W"]!,
            b: modelParams["c1_b"]!,
            relu: true,
            padding: 4,
            stride: 1)

        // b1=L.BatchNormalization(32),
        b1 = BatchNormalizationLayer(
            device: device,
            channelsIn: 32,
            beta: modelParams["b1_beta"]!,
            gamma: modelParams["b1_gamma"]!)

        // c2=L.Convolution2D(32, 64, 4, stride=2, pad=1),
        c2 = ConvolutionLayer(
            device: device,
            kernelSize: 4,
            channelsIn: 32,
            channelsOut: 64,
            w: modelParams["c2_W"]!,
            b: modelParams["c2_b"]!,
            relu: true,
            padding: 1,
            stride: 2)

        // b2=L.BatchNormalization(64),
        b2 = BatchNormalizationLayer(
            device: device,
            channelsIn: 64,
            beta: modelParams["b2_beta"]!,
            gamma: modelParams["b2_gamma"]!)

        // c3=L.Convolution2D(64, 128, 4,stride=2, pad=1),
        c3 = ConvolutionLayer(
            device: device,
            kernelSize: 4,
            channelsIn: 64,
            channelsOut: 128,
            w: modelParams["c3_W"]!,
            b: modelParams["c3_b"]!,
            relu: true,
            padding: 1,
            stride: 2)

        // b3=L.BatchNormalization(128),
        b3 = BatchNormalizationLayer(
            device: device,
            channelsIn: 128,
            beta: modelParams["b3_beta"]!,
            gamma: modelParams["b3_gamma"]!)

        // r1=ResidualBlock(128, 128),
        r1 = ResidualBlock(
            device: device,
            modelParams: modelParams,
            blockName: "r1",
            channelsIn: 128,
            channelsOut: 128)

        // r2=ResidualBlock(128, 128),
        r2 = ResidualBlock(
            device: device,
            modelParams: modelParams,
            blockName: "r2",
            channelsIn: 128,
            channelsOut: 128)

        // r3=ResidualBlock(128, 128),
        r3 = ResidualBlock(
            device: device,
            modelParams: modelParams,
            blockName: "r3",
            channelsIn: 128,
            channelsOut: 128)

        // r4=ResidualBlock(128, 128),
        r4 = ResidualBlock(
            device: device,
            modelParams: modelParams,
            blockName: "r4",
            channelsIn: 128,
            channelsOut: 128)

        // r5=ResidualBlock(128, 128),
        r5 = ResidualBlock(
            device: device,
            modelParams: modelParams,
            blockName: "r5",
            channelsIn: 128,
            channelsOut: 128)

        // d1=L.Deconvolution2D(128, 64, 4, stride=2, pad=1),
        d1 = DeconvolutionLayer(
            device: device,
            channelsIn: 128,
            channelsOut: 64,
            kernelSize: 4,
            w: modelParams["d1_W"]!,
            b: modelParams["d1_b"]!,
            padding: true,
            stride: 2)

        // b4=L.BatchNormalization(64),
        b4 = BatchNormalizationLayer(
            device: device,
            channelsIn: 64,
            beta: modelParams["b4_beta"]!,
            gamma: modelParams["b4_gamma"]!)

        // d2=L.Deconvolution2D(64, 32, 4, stride=2, pad=1),
        d2 = DeconvolutionLayer(
            device: device,
            channelsIn: 64,
            channelsOut: 32,
            kernelSize: 4,
            w: modelParams["d2_W"]!,
            b: modelParams["d2_b"]!,
            stride: 2)

        // b5=L.BatchNormalization(32),
        b5 = BatchNormalizationLayer(
            device: device,
            channelsIn: 32,
            beta: modelParams["b5_beta"]!,
            gamma: modelParams["b5_gamma"]!)

        // d3=L.Deconvolution2D(32, 3, 9, stride=1, pad=4),
        d3 = DeconvolutionLayer(
            device: device,
            channelsIn: 32,
            channelsOut: 3,
            kernelSize: 9,
            w: modelParams["d3_W"]!,
            b: modelParams["d3_b"]!,
            stride: 1)

        // TODO: Init last tanh layer

        /* Chain model encoders together */
        var h: CommandEncoder

        // h = self.b1(F.elu(self.c1(top)), test=test)
        h = b1.chain(c1)

        // h = self.b2(F.elu(self.c2(h)), test=test)
        h = b2.chain(c2.chain(h))

        // h = self.b3(F.elu(self.c3(h)), test=test)
        h = b3.chain(c3.chain(h))

        // h = self.r1(h, test=test)
        h = r1.chain(h)

        // h = self.r2(h, test=test)
        h = r2.chain(h)

        // h = self.r3(h, test=test)
        h = r3.chain(h)

        // h = self.r4(h, test=test)
        h = r4.chain(h)

        // h = self.r5(h, test=test)
        h = r5.chain(h)

        // h = self.b4(F.elu(self.d1(h)), test=test)
        h = b4.chain(d1.chain(h))

        // h = self.b5(F.elu(self.d2(h)), test=test)
        h = b5.chain(d2.chain(h))

        // y = self.d3(h)
        h = d3.chain(h)

        // return (F.tanh(y)+1)*127.5
        // TODO: chain last tanh layer
        
        modelHandle = h
    }

    func forward(commandQueue: MTLCommandQueue, sourceImage: MPSImage) -> MPSImage {
        var outputImage: MPSImage? = nil

        if (debug) {
            print("Four corners as MPSImage; inital values")
            sourceImage.fourCorners()
            print("---------------------------")
        }

        autoreleasepool {
            let commandBuffer = commandQueue.makeCommandBuffer()
            outputImage = modelHandle.execute(commandBuffer: commandBuffer, sourceImage: sourceImage)
        }

        return outputImage!
    }
}
