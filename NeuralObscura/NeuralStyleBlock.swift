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


class NeuralStyleBlock: CommandEncoder {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let useTemporary: Bool
    let layerData = [String: StyleModelData]()
    
    var top: CommandEncoder?
    var bottom: CommandEncoder?
    
    let firstLayer: CommandEncoder?
    var lastLayer: CommandEncoder?
    
    let c1, c2, c3: ConvolutionLayer
    let b1, b2, b3, b4, b5: BatchNormalizationLayer
    let r1, r2, r3, r4, r5: ResidualBlock
    let d1, d2, d3: DeconvolutionLayer

    init(device: MTLDevice, commandQueue: MTLCommandQueue, modelParams: [String: StyleModelData], useTemporary: Bool = true) {
        self.device = device
        self.commandQueue = commandQueue
        self.useTemporary = useTemporary
        
        /* Init model encoders */
        
        // c1=L.Convolution2D(3, 32, 9, stride=1, pad=4),
        c1 = ConvolutionLayer(
            device: device,
            kernelSize: 9,
            channelsIn: 3,
            channelsOut: 32,
            w: modelParams["c1_W"]!,
            b: modelParams["c1_b"]!,
            stride: 1)
        firstLayer = c1
        
        // b1=L.BatchNormalization(32),
        
        // c2=L.Convolution2D(32, 64, 4, stride=2, pad=1),
        c2 = ConvolutionLayer(
            device: device,
            kernelSize: 4,
            channelsIn: 32,
            channelsOut: 64,
            w: modelParams["c2_W"]!,
            b: modelParams["c2_b"]!,
            stride: 2)
        
        // b2=L.BatchNormalization(64),
        
        // c3=L.Convolution2D(64, 128, 4,stride=2, pad=1),
        c3 = ConvolutionLayer(
            device: device,
            kernelSize: 4,
            channelsIn: 64,
            channelsOut: 128,
            w: modelParams["c3_W"]!,
            b: modelParams["c3_b"]!,
            stride: 2)
        
        // b3=L.BatchNormalization(128),
        
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
            pad: 1,
            stride: 2)
        
        // b4=L.BatchNormalization(64),
        b4 = BatchNormalizationLayer(device: device, channelsIn: 64, beta: modelParams["b4_beta"]!, gamma: modelParams["b4_gamma"]!)
        
        // d2=L.Deconvolution2D(64, 32, 4, stride=2, pad=1),
        d2 = DeconvolutionLayer(
            device: device,
            channelsIn: 64,
            channelsOut: 32,
            kernelSize: 4,
            w: modelParams["d2_W"]!,
            b: modelParams["d2_b"]!,
            pad: 1,
            stride: 2)
        
        // b5=L.BatchNormalization(32),
        b5 = BatchNormalizationLayer(device: device, channelsIn: 32, beta: modelParams["b5_beta"]!, gamma: modelParams["b5_gamma"]!)
        
        // d3=L.Deconvolution2D(32, 3, 9, stride=1, pad=4),
        d3 = DeconvolutionLayer(
            device: device,
            channelsIn: 32,
            channelsOut: 3,
            kernelSize: 9,
            w: modelParams["d3_W"]!,
            b: modelParams["d3_b"]!,
            pad: 4,
            stride: 1)
        
        // TODO: Init last tanh layer
    }
    
    func chain(_ top: CommandEncoder) -> CommandEncoder {
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
        
        self.top = top
        top.setBottom(self)
        return self
    }
    
    func getDestinationImageDescriptor() -> MPSImageDescriptor {
        // TODO: Figure out output dimensions
        return lastLayer!.getDestinationImageDescriptor()
    }
    
    func getDestinationImage(commandBuffer: MTLCommandBuffer) -> MPSImage {
        let destDesc = getDestinationImageDescriptor()
        switch useTemporary {
        case true: return MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: destDesc)
        case false: return MPSImage(device: device, imageDescriptor: destDesc)
        }
    }
    
    func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage) -> MPSImage {
        let destinationImage = getDestinationImage(commandBuffer: commandBuffer)
        
        let modelLayersOutput = firstLayer.encode(commandBuffer: commandBuffer, sourceImage: sourceImage)
        
        let modelOutput = // TODO: Sum residual output with input (something with ^ modelLayersOutput)
        
        switch bottom { // TODO: Considering removing this switch in all command buffers and just making an OutputLayer necessary.
        case .some: return bottom!.encode(commandBuffer: commandBuffer, sourceImage: residualOutputImage)
        case .none: return modelOutput
        }
    }
}
