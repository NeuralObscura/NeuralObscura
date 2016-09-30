//
//  ResidualBlock.swift
//  NeuralObscura
//
//  Created by Edward Knox on 9/25/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class ResidualBlock {
    let device: MTLDevice
    let c1, c2: ConvolutionLayer
    let b1, b2: BatchNormalizationLayer
    
    /**
     * A property to keep info from init time whether we will pad input image or not for use during encode call
     */
    fileprivate var padding = true
    
    init(
        device: MTLDevice,
        modelName: String,
        blockName: String,
        channelsIn: UInt,
        channelsOut: UInt,
        kernelSize: UInt = 3,
        stride: Int = 1) {
        
        self.device = device
        
        /* Load the block parameters */
        let c1_w = StyleModelData(modelName: modelName, rawFileName: blockName + "_c1_W")
        let c1_b = StyleModelData(modelName: modelName, rawFileName: blockName + "_c1_b")
        let c2_w = StyleModelData(modelName: modelName, rawFileName: blockName + "_c2_W")
        let c2_b = StyleModelData(modelName: modelName, rawFileName: blockName + "_c2_b")
        let b1_beta = StyleModelData(modelName: modelName, rawFileName: blockName + "_b1_beta")
        let b1_gamma = StyleModelData(modelName: modelName, rawFileName: blockName + "_b1_gamma")
        let b2_beta = StyleModelData(modelName: modelName, rawFileName: blockName + "_b2_beta")
        let b2_gamma = StyleModelData(modelName: modelName, rawFileName: blockName + "_b2_gamma")
        
        /* Init block encoders */
        c1 = ConvolutionLayer(
            device: device,
            kernelSize: kernelSize,
            channelsIn: channelsIn,
            channelsOut: channelsOut,
            w: c1_w,
            b: c1_b,
            relu: false,
            stride: stride)
        c2 = ConvolutionLayer(
            device: device,
            kernelSize: kernelSize,
            channelsIn: channelsOut,
            channelsOut: channelsOut,
            w: c2_w,
            b: c2_b,
            relu: false,
            stride: stride)
        b1 = BatchNormalizationLayer(device: device, channelsIn: channelsOut, beta: b1_beta, gamma: b1_gamma)
        b2 = BatchNormalizationLayer(device: device, channelsIn: channelsOut, beta: b2_beta, gamma: b2_gamma)
    }
    
    func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
        
        // select offset according to padding being used or not
        // if (padding) {
        //     let pad_along_height = ((destinationImage.height - 1) * strideInPixelsY + kernelSize - sourceImage.height)
        //     let pad_along_width  = ((destinationImage.width - 1) * strideInPixelsX + kernelSize - sourceImage.width)
        //     let pad_top = Int(pad_along_height / 2)
        //     let pad_left = Int(pad_along_width / 2)
        //     
        //     self.offset = MPSOffset(x: ((Int(kernelSize)/2) - pad_left), y: (Int(kernelHeight/2) - pad_top), z: 0)
        // } else {
        //     self.offset = MPSOffset(x: Int(kernelWidth)/2, y: Int(kernelSize)/2, z: 0)
        // }
        
        c1.encode(commandBuffer: commandBuffer, sourceImage: sourceImage, destinationImage: destinationImage)
        b1.encode(commandBuffer: commandBuffer, sourceImage: sourceImage, destinationImage: destinationImage)
        c2.encode(commandBuffer: commandBuffer, sourceImage: sourceImage, destinationImage: destinationImage)
    }
    
}
