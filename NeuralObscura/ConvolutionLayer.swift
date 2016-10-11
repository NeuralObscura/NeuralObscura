//
//  ConvolutionLayer.swift
//  NeuralObscura
//
//  Created by Edward Knox on 9/25/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class ConvolutionLayer: CommandEncoder {
    init(
        device: MTLDevice,
        kernelSize: UInt,
        channelsIn: UInt,
        channelsOut: UInt,
        w: StyleModelData,
        b: StyleModelData,
        relu: Bool,
        padding: Int = 0,
        stride: Int = 1,
        destinationFeatureChannelOffset: UInt = 0,
        groupNum: UInt = 1,
        useTemporary: Bool = true) {
        super.init(
            device: device,
            delegate: ConvolutionLayerDelegate(
                device: device,
                kernelSize: kernelSize,
                channelsIn: channelsIn,
                channelsOut: channelsOut,
                w: w,
                b: b,
                relu: relu,
                padding: padding,
                stride: stride,
                destinationFeatureChannelOffset: destinationFeatureChannelOffset,
                groupNum: groupNum),
            useTemporary: useTemporary)
    }
}

class ConvolutionLayerDelegate: CommandEncoderDelegate {
    let convolution: MPSCNNConvolution
    let padding: Int
    
    init(
        device: MTLDevice,
        kernelSize: UInt,
        channelsIn: UInt,
        channelsOut: UInt,
        w: StyleModelData,
        b: StyleModelData,
        relu: Bool,
        padding: Int = 0,
        stride: Int = 1,
        destinationFeatureChannelOffset: UInt = 0,
        groupNum: UInt = 1) {
        
        var neuronFilter: MPSCNNNeuron?

        if relu {
            neuronFilter = MPSCNNNeuronReLU(device: device, a: 0)
        }
        
        // create appropriate convolution descriptor with appropriate stride
        let convDesc = MPSCNNConvolutionDescriptor(kernelWidth: Int(kernelSize),
                                                   kernelHeight: Int(kernelSize),
                                                   inputFeatureChannels: Int(channelsIn),
                                                   outputFeatureChannels: Int(channelsOut),
                                                   neuronFilter: neuronFilter)
        convDesc.strideInPixelsX = stride
        convDesc.strideInPixelsY = stride
        
        assert((groupNum > 0), "Group size can't be less than 1")
        convDesc.groups = Int(groupNum)
        
        // initialize the convolution layer by calling the parent's (MPSCNNConvlution's) initializer
        convolution = MPSCNNConvolution(
            device: device,
            convolutionDescriptor: convDesc,
            kernelWeights: w.pointer(),
            biasTerms: b.pointer(),
            flags: MPSCNNConvolutionFlags.none)
        convolution.destinationFeatureChannelOffset = Int(destinationFeatureChannelOffset)
        convolution.edgeMode = .zero
        
        // set padding for calculation of offset during encode call
        self.padding = padding
    }
    
    func getDestinationImageDescriptor(sourceImage: MPSImage) -> MPSImageDescriptor {
        // TODO: This won't work for the first layer
        let inHeight = sourceImage.height
        let inWidth = sourceImage.width
        
        let kernelSize = convolution.kernelWidth
        let stride = convolution.strideInPixelsX
        let channelsOut = convolution.outputFeatureChannels

        let outHeight = ((inHeight + (2 * self.padding) - kernelSize) / stride) + 1
        let outWidth = ((inWidth + (2 * self.padding) - kernelSize) / stride) + 1

        return MPSImageDescriptor(channelFormat: textureFormat, width: outWidth, height: outHeight, featureChannels: channelsOut)
    }
    
    func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
        print("conv encode")
        // decrements sourceImage.readCount
        convolution.encode(commandBuffer: commandBuffer, sourceImage: sourceImage, destinationImage: destinationImage)
    }
}
