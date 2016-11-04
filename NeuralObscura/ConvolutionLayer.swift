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
        kernelSize: Int,
        channelsIn: Int,
        channelsOut: Int,
        w: ParameterBuffer,
        b: ParameterBuffer,
        relu: Bool = true,
        padding: Int = 0,
        stride: Int = 1,
        destinationFeatureChannelOffset: Int = 0,
        groupNum: Int = 1,
        outputType: CommandEncoderOutputType = CommandEncoderOutputType.debug) {
        super.init(
            delegate: ConvolutionLayerDelegate(
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
            outputType: outputType)
    }
}

class ConvolutionLayerDelegate: CommandEncoderDelegate {
    let convolution: MPSCNNConvolution
    let padding: Int
    
    init(
        kernelSize: Int,
        channelsIn: Int,
        channelsOut: Int,
        w: ParameterBuffer,
        b: ParameterBuffer,
        relu: Bool = true,
        padding: Int = 0,
        stride: Int = 1,
        destinationFeatureChannelOffset: Int = 0,
        groupNum: Int = 1) {
        self.padding = padding
        
        var neuronFilter: MPSCNNNeuron?
        if relu {
            neuronFilter = MPSCNNNeuronReLU(device: ShaderRegistry.getDevice(), a: 0)
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
            device: ShaderRegistry.getDevice(),
            convolutionDescriptor: convDesc,
            kernelWeights: w.pointer(),
            biasTerms: b.pointer(),
            flags: MPSCNNConvolutionFlags.none)
        convolution.destinationFeatureChannelOffset = Int(destinationFeatureChannelOffset)
        convolution.edgeMode = .zero
        convolution.offset = MPSOffset(x: 1 - padding, y: 1 - padding, z: 0)
    }
    
    func getDestinationImageDescriptor(sourceImage: MPSImage) -> MPSImageDescriptor {
        let inHeight = sourceImage.height
        let inWidth = sourceImage.width
        
        let kernelSize = convolution.kernelWidth
        let stride = convolution.strideInPixelsX
        let channelsOut = convolution.outputFeatureChannels

        let outHeight = ((inHeight + (2 * self.padding) - kernelSize) / stride) + 1
        let outWidth = ((inWidth + (2 * self.padding) - kernelSize) / stride) + 1
        let descriptor = MPSImageDescriptor(
            channelFormat: textureFormat,
            width: outWidth,
            height: outHeight,
            featureChannels: channelsOut)

        return descriptor
    }
    
    func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
        // decrements sourceImage.readCount
        convolution.encode(commandBuffer: commandBuffer, sourceImage: sourceImage, destinationImage: destinationImage)
    }
}
