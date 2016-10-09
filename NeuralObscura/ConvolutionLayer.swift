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
        relu: Bool = true,
        padding: UInt = 0,
        stride: UInt = 1,
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
        relu: Bool = true,
        padding: UInt = 0,
        stride: UInt = 1,
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
        convDesc.strideInPixelsX = Int(stride)
        convDesc.strideInPixelsY = Int(stride)
        
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
        self.padding = Int(padding)
    }
    
    func getDestinationImageDescriptor(sourceImage: MPSImage?) -> MPSImageDescriptor {
        let heightIn = sourceImage!.height
        let widthIn = sourceImage!.width
        
        let kernelSize = convolution.kernelWidth
        let stride = convolution.strideInPixelsX
        let channelsOut = convolution.outputFeatureChannels
        
        let heightOut = ((heightIn + 2 * padding - kernelSize) / stride) + 1
        let widthOut = ((widthIn + 2 * padding - kernelSize) / stride) + 1
        
        return MPSImageDescriptor(
            channelFormat: textureFormat,
            width: widthOut,
            height: heightOut,
            featureChannels: channelsOut)
    }
    
    func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
        
        print("conv encode")
        convolution.offset = MPSOffset(
            x: convolution.kernelWidth / 2 - padding,
            y: convolution.kernelHeight / 2 - padding,
            z: 0)
        
        // Decrements sourceImage.readCount
        convolution.encode(commandBuffer: commandBuffer, sourceImage: sourceImage, destinationImage: destinationImage)
    }
}
