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
        padding willPad: Bool = true,
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
                padding: willPad,
                stride: stride,
                destinationFeatureChannelOffset: destinationFeatureChannelOffset,
                groupNum: groupNum),
            useTemporary: useTemporary)
    }
}

class ConvolutionLayerDelegate: CommandEncoderDelegate {
    let convolution: MPSCNNConvolution
    
    /**
     * A property to keep info from init time whether we will pad
     input image or not for use during encode call
     */
    fileprivate var padding = true
    
    init(
        device: MTLDevice,
        kernelSize: UInt,
        channelsIn: UInt,
        channelsOut: UInt,
        w: StyleModelData,
        b: StyleModelData,
        relu: Bool,
        padding willPad: Bool = true,
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
        padding = willPad
    }
    
    func getDestinationImageDescriptor(sourceImage: MPSImage?) -> MPSImageDescriptor {
        // TODO: This won't work for the first layer
        let inHeight = sourceImage?.height
        let inWidth = sourceImage?.width
        let channelsIn = sourceImage?.featureChannels
        
        let kernelSize = convolution.kernelWidth
        let stride = convolution.strideInPixelsX
        let channelsOut = convolution.outputFeatureChannels
//        let padding = // TODO: Figure out how the hell padding workds
        // TODO: Calculate shape of output
        return MPSImageDescriptor(channelFormat: textureFormat, width: 10, height: 10, featureChannels: channelsOut)
    }
    
    func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
        // select offset according to padding being used or not
        if(padding) {
            let pad_along_height = ((destinationImage.height - 1) * convolution.strideInPixelsY + convolution.kernelHeight - sourceImage.height)
            let pad_along_width  = ((destinationImage.width - 1) * convolution.strideInPixelsX + convolution.kernelWidth - sourceImage.width)
            let pad_top = Int(pad_along_height / 2)
            let pad_left = Int(pad_along_width / 2)
            
            convolution.offset = MPSOffset(x: ((Int(convolution.kernelWidth)/2) - pad_left), y: (Int(convolution.kernelHeight/2) - pad_top), z: 0)
        } else {
            convolution.offset = MPSOffset(x: Int(convolution.kernelWidth)/2, y: Int(convolution.kernelHeight)/2, z: 0)
        }
        convolution.encode(commandBuffer: commandBuffer, sourceImage: sourceImage, destinationImage: destinationImage)
    }
}
