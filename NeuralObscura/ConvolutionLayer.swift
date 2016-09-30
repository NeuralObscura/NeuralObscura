//
//  ConvolutionLayer.swift
//  NeuralObscura
//
//  Created by Edward Knox on 9/25/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

/**
 This depends on MetalPerformanceShaders.framework
 
 The SlimMPSCNNConvolution is a wrapper class around MPSCNNConvolution used to encapsulate:
 - making an MPSCNNConvolutionDescriptor,
 - adding network parameters (weights and bias binaries by memory mapping the binaries)
 - getting our convolution layer
 */

import Foundation
import MetalPerformanceShaders

class ConvolutionLayer: MPSCNNConvolution {
    
    /**
     A property to keep info from init time whether we will pad input image or not for use during encode call
     */
    fileprivate var padding = true
    
    init(
        device: MTLDevice,
        kernelSize: UInt,
        channelsIn: UInt,
        channelsOut: UInt,
        w: StyleModelData,
        b: StyleModelData,
        neuronFilter: MPSCNNNeuron? = MPSCNNNeuronReLU(),
        padding willPad: Bool = true,
        stride: Int = 1,
        destinationFeatureChannelOffset: UInt = 0,
        groupNum: UInt = 1) {
        
        
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
        super.init(device: device,
                   convolutionDescriptor: convDesc,
                   kernelWeights: w.pointer(),
                   biasTerms: b.pointer(),
                   flags: MPSCNNConvolutionFlags.none)
        self.destinationFeatureChannelOffset = Int(destinationFeatureChannelOffset)
        
        
        // set padding for calculation of offset during encode call
        padding = willPad
    }
    
    override func encode(
        commandBuffer: MTLCommandBuffer,
        sourceImage: MPSImage,
        destinationImage: MPSImage) {
        
        // select offset according to padding being used or not
        if(padding){
            let pad_along_height = ((destinationImage.height - 1) * strideInPixelsY + kernelHeight - sourceImage.height)
            let pad_along_width  = ((destinationImage.width - 1) * strideInPixelsX + kernelWidth - sourceImage.width)
            let pad_top = Int(pad_along_height / 2)
            let pad_left = Int(pad_along_width / 2)
            
            self.offset = MPSOffset(x: ((Int(kernelWidth)/2) - pad_left), y: (Int(kernelHeight/2) - pad_top), z: 0)
        }
        else{
            self.offset = MPSOffset(x: Int(kernelWidth)/2, y: Int(kernelHeight)/2, z: 0)
        }
        super.encode(commandBuffer: commandBuffer, sourceImage: sourceImage, destinationImage: destinationImage)
    }
    
    
    // private func makeConv(device: MTLDevice,
    //                       inDepth: Int,
    //                       outDepth: Int,
    //                       weights: UnsafePointer<Float>,
    //                       bias: UnsafePointer<Float>,
    //                       stride: Int) -> MPSCNNConvolution {
    //     
    //     // All VGGNet conv layers use a 3x3 kernel with stride 1.
    //     let desc = MPSCNNConvolutionDescriptor(kernelWidth: 3,
    //                                            kernelHeight: 3,
    //                                            inputFeatureChannels: inDepth,
    //                                            outputFeatureChannels: outDepth,
    //                                            neuronFilter: nil)
    //     desc.strideInPixelsX = stride
    //     desc.strideInPixelsY = stride
    //     
    //     let conv = MPSCNNConvolution(device: device,
    //                                  convolutionDescriptor: desc,
    //                                  kernelWeights: weights,
    //                                  biasTerms: bias,
    //                                  flags: MPSCNNConvolutionFlags.none)
    //     
    //     // To preserve the width and height between conv layers, VGGNet assumes one
    //     // pixel of padding around the edges. Metal apparently has no problem reading
    //     // outside the source image, so we don't have to do anything special here.
    //     conv.edgeMode = .zero
    //     
    //     return conv
    // }
    
}


