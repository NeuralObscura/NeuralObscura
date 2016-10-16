//
//  FullyConnectedLayer.swift
//  NeuralObscura
//
//  Created by Edward Knox on 9/25/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class FullyConnectedLayer: CommandEncoder {
    init(
        device: MTLDevice,
        kernelSize: UInt,
        channelsIn: UInt,
        channelsOut: UInt,
        w: ParameterBuffer,
        b: ParameterBuffer,
        neuronFilter: MPSCNNNeuron? = MPSCNNNeuronSigmoid(),
        destinationFeatureChannelOffset: UInt = 0,
        useTemporary: Bool = true) {
        super.init(
            device: device,
            delegate: FullyConnectedLayerDelegate(
                device: device,
                kernelSize: kernelSize,
                channelsIn: channelsIn,
                channelsOut: channelsOut,
                w: w,
                b: b,
                neuronFilter: neuronFilter,
                destinationFeatureChannelOffset: destinationFeatureChannelOffset),
            useTemporary: useTemporary)
    }
}

class FullyConnectedLayerDelegate: CommandEncoderDelegate {
    let fullyConnected: MPSCNNFullyConnected
    
    init(
        device: MTLDevice,
        kernelSize: UInt,
        channelsIn: UInt,
        channelsOut: UInt,
        w: ParameterBuffer,
        b: ParameterBuffer,
        neuronFilter: MPSCNNNeuron? = MPSCNNNeuronSigmoid(),
        destinationFeatureChannelOffset: UInt = 0){
        
        // create appropriate convolution descriptor (in fully connected, stride is always 1)
        let convDesc = MPSCNNConvolutionDescriptor(kernelWidth: Int(kernelSize),
                                                   kernelHeight: Int(kernelSize),
                                                   inputFeatureChannels: Int(channelsIn),
                                                   outputFeatureChannels: Int(channelsOut),
                                                   neuronFilter: neuronFilter)
        
        // initialize the convolution layer by calling the parent's (MPSCNNFullyConnected's) initializer
        fullyConnected = MPSCNNFullyConnected.init(device: device,
                   convolutionDescriptor: convDesc,
                   kernelWeights: w.pointer(),
                   biasTerms: b.pointer(),
                   flags: MPSCNNConvolutionFlags.none)
        fullyConnected.destinationFeatureChannelOffset = Int(destinationFeatureChannelOffset)
    }
    
    func getDestinationImageDescriptor(sourceImage: MPSImage) -> MPSImageDescriptor {
        return MPSImageDescriptor(
            channelFormat: textureFormat,
            width: 1,
            height: 1,
            featureChannels: fullyConnected.outputFeatureChannels)
    }
    
    func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
        fullyConnected.encode(commandBuffer: commandBuffer, sourceImage: sourceImage, destinationImage: destinationImage)
    }
    
}
