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
        kernelSize: Int,
        channelsIn: Int,
        channelsOut: Int,
        w: ParameterBuffer,
        b: ParameterBuffer,
        neuronFilter: MPSCNNNeuron? = MPSCNNNeuronSigmoid(),
        destinationFeatureChannelOffset: Int = 0,
        debug: Bool = false) {
        super.init(
            delegate: FullyConnectedLayerDelegate(
                kernelSize: kernelSize,
                channelsIn: channelsIn,
                channelsOut: channelsOut,
                w: w,
                b: b,
                neuronFilter: neuronFilter,
                destinationFeatureChannelOffset: destinationFeatureChannelOffset),
            debug: debug)
    }
}

class FullyConnectedLayerDelegate: CommandEncoderDelegate {
    let fullyConnected: MPSCNNFullyConnected
    
    init(
        kernelSize: Int,
        channelsIn: Int,
        channelsOut: Int,
        w: ParameterBuffer,
        b: ParameterBuffer,
        neuronFilter: MPSCNNNeuron? = MPSCNNNeuronSigmoid(),
        destinationFeatureChannelOffset: Int = 0){
        
        // create appropriate convolution descriptor (in fully connected, stride is always 1)
        let convDesc = MPSCNNConvolutionDescriptor(kernelWidth: kernelSize,
                                                   kernelHeight: kernelSize,
                                                   inputFeatureChannels: channelsIn,
                                                   outputFeatureChannels: channelsOut,
                                                   neuronFilter: neuronFilter)
        
        // initialize the convolution layer by calling the parent's (MPSCNNFullyConnected's) initializer
        fullyConnected = MPSCNNFullyConnected.init(device: ShaderRegistry.getDevice(),
                   convolutionDescriptor: convDesc,
                   kernelWeights: w.pointer(),
                   biasTerms: b.pointer(),
                   flags: MPSCNNConvolutionFlags.none)
        fullyConnected.destinationFeatureChannelOffset = destinationFeatureChannelOffset
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
