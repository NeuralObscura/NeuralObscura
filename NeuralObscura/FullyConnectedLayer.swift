//
//  FullyConnectedLayer.swift
//  NeuralObscura
//
//  Created by Edward Knox on 9/25/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class SlimMPSCNNFullyConnected: MPSCNNFullyConnected {
    
    init(
        device: MTLDevice,
        kernelSize: UInt,
        channelsIn: UInt,
        channelsOut: UInt,
        w: StyleModelData,
        b: StyleModelData,
        neuronFilter: MPSCNNNeuron? = MPSCNNNeuronSigmoid(),
        destinationFeatureChannelOffset: UInt = 0){
        
        // create appropriate convolution descriptor (in fully connected, stride is always 1)
        let convDesc = MPSCNNConvolutionDescriptor(kernelWidth: Int(kernelSize),
                                                   kernelHeight: Int(kernelSize),
                                                   inputFeatureChannels: Int(channelsIn),
                                                   outputFeatureChannels: Int(channelsOut),
                                                   neuronFilter: neuronFilter)
        
        // initialize the convolution layer by calling the parent's (MPSCNNFullyConnected's) initializer
        super.init(device: device,
                   convolutionDescriptor: convDesc,
                   kernelWeights: w.pointer(),
                   biasTerms: b.pointer(),
                   flags: MPSCNNConvolutionFlags.none)
        self.destinationFeatureChannelOffset = Int(destinationFeatureChannelOffset)
    }
    
}
