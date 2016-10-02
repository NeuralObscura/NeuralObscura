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
    let fullyConnected: MPSCNNFullyConnected
    let device: MTLDevice
    
    var top: CommandEncoder?
    var bottom: CommandEncoder?
    
    let useTemporary: Bool
    
    init(
        device: MTLDevice,
        kernelSize: UInt,
        channelsIn: UInt,
        channelsOut: UInt,
        w: StyleModelData,
        b: StyleModelData,
        neuronFilter: MPSCNNNeuron? = MPSCNNNeuronSigmoid(),
        destinationFeatureChannelOffset: UInt = 0,
        useTemporary: Bool = true){
        
        self.device = device
        self.useTemporary = useTemporary
        
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
    
    func getDestinationImageDescriptor() -> MPSImageDescriptor {
        return MPSImageDescriptor(
            channelFormat: textureFormat,
            width: 1,
            height: 1,
            featureChannels: fullyConnected.outputFeatureChannels)
    }
    
    func getDestinationImage(commandBuffer: MTLCommandBuffer) -> MPSImage {
        let destDesc = getDestinationImageDescriptor()
        switch useTemporary {
        case true: return MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: destDesc)
        case false: return MPSImage(device: device, imageDescriptor: destDesc)
        }
    }
    
    func setBottom(_ bottom: CommandEncoder) {
        self.bottom = bottom
    }
    
    func destinationImage(cb: MTLCommandBuffer, desc: MPSImageDescriptor) -> MPSImage {
        var image: MPSImage!
        if !useTemporary {
            image = MPSImage(device: device, imageDescriptor: desc)
        } else {
            image = MPSTemporaryImage(commandBuffer: cb, imageDescriptor: desc)
        }
        return image
    }
    
    func chain(_ top: CommandEncoder) -> CommandEncoder {
        self.top = top
        top.setBottom(self)
        return self
    }
    
    func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage) -> MPSImage {
        let destinationImage = getDestinationImage(commandBuffer: commandBuffer)
        
        fullyConnected.encode(commandBuffer: commandBuffer, sourceImage: sourceImage, destinationImage: destinationImage)
        
        switch bottom {
        case .some: return bottom!.encode(commandBuffer: commandBuffer, sourceImage: destinationImage)
        case .none: return destinationImage
        }
    }
    
}
