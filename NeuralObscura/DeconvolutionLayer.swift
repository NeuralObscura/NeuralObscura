//
//  ConvolutionLayer.swift
//  NeuralObscura
//
//  Created by Edward Knox on 9/25/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders


class DeconvolutionLayer: CommandEncoder {
    let device: MTLDevice
    var top: CommandEncoder?
    var bottom: CommandEncoder?
    let useTemporary: Bool
    
    let convolution: MPSCNNConvolution
    
    /**
     A property to keep info from init time whether we will pad input image or not for use during encode call
     */
    fileprivate var padding = true
    
    init(
        device: MTLDevice,
        channelsIn: UInt,
        channelsOut: UInt,
        kernelSize: UInt,
        w: StyleModelData,
        b: StyleModelData,
        neuronFilter: MPSCNNNeuron? = MPSCNNNeuronReLU(),
        pad: Int = 0, // TODO: Revisit this default
        stride: Int = 1,
        destinationFeatureChannelOffset: UInt = 0,
        groupNum: UInt = 1,
        useTemporary: Bool = true) {
        
        self.device = device
        self.useTemporary = useTemporary
        self.padding = willPad
        
        
        // TODO: Configure any MPS encoders required for deconvolution
        
        // set padding for calculation of offset during encode call
    }
    
    
    func setBottom(_ bottom: CommandEncoder) {
        self.bottom = bottom
    }
    
    func getDestinationImageDescriptor() -> MPSImageDescriptor {
        // TODO: This won't work for the first layer
        let inDesc = top?.getDestinationImageDescriptor()
        let inHeight = inDesc?.height
        let inWidth = inDesc?.width
        let channelsIn = inDesc?.featureChannels
        
        let kernelSize = convolution.kernelWidth
        let stride = convolution.strideInPixelsX
        let channelsOut = convolution.outputFeatureChannels
        let padding = // TODO: Figure out how the hell padding workds
        // TODO: Calculate shape of output
        return MPSImageDescriptor(channelFormat: textureFormat, width: ???, height: ???, featureChannels: channelsOut)
    }
    
    func getDestinationImage(cb: MTLCommandBuffer) -> MPSImage {
        let destDesc = getDestinationImageDescriptor()
        switch useTemporary {
        case true: return MPSTemporaryImage(commandBuffer: cb, imageDescriptor: destDesc)
        case false: return MPSImage(device: device, imageDescriptor: destDesc)
        }
    }
    
    func chain(_ top: CommandEncoder) -> CommandEncoder {
        self.top = top
        top.setBottom(self)
        return self
    }
    
    func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage) -> MPSImage {
        let destinationImage = getDestinationImage(cb: commandBuffer)
        
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
        
        // TODO: encode deconvolution
        
        switch bottom {
        case .some: return bottom!.encode(commandBuffer: commandBuffer, sourceImage: destinationImage)
        case .none: return destinationImage
        }
    }
}



