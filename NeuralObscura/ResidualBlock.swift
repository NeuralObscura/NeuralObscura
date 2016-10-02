//
//  ResidualBlock.swift
//  NeuralObscura
//
//  Created by Edward Knox on 9/25/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders


/**
 * Blocks simulate a layer by encapsulating multiple inner layers, structured like a sub-network.
 * chain() and encode() methods should not expose any direct linkages to inner layers
 */
class ResidualBlock : CommandEncoder {
    let device: MTLDevice
    let c1, c2: ConvolutionLayer
    let b1, b2: BatchNormalizationLayer
    
    // Convenience label
    var firstLayer: CommandEncoder?
    
    var top: CommandEncoder?
    var bottom: CommandEncoder?
    
    let useTemporary: Bool
    
    /* A property to keep info from init time whether we will pad input image or not for use during encode call */
    fileprivate var padding = true
    
    init(
        device: MTLDevice,
        modelParams: [String: StyleModelData],
        blockName: String,
        channelsIn: UInt,
        channelsOut: UInt,
        kernelSize: UInt = 3,
        stride: Int = 1,
        useTemporary: Bool = true) {
        
        self.device = device
        self.useTemporary = useTemporary
        
        /* Load the block parameters */
        let c1_w = modelParams[blockName + "_c1_W"]
        let c1_b = modelParams[blockName + "_c1_b"]
        let c2_w = modelParams[blockName + "_c2_W"]
        let c2_b = modelParams[blockName + "_c2_b"]
        let b1_beta = modelParams[blockName + "_b1_beta"]
        let b1_gamma = modelParams[blockName + "_b1_gamma"]
        let b2_beta = modelParams[blockName + "_b2_beta"]
        let b2_gamma = modelParams[blockName + "_b2_gamma"]
        
        /* Init block encoders */
        // TODO: Disable relu!
        c1 = ConvolutionLayer(
            device: device,
            kernelSize: kernelSize,
            channelsIn: channelsIn,
            channelsOut: channelsOut,
            w: c1_w,
            b: c1_b,
            relu: false,
            stride: stride)
        
        // TODO: Disable relu!
        c2 = ConvolutionLayer(
            device: device,
            kernelSize: kernelSize,
            channelsIn: channelsOut,
            channelsOut: channelsOut,
            w: c2_w,
            b: c2_b,
            relu: false,
            stride: stride)
        b1 = BatchNormalizationLayer(device: device, channelsIn: channelsOut, beta: b1_beta, gamma: b1_gamma)
        b2 = BatchNormalizationLayer(device: device, channelsIn: channelsOut, beta: b2_beta, gamma: b2_gamma)
        
    }
    
    func setBottom(_ bottom: CommandEncoder) {
        self.bottom = bottom
    }
    
    func getDestinationImage(cb: MTLCommandBuffer) -> MPSImage {
        let destDesc = getDestinationImageDescriptor()
        switch useTemporary {
        case true: return MPSTemporaryImage(commandBuffer: cb, imageDescriptor: destDesc)
        case false: return MPSImage(device: device, imageDescriptor: destDesc)
        }
    }
    
    func chain(_ top: CommandEncoder) -> CommandEncoder {
        firstLayer = c1
        
        var h: CommandEncoder
        
        // TODO: Add relu around b1
        h = b1.chain(c1)
        
        h = b2.chain(c2.chain(h))
        
        // TODO: Set up summation layer
        
        self.top = top
        top.setBottom(self)
        return self
    }
    
    
    func getDestinationImageDescriptor() -> MPSImageDescriptor {
        // Residual blocks don't modify input shape.
        return top!.getDestinationImageDescriptor()
    }
    
    func getDestinationImage(commandBuffer: MTLCommandBuffer) -> MPSImage {
        let destDesc = getDestinationImageDescriptor()
        switch useTemporary {
        case true: return MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: destDesc)
        case false: return MPSImage(device: device, imageDescriptor: destDesc)
        }
    }
    
    func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage) -> MPSImage {
        let destinationImage = getDestinationImage(cb: commandBuffer)
        
        let residualBlockImage = firstLayer!.encode(commandBuffer: commandBuffer, sourceImage: sourceImage)
        
        let residualOutputImage = // TODO: Sum residual output with input (something with ^ residualBlockImage)
        
        switch bottom {
        case .some: return bottom!.encode(commandBuffer: commandBuffer, sourceImage: residualOutputImage)
        case .none: return residualOutputImage
        }
    }
    
}
