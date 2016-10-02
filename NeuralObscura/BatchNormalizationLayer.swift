//
//  BatchNormalizationLayer.swift
//  NeuralObscura
//
//  Created by Edward Knox on 9/27/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class BatchNormalizationLayer: CommandEncoder {
    var beta: Float
    var gamma: Float
    var channelsIn: Int
    let device: MTLDevice
    var top: CommandEncoder?
    var bottom: CommandEncoder?
    let useTemporary: Bool
    
    init(device: MTLDevice, useTemporary: Bool) {
        self.device = device
        self.useTemporary = useTemporary
    }
    
    func setBottom(_ bottom: CommandEncoder) {
        self.bottom = bottom
    }
    
    func getDestinationImageDescriptor() -> MPSImageDescriptor {
        return top!.getDestinationImageDescriptor()
    }
    
    func getDestinationImage(commandBuffer: MTLCommandBuffer) -> MPSImage {
        let destDesc = getDestinationImageDescriptor()
        switch useTemporary {
        case true: return MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: destDesc)
        case false: return MPSImage(device: device, imageDescriptor: destDesc)
        }
    }
    
    func chain(_ top: CommandEncoder) -> CommandEncoder {
        self.top = top
        top.setBottom(self)
        return self
    }
    
    init(device: MTLDevice, channelsIn: UInt, beta: StyleModelData, gamma: StyleModelData) {
        self.device = device
        self.channelsIn = Int(channelsIn)
        self.beta = beta.pointer().pointee // This is guided by a hunch
        self.gamma = gamma.pointer().pointee
    }
    
    func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage) -> MPSImage {
        let destinationImage = getDestinationImage(commandBuffer: commandBuffer)
        
        // TODO: encode batch normalization
        
        switch bottom {
        case .some: return bottom!.encode(commandBuffer: commandBuffer, sourceImage: destinationImage)
        case .none: return destinationImage
        }
    }
}
