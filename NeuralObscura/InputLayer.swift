//
//  InputActivations.swift
//  NeuralObscura
//
//  Created by Edward Knox on 9/29/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class InputLayer: CommandEncoder {
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
    
    func encode(commandBuffer cb: MTLCommandBuffer, sourceImage: MPSImage) {
        bottom?.encode(commandBuffer: cb, sourceImage: sourceImage)
    }
}
