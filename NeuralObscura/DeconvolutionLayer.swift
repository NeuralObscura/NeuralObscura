//
//  DeconvolutionLayer.swift
//  NeuralObscura
//
//  Created by Edward Knox on 9/27/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class DeconvolutionLayer {
    let device: MTLDevice
    
    init(
        device: MTLDevice,
        channelsIn: UInt,
        channelsOut: UInt,
        kernelSize: UInt,
        w: StyleModelData,
        b: StyleModelData,
        stride: UInt,
        pad: UInt) {
        self.device = device
    }
    
    func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
    }
}
