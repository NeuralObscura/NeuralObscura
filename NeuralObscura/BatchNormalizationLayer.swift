//
//  BatchNormalizationLayer.swift
//  NeuralObscura
//
//  Created by Edward Knox on 9/27/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class BatchNormalizationLayer {
    let device: MTLDevice
    let beta: Float
    let gamma: Float
    
    init(device: MTLDevice, channelsIn: UInt, beta: StyleModelData, gamma: StyleModelData) {
        self.device = device
        self.beta = beta.pointer().pointee // This is guided by a hunch
        self.gamma = gamma.pointer().pointee
    }
    
    func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
    }
}
