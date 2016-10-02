//
//  CommandEncoder.swift
//  NeuralObscura
//
//  Created by Edward Knox on 9/29/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

let textureFormat = MPSImageFeatureChannelFormat.float16


protocol CommandEncoder {
    
    func getDestinationImageDescriptor() -> MPSImageDescriptor
    
    func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage) -> MPSImage
    
    func chain(_ top: CommandEncoder) -> CommandEncoder
    
    func setBottom(_ bottom: CommandEncoder)
    
}
