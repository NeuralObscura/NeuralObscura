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

enum CommandEncoderError: Error {
    case chainMisconfiguration
}

protocol Chain {
    mutating func chain(_ top: CommandEncoder) -> CommandEncoder
}

protocol CommandEncoderDelegate {

    func getDestinationImageDescriptor(sourceImage: MPSImage?) -> MPSImageDescriptor
    
    func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage)
    
}
    
