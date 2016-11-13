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

protocol UnaryChain {
    mutating func chain(_ top: CommandEncoder) -> CommandEncoder
}

protocol BinaryChain {
    mutating func chain(_ topA: CommandEncoder, _ topB: CommandEncoder) -> CommandEncoder
}

protocol CommandEncoderDelegate {
    func getDestinationImageDescriptor(sourceImage: MPSImage) -> MPSImageDescriptor
    
    /* returns true if all inputs have been supplied and delegate is ready for encode() call */
    func supplyInput(sourceImage: MPSImage, sourcePosition: Int) -> Bool
    
    func encode(commandBuffer: MTLCommandBuffer, destinationImage: MPSImage)
    
}

enum CommandEncoderError: Error {
    case chainMisconfiguration
}
