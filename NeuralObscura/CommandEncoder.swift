//
//  CommandEncoder.swift
//  NeuralObscura
//
//  Created by Edward Knox on 10/4/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

open class CommandEncoder: Chain {
    let device: MTLDevice
    let delegate: CommandEncoderDelegate
    let outputType: CommandEncoderOutputType

    weak var head: CommandEncoder?
    var top: CommandEncoder?
    weak var bottom: CommandEncoder?
    
    init(device: MTLDevice,
         delegate: CommandEncoderDelegate,
         outputType: CommandEncoderOutputType = CommandEncoderOutputType.debug) {
        self.device = device
        self.delegate = delegate
        self.outputType = outputType
        self.head = self
    }
    
    func chain(_ top: CommandEncoder) -> CommandEncoder {
        self.top = top
        self.head = top.head
        top.bottom = self
        return self
    }
    
    func forward(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage) -> MPSImage {
        let destDesc = delegate.getDestinationImageDescriptor(sourceImage: sourceImage)
        
        var destinationImage: MPSImage! = nil
        switch self.outputType {
        case CommandEncoderOutputType.debug: destinationImage = MPSImage(device: self.device, imageDescriptor: destDesc)
        case CommandEncoderOutputType.permenant: destinationImage = MPSImage(device: self.device, imageDescriptor: destDesc)
        case CommandEncoderOutputType.temporary: destinationImage = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: destDesc)
        }
        
        delegate.encode(commandBuffer: commandBuffer, sourceImage: sourceImage, destinationImage: destinationImage)

        switch bottom {
        case .some(_):
            return bottom!.forward(commandBuffer: commandBuffer, sourceImage: destinationImage)
        case .none:
            return destinationImage
        }
    }
    
    func execute(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage) -> MPSImage {
        let destinationImage = head!.forward(commandBuffer: commandBuffer, sourceImage: sourceImage)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return destinationImage
    }
}
