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
    var useTemporary: Bool
    let debug: Bool
    
    var head: CommandEncoder?
    var top: CommandEncoder?
    var bottom: CommandEncoder?
    
    init(device: MTLDevice,
         delegate: CommandEncoderDelegate,
         useTemporary: Bool = true,
         debug: Bool = true) {
        self.device = device
        self.delegate = delegate
        self.useTemporary = useTemporary
        self.debug = debug
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
        switch self.useTemporary {
        case true: destinationImage = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: destDesc)
        case false: destinationImage = MPSImage(device: self.device, imageDescriptor: destDesc)
        }
        
        delegate.encode(commandBuffer: commandBuffer, sourceImage: sourceImage, destinationImage: destinationImage)
        if (self.useTemporary) {
            let destinationImageTmp = (destinationImage as! MPSTemporaryImage)
            print(destinationImageTmp.readCount)
            destinationImageTmp.readCount -= 1
        } else if (self.debug) {
            // Only will work on non-temporary images
            destinationImage.fourCorners()
        }

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
