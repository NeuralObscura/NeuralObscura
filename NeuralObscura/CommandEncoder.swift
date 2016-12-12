//
//  CommandEncoder.swift
//  NeuralObscura
//
//  Created by Edward Knox on 10/4/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders


open class CommandEncoder {
    let delegate: CommandEncoderDelegate
    let debug: Bool
    weak var head: CommandEncoder?
    var bottomCallbacks: [(MTLCommandBuffer, MPSImage) -> MPSImage?]
    
    init(delegate: CommandEncoderDelegate,
         debug: Bool = false) {
        self.delegate = delegate
        self.debug = debug
        self.bottomCallbacks = []
        self.head = self
    }
    
    func getDestinationImage(sourceImage: MPSImage, commandBuffer: MTLCommandBuffer) -> MPSImage {
        let destDesc = delegate.getDestinationImageDescriptor(sourceImage: sourceImage)
        let img: MPSImage!
        if debug || bottomCallbacks.isEmpty {
            img = MPSImage(device: ShaderRegistry.getDevice(), imageDescriptor: destDesc)
            if debug {
                DebugFrameStorage.registerFrame(img)
            }
        } else {
            let _img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: destDesc)
            _img .readCount = bottomCallbacks.count
            img = _img
        }
        return img
    }
    
    func forward(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, sourcePosition: Int = 0) -> MPSImage? {
        let ready = delegate.supplyInput(sourceImage: sourceImage, sourcePosition: sourcePosition)
        var returnImage: MPSImage?
        if ready {
            let destinationImage = getDestinationImage(sourceImage: sourceImage, commandBuffer: commandBuffer)
            delegate.encode(commandBuffer: commandBuffer, destinationImage: destinationImage)
            if bottomCallbacks.isEmpty {
                returnImage = destinationImage
            } else {
                for callback in bottomCallbacks {
                    if let result = callback(commandBuffer, destinationImage) {
                        returnImage = result
                    }
                }
            }
        }
        return returnImage
    }
    
    func execute(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage) -> MPSImage {
        let destinationImage = head!.forward(commandBuffer: commandBuffer, sourceImage: sourceImage)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return destinationImage!
    }
    
    func registerBottom(_ callback: @escaping (MTLCommandBuffer, MPSImage) -> MPSImage?) {
        bottomCallbacks.append(callback)
    }
}

open class UnaryCommandEncoder: CommandEncoder, UnaryChain {
    var top: CommandEncoder?
    
    func chain(_ top: CommandEncoder) -> CommandEncoder {
        self.top = top
        self.head = top.head
        top.registerBottom() {
            [unowned self] (commandBuffer: MTLCommandBuffer, sourceImage: MPSImage) -> MPSImage? in
            self.forward(commandBuffer: commandBuffer, sourceImage: sourceImage)
        }
        return self
    }
}

open class BinaryCommandEncoder: CommandEncoder, BinaryChain {
    var topA: CommandEncoder?
    var topB: CommandEncoder?
    
    func chain(_ topA: CommandEncoder, _ topB: CommandEncoder) -> CommandEncoder {
        self.topA = topA
        self.topB = topB
        self.head = topA.head
        topA.registerBottom() {
            [unowned self] (commandBuffer: MTLCommandBuffer, sourceImage: MPSImage) -> MPSImage? in
            self.forward(commandBuffer: commandBuffer, sourceImage: sourceImage, sourcePosition: 0)
        }
        topB.registerBottom() {
            [unowned self] (commandBuffer: MTLCommandBuffer, sourceImage: MPSImage) -> MPSImage? in
            self.forward(commandBuffer: commandBuffer, sourceImage: sourceImage, sourcePosition: 1)
        }
        return self
    }
}
