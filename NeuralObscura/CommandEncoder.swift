//
//  CommandEncoder.swift
//  NeuralObscura
//
//  Created by Edward Knox on 10/4/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders


protocol CommandEncoder {
    associatedtype OutputType
    func forward(commandBuffer: MTLCommandBuffer) -> OutputType
    func registerConsumer()
}

protocol UnaryCommandEncoder: CommandEncoder {
    associatedtype InputType
    mutating func chain(_ top: AnyCommandEncoder<InputType>) -> AnyCommandEncoder<OutputType>
}

protocol BinaryCommandEncoder: CommandEncoder {
    associatedtype InputTypeA
    associatedtype InputTypeB
    mutating func chain(
        _ topA: AnyCommandEncoder<InputTypeA>,
        _ topB: AnyCommandEncoder<InputTypeB>) -> AnyCommandEncoder<OutputType>
}

/* associated type thunk */
class AnyCommandEncoder<T>: CommandEncoder {

    typealias OutputType = T
    
    internal var _value: OutputType? = nil
    private var _forward: ((MTLCommandBuffer) -> OutputType)? = nil
    private var _registerConsumer: (() -> Void)? = nil
    
    init<V: CommandEncoder>(_ exp: V) where V.OutputType == T {
        _forward = exp.forward
        _registerConsumer = exp.registerConsumer
    }
    
    /* alternate constructor for variable subclasses */
    init() {}
    
    func forward(commandBuffer: MTLCommandBuffer) -> OutputType {
        if let v = _value {
            return v
        } else if let f = _forward {
            return f(commandBuffer)
        } else {
            fatalError("Neither constructor used to initialize AnyCommandEncoder")
        }
    }
    
    func registerConsumer() {
        if let rc = _registerConsumer {
            rc()
        }
    }
}


//open class UnaryCommandEncoder: CommandEncoder, UnaryChain {
//    var top: CommandEncoder?
//    
//    func chain(_ top: CommandEncoder) -> CommandEncoder {
//        self.top = top
//        self.head = top.head
//        top.registerBottom() {
//            [unowned self] (commandBuffer: MTLCommandBuffer, sourceImage: MPSImage) -> MPSImage? in
//            self.forward(commandBuffer: commandBuffer, sourceImage: sourceImage)
//        }
//        return self
//    }
//}
//
//open class BinaryCommandEncoder: CommandEncoder, BinaryChain {
//    var topA: CommandEncoder?
//    var topB: CommandEncoder?
//    
//    func chain(_ topA: CommandEncoder, _ topB: CommandEncoder) -> CommandEncoder {
//        self.topA = topA
//        self.topB = topB
//        self.head = topA.head
//        topA.registerBottom() {
//            [unowned self] (commandBuffer: MTLCommandBuffer, sourceImage: MPSImage) -> MPSImage? in
//            self.forward(commandBuffer: commandBuffer, sourceImage: sourceImage, sourcePosition: 0)
//        }
//        topB.registerBottom() {
//            [unowned self] (commandBuffer: MTLCommandBuffer, sourceImage: MPSImage) -> MPSImage? in
//            self.forward(commandBuffer: commandBuffer, sourceImage: sourceImage, sourcePosition: 1)
//        }
//        return self
//    }
//}
