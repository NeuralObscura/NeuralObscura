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
    typealias MTLCmomandEncoderHash = Int
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
            print("VALUE LOADED")
            print(v)
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
