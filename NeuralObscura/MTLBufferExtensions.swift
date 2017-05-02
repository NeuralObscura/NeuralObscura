//
//  MTLBufferExtensions.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 5/1/17.
//  Copyright © 2017 Paul Bergeron. All rights reserved.
//

import MetalPerformanceShaders
import UIKit

protocol Numeric { }

extension Float: Numeric {}
extension Double: Numeric {}
extension Int: Numeric {}

protocol HalfNumeric {}
extension UInt16: HalfNumeric {}

class MTLBufferUtil {
    static public func toString<T: Numeric>(_ buffer : MTLBuffer, type: T.Type) -> String {
        var desc = "MTLBuffer \(buffer.hash) with length: \(buffer.length)\n\n"
        let buff = UnsafeBufferPointer<T>(start: buffer.contents().assumingMemoryBound(to: T.self), count: buffer.length)

        for i in 0...buffer.length-1 {
            desc += String(format: "%.2f ", buff[i] as! CVarArg)
        }

        return desc
    }

    static public func toString<UInt16>(_ buffer : MTLBuffer, type: UInt16) -> String {
        var desc = "MTLBuffer \(buffer.hash) with length: \(buffer.length)\n\n"
        let values = Conversions.float16toFloat32(buffer.contents(), count: buffer.length)

        for i in 0...buffer.length-1 {
            desc += String(format: "%.2f ", values[i])
        }

        return desc
    }
}
