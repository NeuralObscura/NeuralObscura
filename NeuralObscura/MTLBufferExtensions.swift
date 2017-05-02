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
    public static func toString<T: Numeric>(_ buffer : MTLBuffer, type: T.Type) -> String {
        let count = buffer.length / MemoryLayout<T>.size
        var desc = "MTLBuffer \(buffer.hash) with length: \(buffer.length) and count: \(count)\n\n"
        let buff = UnsafeBufferPointer<T>(start: buffer.contents().assumingMemoryBound(to: T.self), count: count)

        for i in 0 ..< count {
            desc += String(format: "%.2f ", buff[i] as! CVarArg)
        }

        return desc
    }

    public static func toString<UInt16>(_ buffer : MTLBuffer, type: UInt16) -> String {
        let count = buffer.length / 2
        print("WEIRD BEHAVIOR: MemoryLayout<UInt16>.size: \(MemoryLayout<UInt16>.size)")
        print("How much other stuff is this breaking?")
        var desc = "MTLBuffer \(buffer.hash) with length: \(buffer.length) and count: \(count)\n\n"
        let values = Conversions.float16toFloat32(buffer.contents(), count: count)

        for i in 0 ..< count {
            desc += String(format: "%.2f ", values[i])
        }

        return desc
    }
    
    public static func loadFromBinary(_ url: URL) -> MTLBuffer {
        let data = try! Data.init(contentsOf: url)
        let count = data.count / MemoryLayout<Float32>.size
        return data.withUnsafeBytes { (pointer: UnsafePointer<Float32>) -> MTLBuffer in
            let converted = Conversions.float32toFloat16(Array(
                UnsafeBufferPointer<Float32>(start: pointer, count: count)))
            return ShaderRegistry.getDevice().makeBuffer(
                bytes: converted,
                length: count * MemoryLayout<UInt16>.size,
                options: MTLResourceOptions.storageModeShared)
        }
    }
}
