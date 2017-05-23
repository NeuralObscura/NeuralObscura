//
//  MTLBufferExtensions.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 5/1/17.
//  Copyright © 2017 Paul Bergeron. All rights reserved.
//

import MetalPerformanceShaders
import UIKit

class MTLBufferUtil {
    public static func toString<T: FloatingPoint>(_ buffer : MTLBuffer, type: T.Type) -> String {
        let count = buffer.length / MemoryLayout<T>.size
        var desc = "MTLBuffer \(buffer.hash) with length: \(buffer.length) and count: \(count)\n\n"
        let buff = UnsafeBufferPointer<T>(start: buffer.contents().assumingMemoryBound(to: T.self), count: count)

        for i in 0 ..< count {
            desc += String(format: "%.2f ", buff[i] as! CVarArg)
        }

        return desc
    }

    public static func toString<T: Integer>(_ buffer : MTLBuffer, type: T.Type) -> String {
        let count = buffer.length / MemoryLayout<T>.size
        var desc = "MTLBuffer \(buffer.hash) with length: \(buffer.length) and count: \(count)\n\n"
        let buff = UnsafeBufferPointer<T>(start: buffer.contents().assumingMemoryBound(to: T.self), count: count)

        for i in 0 ..< count {
            desc += String(format: "%.2f ", buff[i] as! CVarArg)
        }

        return desc
    }

    public static func toString<UInt16>(_ buffer : MTLBuffer, type: UInt16.Type) -> String {
        let count = buffer.length / ExpectedFloat16Size
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
                length: count * ExpectedFloat16Size,
                options: MTLResourceOptions.storageModeShared)
        }
    }

    public static func lossyEqual(lhs : MTLBuffer, rhs : MTLBuffer, precision: Int, type: Float.Type) -> Bool {
        guard ( lhs.length == rhs.length ) else { return false }
        let maxDifference = powf(10.0, Float(-precision))
        let count = lhs.length / MemoryLayout<Float>.size

        let lhsPtr = lhs.contents().bindMemory(to: Float.self, capacity: count)
        let rhsPtr = rhs.contents().bindMemory(to: Float.self, capacity: count)

        for (a, b) in zip(Array(UnsafeBufferPointer(start: lhsPtr, count: count)),
                          Array(UnsafeBufferPointer(start: rhsPtr, count: count))) {
            if abs(a - b) > maxDifference {
                return false
            }
        }

        return true
    }

    public static func lossyEqual<UInt16>(lhs : MTLBuffer, rhs : MTLBuffer, precision: Int, type: UInt16.Type) -> Bool {
        guard ( lhs.length == rhs.length ) else { return false }
        let maxDifference = powf(10.0, Float(-precision))
        let count = lhs.length / ExpectedFloat16Size

        let lhsPtr = lhs.contents().bindMemory(to: UInt16.self, capacity: count)
        let rhsPtr = rhs.contents().bindMemory(to: UInt16.self, capacity: count)

        let lhsFloat32 = Conversions.float16toFloat32(lhsPtr, count: count)
        let rhsFloat32 = Conversions.float16toFloat32(rhsPtr, count: count)

        for (a, b) in zip(Array(UnsafeBufferPointer(start: lhsFloat32, count: count)),
                          Array(UnsafeBufferPointer(start: rhsFloat32, count: count))) {
            if abs(a - b) > maxDifference {
                return false
            }
        }
        
        return true
    }
}
