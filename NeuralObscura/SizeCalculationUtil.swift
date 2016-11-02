//
//  SizeCalculationUtil.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 11/1/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class SizeCalculationUtil {
    static func calculateBytesPerRow(width: Int, pixelFormat: MTLPixelFormat) -> Int {
        return width *
            pixelFormatToPixelCount(pixelFormat: pixelFormat) *
            pixelFormatToSizeOfDataType(pixelFormat: pixelFormat)
    }

    static func calculateBytesPerImage(height: Int, bytesPerRow: Int) -> Int {
        return height * bytesPerRow
    }

    static func calculateFeatureChannels(arrayLength: Int, pixelFormat: MTLPixelFormat) -> Int {
        return arrayLength * pixelFormatToPixelCount(pixelFormat: pixelFormat)
    }

    static func pixelFormatToPixelCount(pixelFormat: MTLPixelFormat) -> Int {
        switch pixelFormat {
        case .rgba8Unorm:
            return 4
        case .r8Unorm:
            return 1
        case .rgba32Float:
            return 4
        default:
            fatalError("Unknown MTLPixelFormat: \(pixelFormat)")
        }
    }

    static func pixelFormatToSizeOfDataType(pixelFormat: MTLPixelFormat) -> Int {
        switch pixelFormat {
        case .rgba8Unorm:
            return MemoryLayout<UInt8>.size
        case .r8Unorm:
            return MemoryLayout<UInt8>.size
        case .rgba32Float:
            return MemoryLayout<Float32>.size
        default:
            fatalError("Unknown MTLPixelFormat: \(pixelFormat)")
        }
    }
}
