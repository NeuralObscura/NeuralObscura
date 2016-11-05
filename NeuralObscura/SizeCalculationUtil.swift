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

    static func calculateTypedSize(width: Int, height: Int, pixelFormat: MTLPixelFormat) -> Int {
        return width * height * pixelFormatToPixelCount(pixelFormat: pixelFormat)
    }

    static func pixelFormatToPixelCount(pixelFormat: MTLPixelFormat) -> Int {
        switch pixelFormat {
        case .rgba8Unorm:
            return 4
        case .r8Unorm:
            return 1
        case .rgba32Float:
            return 4
        case .rgba16Float:
            return 4
        case .r32Float:
            return 1
        case .r16Float:
            return 1
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
        case .r32Float:
            return MemoryLayout<Float32>.size
        case .r16Float:
            return MemoryLayout<UInt8>.size * 2
        case .rgba16Float:
            return MemoryLayout<UInt8>.size * 2
        default:
            fatalError("Unknown MTLPixelFormat: \(pixelFormat)")
        }
    }

    static func pixelFormatWithFeatureChannelsToSlices(pixelFormat: MTLPixelFormat, featureChannels: Int) -> Int {
        var slices = ((featureChannels + pixelFormatToPixelCount(pixelFormat: pixelFormat)-1) / pixelFormatToPixelCount(pixelFormat: pixelFormat))
        if slices <= 0 {
            slices = 1
        }
        return slices
    }
}
