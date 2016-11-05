//
//  TestImage.swift
//  NeuralObscura
//
//  Created by Edward Knox on 10/16/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders
@testable import NeuralObscura
import Accelerate

extension MTLDevice {
    func MakeTestMPSImage(width: Int, height: Int, values: [UInt8]) -> MPSImage {
        var convertedValues = [Float32]()
        for value in values {
            convertedValues.append(Float32(value))
        }

        return MakeTestMPSImageWithMultipleFeatureChannels(width: width, height: height, featureChannels: 1, pixelFormat: .r32Float, values: convertedValues)
    }

    func MakeTestMPSImageUnorm8(width: Int, height: Int, values: [UInt8]) -> MPSImage {
        var convertedValues = [Float32]()
        for value in values {
            convertedValues.append(Float32(value))
        }

        return MakeTestMPSImageWithMultipleFeatureChannels(width: width, height: height, featureChannels: 1, pixelFormat: .r8Unorm, values: convertedValues)

    }

    func MakeTestMPSImage(width: Int, height: Int, featureChannels: Int, pixelFormat: MTLPixelFormat, values: [[UInt8]]) -> MPSImage {
        // ravel the values
        var ravelValues = [Float32]()
        for pixel in values {
            for channel in pixel {
                ravelValues.append(Float32(channel))
            }
        }

        return MakeTestMPSImageWithMultipleFeatureChannels(width: width, height: height, featureChannels: featureChannels, pixelFormat: pixelFormat, values: ravelValues)
    }

    // values are expected to be raveled
    func MakeTestMPSImageWithMultipleFeatureChannels(width: Int, height: Int, featureChannels: Int, pixelFormat: MTLPixelFormat, values: [Float32]) -> MPSImage {
        let textureDesc = MTLTextureDescriptor()
        textureDesc.textureType = .type2DArray
        textureDesc.width = width
        textureDesc.height = height
        textureDesc.pixelFormat = pixelFormat
        let texture = self.makeTexture(descriptor: textureDesc)
        let bytesPerRow = SizeCalculationUtil.calculateBytesPerRow(width: textureDesc.width, pixelFormat: textureDesc.pixelFormat)

        texture.replace(
            region: MTLRegionMake2D(0, 0, textureDesc.width, textureDesc.height),
            mipmapLevel: 0,
            withBytes: values,
            bytesPerRow: bytesPerRow)
        return MPSImage(texture: texture, featureChannels: featureChannels)
    }
}


extension MPSImage {
    
    override open func isEqual(_ rawRhs: Any?) -> Bool {
        let lhs = self

        guard let rhs = rawRhs as? MPSImage else {
            return false
        }

        guard ( lhs.width == rhs.width &&
            lhs.height == rhs.height &&
            lhs.pixelSize == rhs.pixelSize &&
            lhs.pixelFormat == rhs.pixelFormat) else { return false }

        let lhsRowSize: Int = SizeCalculationUtil.calculateBytesPerRow(width: lhs.width, pixelFormat: lhs.pixelFormat)
        let lhsImageSize = SizeCalculationUtil.calculateBytesPerImage(height: lhs.height, bytesPerRow: lhsRowSize)
        let rhsRowSize: Int = SizeCalculationUtil.calculateBytesPerRow(width: rhs.width, pixelFormat: rhs.pixelFormat)
        let rhsImageSize = SizeCalculationUtil.calculateBytesPerImage(height: rhs.height, bytesPerRow: rhsRowSize)

        let lhsTexture = lhs.texture
        let rhsTexture = rhs.texture

        let lhsRawPtr = UnsafeMutableRawPointer.allocate(bytes: lhsImageSize, alignedTo: lhs.pixelSize)
        let rhsRawPtr = UnsafeMutableRawPointer.allocate(bytes: rhsImageSize, alignedTo: rhs.pixelSize)

        let slices = SizeCalculationUtil.pixelFormatWithFeatureChannelsToSlices(
            pixelFormat: self.pixelFormat,
            featureChannels: featureChannels)


        for i in 0...(slices-1) {
            lhsTexture.getBytes(lhsRawPtr,
                                bytesPerRow: lhsRowSize,
                                bytesPerImage: lhsImageSize,
                                from: MTLRegionMake2D(0, 0, self.width, self.height),
                                mipmapLevel: 0,
                                slice: i)
            rhsTexture.getBytes(rhsRawPtr,
                                bytesPerRow: rhsRowSize,
                                bytesPerImage: rhsImageSize,
                                from: MTLRegionMake2D(0, 0, self.width, self.height),
                                mipmapLevel: 0,
                                slice: i)

            let lhsPtr = lhsRawPtr.bindMemory(to: Float32.self, capacity: lhs.width * lhs.height)
            let rhsPtr = rhsRawPtr.bindMemory(to: Float32.self, capacity: rhs.width * rhs.height)

            let lhsBufferPtr = UnsafeBufferPointer<Float32>(start: lhsPtr, count: lhs.width * lhs.height)
            let rhsBufferPtr = UnsafeBufferPointer<Float32>(start: rhsPtr, count: rhs.width * rhs.height)

            if lhsBufferPtr.elementsEqual(rhsBufferPtr) == false {
                return false
            }
        }

        return true
    }

    override open var description: String {
        switch self.pixelFormat {
        case .r8Unorm:
            return UnormToString()
        case .rgba8Unorm:
            return UnormToString()
        case .rgba32Float:
            return Float32ToString()
        case .r32Float:
            return Float32ToString()
        default:
            fatalError("Unknown MTLPixelFormat: \(texture.pixelFormat)")
        }
    }

    func UnormToString() -> String {
        let bytesPerRow = SizeCalculationUtil.calculateBytesPerRow(width: self.width, pixelFormat: self.pixelFormat)
        let bytesPerImage = SizeCalculationUtil.calculateBytesPerImage(height: self.height, bytesPerRow: bytesPerRow)
        let ptr = UnsafeMutablePointer<UInt8>.allocate(capacity: SizeCalculationUtil.calculateTypedSize(width: self.width,
                                                                                                        height: self.height,
                                                                                                        pixelFormat: self.pixelFormat))
        var outputString: String = ""

        let slices = SizeCalculationUtil.pixelFormatWithFeatureChannelsToSlices(
            pixelFormat: self.pixelFormat,
            featureChannels: featureChannels)

        for i in 0...(slices-1) {
            self.texture.getBytes(ptr,
                                  bytesPerRow: bytesPerRow,
                                  bytesPerImage: bytesPerImage,
                                  from: MTLRegionMake2D(0, 0, self.width, self.height),
                                  mipmapLevel: 0,
                                  slice: i)
            let buffer = UnsafeBufferPointer<UInt8>(start: ptr, count: SizeCalculationUtil.calculateTypedSize(width: self.width,
                                                                                                              height: self.height,
                                                                                                              pixelFormat: self.pixelFormat))
            outputString += buffer.enumerated().map { [unowned self] (idx, e) in
                var r = ""
                if idx % (self.width * SizeCalculationUtil.pixelFormatToPixelCount(pixelFormat: self.pixelFormat)) == 0 {
                    r += String(format: "\n%2X ", e)
                } else {
                    if SizeCalculationUtil.pixelFormatToPixelCount(pixelFormat: self.pixelFormat) > 1 && idx % SizeCalculationUtil.pixelFormatToPixelCount(pixelFormat: self.pixelFormat) == 0 {
                        r += "| "
                    }
                    r += String(format: "%2X ", e)
                }
                return r
            }.joined() + "\n"
        }

        return outputString
    }

    func Float32ToString() -> String {
        let bytesPerRow = SizeCalculationUtil.calculateBytesPerRow(width: self.width, pixelFormat: self.pixelFormat)
        let bytesPerImage = SizeCalculationUtil.calculateBytesPerImage(height: self.height, bytesPerRow: bytesPerRow)

        let ptr = UnsafeMutablePointer<Float32>.allocate(capacity: SizeCalculationUtil.calculateTypedSize(width: self.width,
                                                                                                          height: self.height,
                                                                                                          pixelFormat: self.pixelFormat))
        var outputString: String = ""

        let slices = SizeCalculationUtil.pixelFormatWithFeatureChannelsToSlices(
            pixelFormat: self.pixelFormat,
            featureChannels: featureChannels)

        for i in 0...(slices-1) {
            self.texture.getBytes(ptr,
                                  bytesPerRow: bytesPerRow,
                                  bytesPerImage: bytesPerImage,
                                  from: MTLRegionMake2D(0, 0, self.width, self.height),
                                  mipmapLevel: 0,
                                  slice: i)
            let buffer = UnsafeBufferPointer<Float32>(start: ptr, count: SizeCalculationUtil.calculateTypedSize(width: self.width,
                                                                                                                height: self.height,
                                                                                                                pixelFormat: self.pixelFormat))
            outputString += buffer.enumerated().map { [unowned self] (idx, e) in
                var r = ""
                if idx % (self.width * SizeCalculationUtil.pixelFormatToPixelCount(pixelFormat: self.pixelFormat)) == 0 {
                    r += String(format: " \n%.2f ", e)
                } else {
                    if SizeCalculationUtil.pixelFormatToPixelCount(pixelFormat: self.pixelFormat) > 1 && idx % SizeCalculationUtil.pixelFormatToPixelCount(pixelFormat: self.pixelFormat) == 0 {
                        r += "| "
                    }
                    r += String(format: "%.2f ", e)
                }
                return r
            }.joined() + "\n"
        }

        return outputString
    }
}
