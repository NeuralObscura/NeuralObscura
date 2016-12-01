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
    func MakeTestMPSImage(width: Int,
                          height: Int,
                          featureChannels: Int = 1,
                          pixelFormat: MTLPixelFormat = .r16Float,
                          textureType: MTLTextureType = .type2DArray,
                          values: [[Float32]]) -> MPSImage {
        // ravel the values
        var ravelValues = [Float32]()
        for pixel in values {
            for channel in pixel {
                ravelValues.append(channel)
            }
        }

        return MakeTestMPSImage(width: width,
                                height: height,
                                featureChannels: featureChannels,
                                pixelFormat: pixelFormat,
                                textureType: textureType,
                                values: ravelValues)
    }

    // values are expected to be raveled
    func MakeTestMPSImage(width: Int,
                          height: Int,
                          featureChannels: Int = 1,
                          pixelFormat: MTLPixelFormat = .r16Float,
                          textureType: MTLTextureType = .type2DArray,
                          values: [Float32]) -> MPSImage {
        let textureDesc = MTLTextureDescriptor()
        textureDesc.textureType = textureType
        textureDesc.width = width
        textureDesc.height = height
        textureDesc.pixelFormat = pixelFormat
        let texture = self.makeTexture(descriptor: textureDesc)
        let bytesPerRow = textureDesc.pixelFormat.bytesPerRow(textureDesc.width)

        switch pixelFormat {
        case .rgba16Float, .r16Float:
            let sourceBytes = Conversions.float32toFloat16(values)
            texture.replace(
                region: MTLRegionMake2D(0, 0, textureDesc.width, textureDesc.height),
                mipmapLevel: 0,
                withBytes: sourceBytes,
                bytesPerRow: bytesPerRow)
        case .rgba8Unorm, .r8Unorm:
            var convertedValues = [UInt8]()
            for v in values {
                convertedValues.append(UInt8(v))
            }
            texture.replace(
                region: MTLRegionMake2D(0, 0, textureDesc.width, textureDesc.height),
                mipmapLevel: 0,
                withBytes: convertedValues,
                bytesPerRow: bytesPerRow)
        default:
            texture.replace(
                region: MTLRegionMake2D(0, 0, textureDesc.width, textureDesc.height),
                mipmapLevel: 0,
                withBytes: values,
                bytesPerRow: bytesPerRow)
        }

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

        let lhsRowSize: Int = lhs.pixelFormat.bytesPerRow(lhs.width)
        let lhsImageSize = lhs.height * lhsRowSize
        let lhsPixelArea = lhs.width * lhs.height * lhs.pixelFormat.pixelCount
        let rhsRowSize: Int = rhs.pixelFormat.bytesPerRow(rhs.width)
        let rhsImageSize = rhs.height * rhsRowSize
        let rhsPixelArea = rhs.width * rhs.height * rhs.pixelFormat.pixelCount

        let lhsTexture = lhs.texture
        let rhsTexture = rhs.texture

        let lhsRawPtr = UnsafeMutableRawPointer.allocate(bytes: lhsImageSize, alignedTo: lhs.pixelSize)
        let rhsRawPtr = UnsafeMutableRawPointer.allocate(bytes: rhsImageSize, alignedTo: rhs.pixelSize)

        let slices = self.pixelFormat.featureChannelsToSlices(featureChannels)

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

            switch lhs.pixelFormat {
            case .r16Float, .rgba16Float:
                let lhsPtr = lhsRawPtr.bindMemory(to: UInt16.self, capacity: lhsPixelArea)
                let rhsPtr = rhsRawPtr.bindMemory(to: UInt16.self, capacity: rhsPixelArea)

                let lhsBufferPtr = UnsafeBufferPointer<UInt16>(start: lhsPtr, count: lhsPixelArea)
                let rhsBufferPtr = UnsafeBufferPointer<UInt16>(start: rhsPtr, count: rhsPixelArea)
                
                if lhsBufferPtr.elementsEqual(rhsBufferPtr) == false {
                    return false
                }
            case .r8Unorm, .rgba8Unorm:
                let lhsPtr = lhsRawPtr.bindMemory(to: UInt8.self, capacity: lhsPixelArea)
                let rhsPtr = rhsRawPtr.bindMemory(to: UInt8.self, capacity: rhsPixelArea)

                let lhsBufferPtr = UnsafeBufferPointer<UInt8>(start: lhsPtr, count: lhsPixelArea)
                let rhsBufferPtr = UnsafeBufferPointer<UInt8>(start: rhsPtr, count: rhsPixelArea)

                if lhsBufferPtr.elementsEqual(rhsBufferPtr) == false {
                    return false
                }
            default:
                print("Unrecognized pixel format \(lhs.pixelFormat)")
                return false
            }
        }

        return true
    }

    func isLossyEqual(_ rhs: [Float32], percision: Int) -> Bool {
        let lhs = self

        guard ( lhs.width * lhs.height * lhs.featureChannels == rhs.count ) else { return false }

        let lhsRowSize: Int = lhs.pixelFormat.bytesPerRow(lhs.width)
        let lhsImageSize = lhs.height * lhsRowSize
        let lhsPixelArea = lhs.width * lhs.height * lhs.pixelFormat.pixelCount

        let lhsTexture = lhs.texture

        let lhsRawPtr = UnsafeMutableRawPointer.allocate(bytes: lhsImageSize, alignedTo: lhs.pixelSize)

        let slices = self.pixelFormat.featureChannelsToSlices(featureChannels)
        var lhsStr = [String]()

        for i in 0...(slices-1) {
            lhsTexture.getBytes(lhsRawPtr,
                                bytesPerRow: lhsRowSize,
                                bytesPerImage: lhsImageSize,
                                from: MTLRegionMake2D(0, 0, self.width, self.height),
                                mipmapLevel: 0,
                                slice: i)
            switch lhs.pixelFormat {
            case .r16Float, .rgba16Float:
                let lhsPtr = lhsRawPtr.bindMemory(to: UInt16.self, capacity: lhsPixelArea)

                let lhsBufferPtr = UnsafeBufferPointer<UInt16>(start: lhsPtr, count: lhsPixelArea)

                let lhsFloat16Values = lhsBufferPtr.enumerated().map { [unowned self] (idx, e) in
                    e
                }

                let lhsFloatValues = Conversions.float16toFloat32(lhsFloat16Values)

                lhsStr += lhsFloatValues.enumerated().map { [unowned self] (idx, e) in
                    String(format: "%.\(percision)f ", e)
                }
            default:
                print("Unrecognized pixel format \(lhs.pixelFormat)")
                return false
            }
        }

        let rhsStr = rhs.map { (e) in
            String(format: "%.\(percision)f ", e)
        }

        if lhsStr != rhsStr {
            return false
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
        case .r16Float:
            return Float16ToString()
        case .rgba16Float:
            return Float16ToString()
        default:
            fatalError("Unknown MTLPixelFormat: \(texture.pixelFormat)")
        }
    }

    func UnormToString() -> String {
        let bytesPerRow = self.pixelFormat.bytesPerRow(self.width)
        let bytesPerImage = self.height * bytesPerRow
        let ptr = UnsafeMutablePointer<UInt8>.allocate(capacity: self.pixelFormat.typedSize(width: self.width,
                                                                                            height: self.height))
        var outputString: String = ""

        let slices = self.pixelFormat.featureChannelsToSlices(featureChannels)

        for i in 0...(slices-1) {
            self.texture.getBytes(ptr,
                                  bytesPerRow: bytesPerRow,
                                  bytesPerImage: bytesPerImage,
                                  from: MTLRegionMake2D(0, 0, self.width, self.height),
                                  mipmapLevel: 0,
                                  slice: i)
            let buffer = UnsafeBufferPointer<UInt8>(start: ptr, count: self.pixelFormat.typedSize(width: self.width,
                                                                                                  height: self.height))
            outputString += buffer.enumerated().map { [unowned self] (idx, e) in
                var r = ""
                if idx % (self.width * self.pixelFormat.pixelCount) == 0 {
                    r += String(format: "\n%2X ", e)
                } else {
                    if self.pixelFormat.pixelCount > 1 && idx % self.pixelFormat.pixelCount == 0 {
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
        let bytesPerRow = self.pixelFormat.bytesPerRow(self.width)
        let bytesPerImage = self.height * bytesPerRow

        let ptr = UnsafeMutablePointer<Float32>.allocate(capacity: self.pixelFormat.typedSize(width: self.width,
                                                                                              height: self.height))
        var outputString: String = ""

        let slices = self.pixelFormat.featureChannelsToSlices(featureChannels)

        for i in 0...(slices-1) {
            self.texture.getBytes(ptr,
                                  bytesPerRow: bytesPerRow,
                                  bytesPerImage: bytesPerImage,
                                  from: MTLRegionMake2D(0, 0, self.width, self.height),
                                  mipmapLevel: 0,
                                  slice: i)
            let buffer = UnsafeBufferPointer<Float32>(start: ptr, count: self.pixelFormat.typedSize(width: self.width,
                                                                                                    height: self.height))
            outputString += buffer.enumerated().map { [unowned self] (idx, e) in
                var r = ""
                if idx % (self.width * self.pixelFormat.pixelCount) == 0 {
                    r += String(format: " \n%.2f ", e)
                } else {
                    if self.pixelFormat.pixelCount > 1 && idx % self.pixelFormat.pixelCount == 0 {
                        r += "| "
                    }
                    r += String(format: "%.2f ", e)
                }
                return r
                }.joined() + "\n"
        }
        
        return outputString
    }

    func Float16ToString() -> String {
        let bytesPerRow = self.pixelFormat.bytesPerRow(self.width)
        let bytesPerImage = self.height * bytesPerRow

        let ptr = UnsafeMutablePointer<UInt16>.allocate(capacity: self.pixelFormat.typedSize(width: self.width,
                                                                                              height: self.height))
        var outputString: String = ""

        let slices = self.pixelFormat.featureChannelsToSlices(featureChannels)

        for i in 0...(slices-1) {
            self.texture.getBytes(ptr,
                                  bytesPerRow: bytesPerRow,
                                  bytesPerImage: bytesPerImage,
                                  from: MTLRegionMake2D(0, 0, self.width, self.height),
                                  mipmapLevel: 0,
                                  slice: i)
            let buffer = UnsafeBufferPointer<UInt16>(start: ptr, count: self.pixelFormat.typedSize(width: self.width,
                                                                                                    height: self.height))


            let convertedBuffer = Conversions.float16toFloat32(Array(buffer))

            outputString += convertedBuffer.enumerated().map { [unowned self] (idx, e) in
                var r = ""
                if idx % (self.width * self.pixelFormat.pixelCount) == 0 {
                    r += String(format: " \n%.2f ", e)
                } else {
                    if self.pixelFormat.pixelCount > 1 && idx % self.pixelFormat.pixelCount == 0 {
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
