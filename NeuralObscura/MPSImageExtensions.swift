//
//  MPSImageExtensions.swift
//  NeuralObscura
//
//  Created by Edward Knox on 12/29/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import MetalPerformanceShaders
import UIKit

extension MPSImage {
    
    func toUIImage(orientation: UIImageOrientation) -> UIImage {
        let texture = self.texture
        let bytesPerPixel = self.pixelSize
        let bytesPerRow = bytesPerPixel * texture.width
        var imageBytes = [UInt8](repeating: 0, count: texture.width * texture.height * bytesPerPixel)
        let region = MTLRegionMake2D(0, 0, texture.width, texture.height)
        texture.getBytes(&imageBytes, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)
        
        let providerRef = CGDataProvider(
            data: NSData(
                bytes: &imageBytes,
                length: imageBytes.count * MemoryLayout<UInt8>.size))
        let bitmapInfo =
            CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue)
        let imageRef = CGImage(
            width: texture.width,
            height: texture.height,
            bitsPerComponent: 8,
            bitsPerPixel: bytesPerPixel * 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: bitmapInfo,
            provider: providerRef!,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent)!
        return UIImage(cgImage: imageRef, scale: 0, orientation: orientation)
    }
    
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
        let lhsPixelArea = lhs.width * lhs.height * lhs.pixelFormat.channelCount
        let rhsRowSize: Int = rhs.pixelFormat.bytesPerRow(rhs.width)
        let rhsImageSize = rhs.height * rhsRowSize
        let rhsPixelArea = rhs.width * rhs.height * rhs.pixelFormat.channelCount
        
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
                
                debugPrint(lhsBufferPtr)
                debugPrint(rhsBufferPtr)
                
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
            case .r32Float, .rgba32Float:
                let lhsPtr = lhsRawPtr.bindMemory(to: Float32.self, capacity: lhsPixelArea)
                let rhsPtr = rhsRawPtr.bindMemory(to: Float32.self, capacity: rhsPixelArea)
                
                let lhsBufferPtr = UnsafeBufferPointer<Float32>(start: lhsPtr, count: lhsPixelArea)
                let rhsBufferPtr = UnsafeBufferPointer<Float32>(start: rhsPtr, count: rhsPixelArea)
                
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
        let maxDifference = powf(10.0, Float(-percision))
        let lhs = self
        
        guard ( lhs.width * lhs.height * lhs.featureChannels == rhs.count ) else { return false }
        
        let lhsRowSize: Int = lhs.pixelFormat.bytesPerRow(lhs.width)
        let lhsImageSize = lhs.height * lhsRowSize
        let lhsPixelArea = lhs.width * lhs.height * lhs.pixelFormat.channelCount
        
        let lhsTexture = lhs.texture
        
        let lhsRawPtr = UnsafeMutableRawPointer.allocate(bytes: lhsImageSize, alignedTo: lhs.pixelSize)
        
        let slices = self.pixelFormat.featureChannelsToSlices(featureChannels)
        var lhsFloats = [Float32]()
        
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
                let lhsFloat16Values = lhsBufferPtr.enumerated().map { (idx, e) in e }
                let lhsFloatValues = Conversions.float16toFloat32(lhsFloat16Values)
                lhsFloats += lhsFloatValues.enumerated().map { (idx, e) in Float32(e) }
            case .r32Float, .rgba32Float:
                let lhsPtr = lhsRawPtr.bindMemory(to: Float32.self, capacity: lhsPixelArea)
                let lhsBufferPtr = UnsafeBufferPointer<Float32>(start: lhsPtr, count: lhsPixelArea)
                let lhsFloatValues = lhsBufferPtr.enumerated().map { (idx, e) in e }
                lhsFloats += lhsFloatValues.enumerated().map { (idx, e) in Float32(e) }
            default:
                print("Unrecognized pixel format \(lhs.pixelFormat)")
                return false
            }
        }
        
        for i in 0...(rhs.count - 1) {
            if abs(lhsFloats[i] - rhs[i]) > maxDifference {
                return false
            }
        }
        
        return true
    }
    
    override open var description: String {
        switch self.pixelFormat {
        case .r8Unorm, .rgba8Unorm:
            return UnormToString()
        case .r32Float, .rgba32Float:
            return Float32ToString()
        case .r16Float, .rgba16Float:
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
            var channels = Array(repeating: "", count: self.pixelFormat.channelCount)
            buffer.enumerated().forEach { [unowned self] (idx, e) in
                if idx % (self.width * self.pixelFormat.channelCount) == 0 {
                    channels[idx % self.pixelFormat.channelCount] += String(format: "\n ", e)
                } else {
                    channels[idx % self.pixelFormat.channelCount] += String(format: "%.2f ", e)
                }
            }
            for c in channels {
                outputString += c + "\n"
            }
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
            var channels = Array(repeating: "", count: self.pixelFormat.channelCount)
            buffer.enumerated().forEach { [unowned self] (idx, e) in
                if idx % (self.width * self.pixelFormat.channelCount) == 0 {
                    channels[idx % self.pixelFormat.channelCount] += String(format: "\n%.2f ", e)
                } else {
                    channels[idx % self.pixelFormat.channelCount] += String(format: "%.2f ", e)
                }
            }
            for c in channels {
                outputString += c + "\n"
            }
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
            
            var channels = Array(repeating: "", count: self.pixelFormat.channelCount)
            convertedBuffer.enumerated().forEach { [unowned self] (idx, e) in
                if idx % (self.width * self.pixelFormat.channelCount) == 0 {
                    channels[idx % self.pixelFormat.channelCount] += String(format: "\n%.2f ", e)
                } else {
                    channels[idx % self.pixelFormat.channelCount] += String(format: "%.2f ", e)
                }
            }
            for c in channels {
                outputString += c + "\n"
            }

        }
        
        return outputString
    }
}
