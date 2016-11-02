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

extension MTLDevice {
    func MakeTestMPSImage(width: Int, height: Int, values: [UInt8]) -> MPSImage {
        let textureDesc = MTLTextureDescriptor()
        textureDesc.textureType = .type2D
        textureDesc.width = width
        textureDesc.height = height
        textureDesc.pixelFormat = .r8Unorm
        let texture = self.makeTexture(descriptor: textureDesc)
        let bytesPerRow = SizeCalculationUtil.calculateBytesPerRow(width: textureDesc.width, pixelFormat: textureDesc.pixelFormat)

        texture.replace(
            region: MTLRegionMake2D(0, 0, textureDesc.width, textureDesc.height),
            mipmapLevel: 0,
            withBytes: values,
            bytesPerRow: bytesPerRow)
        return MPSImage(texture: texture, featureChannels: 1)
    }

    func MakeDeepTestMPSImage(width: Int, height: Int, values: [[UInt8]]) -> MPSImage {
        let textureDesc = MTLTextureDescriptor()
        textureDesc.textureType = .type2DArray
        textureDesc.width = width
        textureDesc.height = height
        textureDesc.arrayLength = values.count
        textureDesc.pixelFormat = .r8Unorm
        let texture = self.makeTexture(descriptor: textureDesc)
        let bytesPerRow = SizeCalculationUtil.calculateBytesPerRow(width: textureDesc.width, pixelFormat: textureDesc.pixelFormat)
        let bytesPerImage = SizeCalculationUtil.calculateBytesPerImage(height: textureDesc.height, bytesPerRow: bytesPerRow)
        for (index, element) in values.enumerated() {
            texture.replace(
                region: MTLRegionMake2D(0, 0, textureDesc.width, textureDesc.height),
                mipmapLevel: 0,
                slice: index,
                withBytes: element,
                bytesPerRow: bytesPerRow,
                bytesPerImage: bytesPerImage)
        }
        debugPrint(texture)
        return MPSImage(texture: texture, featureChannels: SizeCalculationUtil.calculateFeatureChannels(arrayLength: textureDesc.arrayLength,
                                                                                                        pixelFormat: textureDesc.pixelFormat))
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
        
        let lhsRowSize: Int = lhs.pixelSize * lhs.width
        let rhsRowSize: Int = rhs.pixelSize * rhs.width
        
        let lhsTexture = lhs.texture
        let rhsTexture = rhs.texture
        
        let lhsRawPtr = UnsafeMutableRawPointer.allocate(bytes: lhsRowSize * lhs.height, alignedTo: lhs.pixelSize)
        let rhsRawPtr = UnsafeMutableRawPointer.allocate(bytes: rhsRowSize * rhs.height, alignedTo: rhs.pixelSize)
        for i in 0...(featureChannels-1) {
            lhsTexture.getBytes(lhsRawPtr,
                                bytesPerRow: lhsRowSize,
                                bytesPerImage: lhs.height * lhsRowSize,
                                from: MTLRegionMake2D(0, 0, self.width, self.height),
                                mipmapLevel: 0,
                                slice: i)
            rhsTexture.getBytes(rhsRawPtr,
                                bytesPerRow: rhsRowSize,
                                bytesPerImage: rhs.height * rhsRowSize,
                                from: MTLRegionMake2D(0, 0, self.width, self.height),
                                mipmapLevel: 0,
                                slice: i)

            let lhsPtr = lhsRawPtr.bindMemory(to: UInt8.self, capacity: lhsRowSize * lhs.height)
            let rhsPtr = rhsRawPtr.bindMemory(to: UInt8.self, capacity: rhsRowSize * rhs.height)
        
            let lhsBufferPtr = UnsafeBufferPointer<UInt8>(start: lhsPtr, count: lhsRowSize * lhs.height)
            let rhsBufferPtr = UnsafeBufferPointer<UInt8>(start: rhsPtr, count: rhsRowSize * rhs.height)

            if lhsBufferPtr.elementsEqual(rhsBufferPtr) == false {
                return false
            }
        }
        return true
    }
    
    override open var description: String {
        let ptr = UnsafeMutablePointer<UInt8>.allocate(capacity: self.width * self.height)
        var outputString: String = ""

        for i in 0...(featureChannels-1) {
            self.texture.getBytes(ptr,
                                  bytesPerRow: self.width * self.pixelSize,
                                  bytesPerImage: self.height * self.width * self.pixelSize,
                                  from: MTLRegionMake2D(0, 0, self.width, self.height),
                                  mipmapLevel: 0,
                                  slice: i)
            let buffer = UnsafeBufferPointer<UInt8>(start: ptr, count: self.width * self.height)
            outputString += buffer.enumerated().map { [unowned self] (idx, e) in
                if idx % self.width == 0 {
                    return String(format: "\n%2X ", e)
                } else {
                    return String(format: "%2X ", e)
                }
                }.joined() + "\n"
        }


        return outputString
   }
    
}
