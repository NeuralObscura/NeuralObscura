//
//  TestImage.swift
//  NeuralObscura
//
//  Created by Edward Knox on 10/16/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

extension MTLDevice {
    func MakeTestMPSImage(width: Int, height: Int, values: [UInt8]) -> MPSImage {
        let textureDesc = MTLTextureDescriptor()
        textureDesc.textureType = .type2D
        textureDesc.width = width
        textureDesc.height = height
        textureDesc.pixelFormat = .r8Unorm
        let texture = self.makeTexture(descriptor: textureDesc)
        texture.replace(
            region: MTLRegionMake2D(0, 0, texture.width, texture.height),
            mipmapLevel: 0,
            withBytes: values,
            bytesPerRow: texture.width * MemoryLayout<UInt8>.size)
        return MPSImage(texture: texture, featureChannels: 1)
    }

    func MakeDeepTestMPSImage(width: Int, height: Int, values: [[UInt8]]) -> MPSImage {
        let textureDesc = MTLTextureDescriptor()
        textureDesc.textureType = .type2DArray
        textureDesc.width = width
        textureDesc.height = height
        textureDesc.arrayLength = values.count
        textureDesc.pixelFormat = .rgba8Uint
        let texture = self.makeTexture(descriptor: textureDesc)
        for (index, element) in values.enumerated() {
            texture.replace(
                region: MTLRegionMake2D(0, 0, texture.width, texture.height),
                mipmapLevel: 0,
                slice: index,
                withBytes: element,
                bytesPerRow: texture.width * MemoryLayout<UInt8>.size * 4,
                bytesPerImage: texture.width * texture.height * MemoryLayout<UInt8>.size * 4)
        }
        debugPrint(texture)
        return MPSImage(texture: texture, featureChannels: values.count)
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
        
        lhsTexture.getBytes(lhsRawPtr,
            bytesPerRow: lhsRowSize,
            from: MTLRegionMake2D(0, 0, lhs.width, lhs.height),
            mipmapLevel: 0)
        rhsTexture.getBytes(rhsRawPtr,
            bytesPerRow: rhsRowSize,
            from: MTLRegionMake2D(0, 0, rhs.width, rhs.height),
            mipmapLevel: 0)
        
        let lhsPtr = lhsRawPtr.bindMemory(to: UInt8.self, capacity: lhsRowSize * lhs.height)
        let rhsPtr = rhsRawPtr.bindMemory(to: UInt8.self, capacity: rhsRowSize * rhs.height)
        
        let lhsBufferPtr = UnsafeBufferPointer<UInt8>(start: lhsPtr, count: lhsRowSize * lhs.height)
        let rhsBufferPtr = UnsafeBufferPointer<UInt8>(start: rhsPtr, count: rhsRowSize * rhs.height)
        
        return lhsBufferPtr.elementsEqual(rhsBufferPtr)
    }
    
    override open var description: String {
        let ptr = UnsafeMutablePointer<UInt8>.allocate(capacity: self.width * self.height)
        self.texture.getBytes(ptr,
            bytesPerRow: self.width * self.pixelSize,
            from: MTLRegionMake2D(0, 0, self.width, self.height),
            mipmapLevel: 0)
        let buffer = UnsafeBufferPointer<UInt8>(start: ptr, count: self.width * self.height)
        return buffer.enumerated().map { [unowned self] (idx, e) in
            if idx % self.width == 0 {
                return String(format: "\n%2X ", e)
            } else {
                return String(format: "%2X ", e)
            }
        }.joined() + "\n"
   }
    
}
