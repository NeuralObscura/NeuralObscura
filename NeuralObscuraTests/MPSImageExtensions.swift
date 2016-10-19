//
//  TestImage.swift
//  NeuralObscura
//
//  Created by Edward Knox on 10/16/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders


extension MPSImage {
    
    convenience init(device: MTLDevice, width: Int, values: [UInt8]) {
        let texture = values.withUnsafeBufferPointer {
            [unowned device] (buffer: UnsafeBufferPointer) -> MTLTexture in
            let height = (values.count + width - 1) / width
            let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .r8Unorm,
                width: width,
                height: height,
                mipmapped: false)
            let texture = device.makeTexture(descriptor: textureDescriptor)
            let region = MTLRegion(
                origin: MTLOriginMake(0, 0, 0),
                size: MTLSizeMake(width, height, 1))
            texture.replace(
                region: region,
                mipmapLevel: 0,
                withBytes: buffer.baseAddress!,
                bytesPerRow: width * MemoryLayout<UInt8>.stride)
            return texture
        }
        self.init(texture: texture, featureChannels: 1)
    }
    
    override open var description: String {
        let count = self.width * self.height
        let ptr = UnsafeMutablePointer<UInt8>.allocate(capacity: count)
        self.texture.getBytes(
            ptr,
            bytesPerRow: self.width * MemoryLayout<UInt8>.size,
            from: MTLRegionMake2D(0, 0, self.width, self.height),
            mipmapLevel: 0)
        let buffer = UnsafeBufferPointer<UInt8>(start: ptr, count: count)
        var str = ""
        for (idx, e) in buffer.enumerated() {
            if idx % self.width == 0 {
                str += "\n"
            }
            str += e.description + " "
        }
        str += "\n"
        return str
   }
}
