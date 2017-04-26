//
//  MTLTextureExtensions.swift
//  NeuralObscura
//
//  Created by Edward Knox on 12/29/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import Metal
import UIKit

extension MTLTexture {
    func toUIImage(texture: MTLTexture, orientation: UIImageOrientation) -> UIImage {
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * texture.width
        var imageBytes = [UInt8](repeating: 0, count: texture.width * texture.height * bytesPerPixel)
        let region = MTLRegionMake2D(0, 0, texture.width, texture.height)
        texture.getBytes(&imageBytes, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)

        let providerRef = CGDataProvider(data: NSData(bytes: &imageBytes, length: imageBytes.count * MemoryLayout<UInt8>.size))
        let bitmapInfo = CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue)
        let imageRef = CGImage(width: texture.width, height: texture.height, bitsPerComponent: 8, bitsPerPixel: bytesPerPixel * 8, bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bitmapInfo, provider: providerRef!, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!

        return UIImage(cgImage: imageRef, scale: 0, orientation: orientation)
    }

    func fill(_ sourceBytes: UnsafeRawPointer,
              slice: Int = 0) {
        let bytesPerRow = self.pixelFormat.bytesPerRow(self.width)
        let bytesPerImage = bytesPerRow * self.height

        self.replace(
            region: MTLRegionMake2D(0, 0, self.width, self.height),
            mipmapLevel: 0,
            slice: slice,
            withBytes: sourceBytes,
            bytesPerRow: bytesPerRow,
            bytesPerImage: bytesPerImage)
    }
    
//    /* description must be referenced directly until we can add protocol conformance
//     in an extension to another protocol. (CustomStringConvertible) */
//    var bufferDescription: String {
//        var desc = "MTLTexture \(self.hash) with width: \(self.width), height: \(self.height), arrayLength: \(self.arrayLength), pixelFormat raw value: \(self.pixelFormat.rawValue)\n"
//        let rowLength = self.pixelFormat.sizeOfDataType * self.width
//        let imageLength = rowLength * self.height
//        let imageCount = self.pixelFormat.channelCount * self.width * self.height
//        let ptr = UnsafeMutableRawPointer.allocate(bytes: imageLength, alignedTo: MemoryLayout<UInt16>.alignment)
//        let region = MTLRegionMake3D(0, 0, 0, self.width, self.height, 4)
//        for slice in 0 ..< self.arrayLength {
//            self.getBytes(ptr, bytesPerRow: rowLength, bytesPerImage: imageLength, from: region, mipmapLevel: 0, slice: slice)
//            let converted = Conversions.float16toFloat32(ptr, count: imageCount)
//            for row in 0 ..< self.height {
//                for col in 0 ..< self.width {
//                    for pos in 0 ..< 4 {
//                        let e = converted[row * (width * 4) + col * 4 + pos]
//                        print(e)
//                        desc += String(format: "%.2f ", e)
//                    }
//                    desc += "\t"
//                }
//                desc += "\n"
//            }
//            desc += "\n"
//        }
//        return desc
//    }
}
