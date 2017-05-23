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

    func toUIImage(orientation: UIImageOrientation) -> UIImage {
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * self.width
        var imageBytes = [UInt8](repeating: 0, count: self.width * self.height * bytesPerPixel)
        let region = MTLRegionMake2D(0, 0, self.width, self.height)
        self.getBytes(&imageBytes, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)

        let providerRef = CGDataProvider(data: NSData(bytes: &imageBytes, length: imageBytes.count * MemoryLayout<UInt8>.size))
        let bitmapInfo = CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue)
        let imageRef = CGImage(width: self.width, height: self.height, bitsPerComponent: 8, bitsPerPixel: bytesPerPixel * 8, bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bitmapInfo, provider: providerRef!, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!

        return UIImage(cgImage: imageRef, scale: 0, orientation: orientation)
    }

    func fillSlice(_ sourceBytes: UnsafeRawPointer, slice: Int = 0) {
        let bytesPerRow = self.width * self.pixelFormat.channelCount * self.pixelFormat.sizeOfDataType
        let bytesPerSlice = bytesPerRow * self.height

        self.replace(
            region: MTLRegionMake2D(0, 0, self.width, self.height),
            mipmapLevel: 0,
            slice: slice,
            withBytes: sourceBytes,
            bytesPerRow: bytesPerRow,
            bytesPerImage: bytesPerSlice)
    }
}
