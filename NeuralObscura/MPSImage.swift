//
//  MPSImage.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 10/6/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import MetalPerformanceShaders

extension MPSImage {
    func fourCorners() {
        let texture = self.texture
        let bytesPerPixel = self.pixelSize
        let bytesPerRow = bytesPerPixel * texture.width
        var imageBytes = [UInt8](repeating: 0, count: texture.width * texture.height * bytesPerPixel)
        let region = MTLRegionMake2D(0, 0, texture.width, texture.height)
        texture.getBytes(&imageBytes, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)

        let providerRef = CGDataProvider(data: NSData(bytes: &imageBytes, length: imageBytes.count * MemoryLayout<UInt8>.size))
        let rawData = providerRef!.data

        let buf = CFDataGetBytePtr(rawData)
        print(buf![0],
              buf![(bytesPerRow)-4],
              buf![((bytesPerRow*texture.height)-bytesPerRow)],
              buf![(bytesPerRow*texture.height)-4])
    }
}
