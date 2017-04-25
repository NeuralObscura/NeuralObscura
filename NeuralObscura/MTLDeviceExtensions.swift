//
//  TestImage.swift
//  NeuralObscura
//
//  Created by Edward Knox on 10/16/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import MetalPerformanceShaders
import CoreImage
import UIKit

extension MTLDevice {
    
    func makeMPSImage(width: Int, height: Int, values: [Int]) -> MPSImage {
        let converted = values.map { (val) -> Float32 in
            Float32(val)
        }
        return makeMPSImage(width: width, height: height, values: converted)
    }
    
    func makeMPSImage(width: Int, height: Int, values: [[Int]]) -> MPSImage {
        let converted = values.map { (channel) -> [Float32] in
            channel.map { (val) -> Float32 in
                Float32(val)
            }
        }
        return makeMPSImage(width: width, height: height, values: converted)
    }
    
    /* First form takes a single feature channel, with width and height dimensions raveled together. 
       Channel raveling is not supported. */
    func makeMPSImage(width: Int, height: Int, values: [Float32]) -> MPSImage {
        guard values.count == width * height else {
            fatalError("Values array has an incorrect number of elements. values.count: \(values.count), width * height: \(width * height).")
        }
        
        var channelPaddedValues = [Float32]()
        channelPaddedValues.append(contentsOf: values)
        channelPaddedValues.append(contentsOf: [Float32](repeating: 0.0, count: width * height * 3))
        
        let textureDesc = MTLTextureDescriptor()
        textureDesc.textureType = .type2DArray
        textureDesc.width = width
        textureDesc.height = height
        textureDesc.pixelFormat = .rgba16Float
        textureDesc.arrayLength = 1
        let texture = self.makeMTLTexture(textureDesc: textureDesc, values: channelPaddedValues)
        return MPSImage(texture: texture, featureChannels: 4)
    }
    
    /* Second form takes multiple feature channels, with width and height raveled together. */
    func makeMPSImage(width: Int, height: Int, values: [[Float32]]) -> MPSImage {
        let nslices: Int = (values.count + 3) / 4
        let channelsToPad = (4 * nslices - values.count)
        var channelPaddedValues = [Float32]()
        for channel in values {
            channelPaddedValues.append(contentsOf: channel)
        }
        channelPaddedValues.append(contentsOf: [Float32](repeating: 0.0, count: width * height * channelsToPad))
        
        let textureDesc = MTLTextureDescriptor()
        textureDesc.textureType = .type2DArray
        textureDesc.width = width
        textureDesc.height = height
        textureDesc.pixelFormat = .rgba16Float
        textureDesc.arrayLength = nslices
        let texture = self.makeMTLTexture(textureDesc: textureDesc, values: channelPaddedValues)
        /* TODO: return to see if we should use nslices * 4 instead of values.count below */
        return MPSImage(texture: texture, featureChannels: values.count)
    }


    /* Expects an raveled 3D array with shape (channels, height, width). End of story. */
    func makeMTLTexture(textureDesc: MTLTextureDescriptor,
                                values: [Float32]) -> MTLTexture {
        let texture = self.makeTexture(descriptor: textureDesc)
        
        let width = textureDesc.width
        let height = textureDesc.height
        let nslices = textureDesc.arrayLength

        for slice in 0 ..< nslices {
            var sliceValues = [Float32].init(repeating: 0.0, count: width * height * 4)
            for row in 0 ..< height {
                for col in 0 ..< width {
                    for pos in 0 ..< 4 {
                        let placementIndex = row * (width * 4) + col * 4 + pos
                        let channel = slice * 4 + pos
                        let selectionIndex = channel * (height * width) + row * width + col
                        sliceValues[placementIndex] = values[selectionIndex]
                    }
                }
            }
            let converted = Conversions.float32toFloat16(sliceValues)
            texture.fill(converted, slice: slice)
        }
        
        return texture
    }

    func makeMTLTexture(uiImage: UIImage, pixelFormat: MTLPixelFormat = .rgba8Unorm) -> MTLTexture {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let ciContext = CIContext.init(mtlDevice: self)

        let cgImage: CGImage!
        if let img = uiImage.cgImage {
            cgImage = img
        } else {
            let ciImage = CIImage(image: uiImage)
            cgImage = ciContext.createCGImage(ciImage!, from: ciImage!.extent)
        }

        let width = cgImage.width
        let height = cgImage.height
        let bounds = CGRect(x: 0, y: 0, width: width, height: height)
        let rowBytes = MTLPixelFormat.rgba8Unorm.bytesPerRow(width)
        let pixelArea = width * height * pixelFormat.channelCount

        let context = CGContext(data: nil,
                                width: width,
                                height: height,
                                bitsPerComponent: 8,
                                bytesPerRow: rowBytes,
                                space: colorSpace,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        context!.clear(bounds)
        context!.draw(cgImage, in: bounds)

        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: pixelFormat,
                                                                         width: width,
                                                                         height: height,
                                                                         mipmapped: false)
        let texture = self.makeTexture(descriptor: textureDescriptor)

        switch pixelFormat {
        case .rgba8Unorm:
            texture.fill(context!.data!)
        case .rgba16Float:
            let imageBuffer = UnsafeBufferPointer<UInt8>(
                start: context!.data!.bindMemory(to: UInt8.self, capacity: pixelArea),
                count: pixelArea)
            let imageFloats = imageBuffer.enumerated().map { (idx, e) in Float32(e) }
            let imageFloat16 = Conversions.float32toFloat16(imageFloats)
            texture.fill(imageFloat16)
        case .rgba32Float:
            let imageBuffer = UnsafeBufferPointer<UInt8>(
                start: context!.data!.bindMemory(to: UInt8.self, capacity: pixelArea),
                count: pixelArea)
            let imageFloats = imageBuffer.enumerated().map { (idx, e) in Float32(e) }
            texture.fill(imageFloats)
        default:
            fatalError("Unknown MTLPixelFormat: \(self)")
        }
        
        return texture
    }
}


