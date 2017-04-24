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
    func MakeMPSImage(width: Int,
                      height: Int,
                      featureChannels: Int = 4,
                      pixelFormat: MTLPixelFormat = .rgba16Float,
                      textureType: MTLTextureType = .type2DArray,
                      values: [[Float32]]) -> MPSImage {
        
        let nslices = pixelFormat.featureChannelsToSlices(featureChannels)
        let allocatedFeatureChannels = pixelFormat.channelCount * nslices
        var raveledValues = [Float32]()
        for pixel in values {
            let channelValue = { (channel: Int) -> Float32 in
                if channel < pixel.count {
                    return pixel[channel]
                } else {
                    return 0.0
                }
            }
            for channel in 0..<allocatedFeatureChannels {
                raveledValues.append(channelValue(channel))
            }
        }
        print("raveled values: \(raveledValues)")
        return MakeMPSImage(width: width,
                            height: height,
                            featureChannels: featureChannels,
                            values: raveledValues)
    }

    // values are expected to be raveled
    func MakeMPSImage(width: Int,
                      height: Int,
                      featureChannels: Int = 1,
                      pixelFormat: MTLPixelFormat = .rgba16Float,
                      textureType: MTLTextureType = .type2DArray,
                      values: [Float32]) -> MPSImage {
        
        guard textureType == .type2D || textureType == .type2DArray else {
            fatalError("Unsupported MTLTextureType: \(textureType)")
        }
        
        let textureDesc = MTLTextureDescriptor()
        textureDesc.textureType = textureType
        textureDesc.width = width
        textureDesc.height = height
        textureDesc.pixelFormat = pixelFormat
        textureDesc.arrayLength = pixelFormat.featureChannelsToSlices(featureChannels)

        let texture = self.MakeMTLTexture(textureDesc: textureDesc, values: values)
        print("texture.arrayLength: \(texture.arrayLength), featureChannels: \(featureChannels)")
        return MPSImage(texture: texture, featureChannels: featureChannels)
    }

    private func MakeMTLTexture(textureDesc: MTLTextureDescriptor,
                                values: [Float32]) -> MTLTexture {
        let texture = self.makeTexture(descriptor: textureDesc)
        
        let sliceWidth =
            textureDesc.width *
            textureDesc.height *
            textureDesc.pixelFormat.channelCount

        for (index, sliceStart) in zip(0..<textureDesc.arrayLength, stride(from: 0, to: values.count, by: sliceWidth)) {
            print("values.count: \(values.count), sliceStart: \(sliceStart), sliceWidth: \(sliceWidth)")
            let slice = Array(values[sliceStart..<sliceStart + sliceWidth])
            
            switch textureDesc.pixelFormat {
            case .rgba32Float, .r32Float:
                texture.fill(slice, slice: index)
            case .rgba16Float, .r16Float:
                let converted = Conversions.float32toFloat16(slice)
                texture.fill(converted, slice: index)
            case .rgba8Unorm, .r8Unorm:
                let converted = Conversions.float32toUInt8(slice)
                texture.fill(converted, slice: index)
            default:
                fatalError("Unsupported MTLPixelFormat: \(textureDesc.pixelFormat)")
            }
        }
        return texture
    }

    func MakeMTLTexture(uiImage: UIImage, pixelFormat: MTLPixelFormat = .rgba8Unorm) -> MTLTexture {
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


