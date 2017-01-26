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
                      featureChannels: Int = 1,
                      pixelFormat: MTLPixelFormat = .r32Float,
                      textureType: MTLTextureType = .type2DArray,
                      values: [[Float32]]) -> MPSImage {
        // ravel the values
        var ravelValues = [Float32]()
        for pixel in values {
            for channel in pixel {
                ravelValues.append(channel)
            }
        }

        return MakeMPSImage(width: width,
                            height: height,
                            featureChannels: featureChannels,
                            pixelFormat: pixelFormat,
                            textureType: textureType,
                            values: ravelValues)
    }

    // values are expected to be raveled
    func MakeMPSImage(width: Int,
                      height: Int,
                      featureChannels: Int = 1,
                      pixelFormat: MTLPixelFormat = .r32Float,
                      textureType: MTLTextureType = .type2DArray,
                      values: [Float32]) -> MPSImage {
        let textureDesc = MTLTextureDescriptor()
        textureDesc.textureType = textureType
        textureDesc.width = width
        textureDesc.height = height
        textureDesc.pixelFormat = pixelFormat
        textureDesc.arrayLength = pixelFormat.featureChannelsToSlices(featureChannels)

        let texture = { () -> MTLTexture in 
            switch textureType {
            case .type2D:
                return self.createTextureByRegion(textureDesc: textureDesc, values: values)
            case .type2DArray:
                return self.createTextureBySlices(textureDesc: textureDesc, values: values)
            default:
                fatalError("Unknown MTLTextureType: \(textureType)")
            }
        }()

        return MPSImage(texture: texture, featureChannels: featureChannels)
    }

    private func createTextureByRegion(textureDesc: MTLTextureDescriptor,
                                       values: [Float32]) -> MTLTexture {
        let bytesPerRow = textureDesc.pixelFormat.bytesPerRow(textureDesc.width)
        
        let texture = self.makeTexture(descriptor: textureDesc)

        switch textureDesc.pixelFormat {
        case .rgba16Float, .r16Float:
            let sourceBytes = Conversions.float32toFloat16(values)
            texture.replace(
                region: MTLRegionMake2D(0, 0, texture.width, texture.height),
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
        case .rgba32Float, .r32Float:
            texture.replace(
                region: MTLRegionMake2D(0, 0, textureDesc.width, textureDesc.height),
                mipmapLevel: 0,
                withBytes: values,
                bytesPerRow: bytesPerRow)
        default:
            fatalError("Unknown MTLPixelFormat: \(textureDesc.pixelFormat)")
        }
        return texture
    }

    private func createTextureBySlices(textureDesc: MTLTextureDescriptor,
                                       values: [Float32]) -> MTLTexture {
        let bytesPerRow = textureDesc.pixelFormat.bytesPerRow(textureDesc.width)
        let bytesPerImage = bytesPerRow * textureDesc.height
        let sliceWidth = textureDesc.width * textureDesc.height * textureDesc.pixelFormat.channelCount

        let valuesBySlice = stride(from: 0, to: values.count, by: sliceWidth).map {
            Array(values[$0..<min($0 + sliceWidth, values.count)])
        }
        let texture = self.makeTexture(descriptor: textureDesc)

        for slice in 0...textureDesc.arrayLength-1 {
            switch textureDesc.pixelFormat {
            case .rgba16Float, .r16Float:
                let sourceBytes = Conversions.float32toFloat16(valuesBySlice[slice])
                texture.replace(
                    region: MTLRegionMake2D(0, 0, textureDesc.width, textureDesc.height),
                    mipmapLevel: 0,
                    slice: slice,
                    withBytes: sourceBytes,
                    bytesPerRow: bytesPerRow,
                    bytesPerImage: bytesPerImage)
            case .rgba8Unorm, .r8Unorm:
                var convertedValues = [UInt8]()
                for v in valuesBySlice[slice] {
                    convertedValues.append(UInt8(v))
                }
                texture.replace(
                    region: MTLRegionMake2D(0, 0, textureDesc.width, textureDesc.height),
                    mipmapLevel: 0,
                    slice: slice,
                    withBytes: convertedValues,
                    bytesPerRow: bytesPerRow,
                    bytesPerImage: bytesPerImage)
            case .rgba32Float, .r32Float:
                texture.replace(
                    region: MTLRegionMake2D(0, 0, textureDesc.width, textureDesc.height),
                    mipmapLevel: 0,
                    slice: slice,
                    withBytes: valuesBySlice[slice],
                    bytesPerRow: bytesPerRow,
                    bytesPerImage: bytesPerImage)
            default:
                fatalError("Unknown MTLPixelFormat: \(textureDesc.pixelFormat)")
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
            texture.replace(region: MTLRegionMake2D(0, 0, width, height),
                            mipmapLevel: 0,
                            withBytes: context!.data!,
                            bytesPerRow: pixelFormat.bytesPerRow(width))
        case .rgba16Float:
            let imageBuffer = UnsafeBufferPointer<UInt8>(
                start: context!.data!.bindMemory(to: UInt8.self, capacity: pixelArea),
                count: pixelArea)
            let imageFloats = imageBuffer.enumerated().map { (idx, e) in Float32(e) }
            let imageFloat16 = Conversions.float32toFloat16(imageFloats)
            texture.replace(region: MTLRegionMake2D(0, 0, width, height),
                            mipmapLevel: 0,
                            withBytes: imageFloat16,
                            bytesPerRow: pixelFormat.bytesPerRow(width))
        case .rgba32Float:
            let imageBuffer = UnsafeBufferPointer<UInt8>(
                start: context!.data!.bindMemory(to: UInt8.self, capacity: pixelArea),
                count: pixelArea)
            let imageFloats = imageBuffer.enumerated().map { (idx, e) in Float32(e) }
            texture.replace(region: MTLRegionMake2D(0, 0, width, height),
                            mipmapLevel: 0,
                            withBytes: imageFloats,
                            bytesPerRow: pixelFormat.bytesPerRow(width))
        default:
            fatalError("Unknown MTLPixelFormat: \(self)")
        }
        
        return texture
    }
}


