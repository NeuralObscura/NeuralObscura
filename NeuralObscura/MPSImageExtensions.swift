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

    func isLossyEqual(_ rhs: [Float32], precision: Int) -> Bool {
        let maxDifference = powf(10.0, Float(-precision))
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
                print(lhsFloats[i])
                print(rhs[i])
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
            fatalError("Unknown MTLPixelFormat: \(self.pixelFormat)")
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
            for channel in 0..<self.pixelFormat.channelCount {
                for i in stride(from: channel, to: buffer.count, by: self.pixelFormat.channelCount) {
                    outputString += String(format: "%.2f ", buffer[i])
                }
                outputString += "\n"
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

            for channel in 0..<self.pixelFormat.channelCount {
                for i in stride(from: channel, to: convertedBuffer.count, by: self.pixelFormat.channelCount) {
                    outputString += String(format: "%.2f ", convertedBuffer[i])
                }
                outputString += "\n"
            }
        }

        return outputString
    }

    static func loadFromNumpy(_ url: URL,
                              destinationPixelFormat: MTLPixelFormat = .rgba32Float
        ) -> MPSImage {
        // determine correct way of addressing file (string?)
        // first open file for binary reading
        let data = try! NSMutableData(contentsOf: url, options: Data.ReadingOptions.uncached)
        let ptr = data.bytes

        // read header to determine shape, assume float
        let magicStringPtr = ptr.bindMemory(to: UInt8.self, capacity: 8)
        let magicStringBuf: UnsafeBufferPointer<UInt8> = UnsafeBufferPointer<UInt8>.init(start: magicStringPtr, count: 8)
        let expectedMagicString: [UInt8] = [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00] /* 0x93NUMPY10 */
        assert(magicStringBuf.elementsEqual(expectedMagicString), "Invalid .npy file")
        let headerLen = Int((ptr + 8).bindMemory(to: UInt16.self, capacity: 1).pointee)
        let headerData = data.subdata(with: NSMakeRange(10, headerLen))
        let headerString = String(data: headerData, encoding: .ascii)!
        let cmpts = headerString.components(separatedBy: "'")

        // Parse numpy type description
        let descr = cmpts[3]
        assert(descr == "<f4", "little-endian 32 bit floats (<f4) are the only numpy type currently supported")

        // Assume 'fortran_order': False

        // Parse shape
        let shapeArea = cmpts[8]
        let shapeEndRange = shapeArea.range(of: "}")!
        let shapeStart = shapeArea.index(shapeArea.startIndex, offsetBy: 3)
        let shapeEnd = shapeArea.index(shapeEndRange.upperBound, offsetBy: -4)
        let shapeString = String(shapeArea.substring(with: shapeStart ..< shapeEnd))!.components(separatedBy: ", ")
        let shape = shapeString .map { (dim) -> Int in
            Int(dim)!
        }

        let bodyLen = shape.reduce(1,*)

        /*
         The first 6 bytes are a magic string: exactly “x93NUMPY”.
         The next 1 byte is an unsigned byte: the major version number of the
         file format, e.g. x01.
         The next 1 byte is an unsigned byte: the minor version number of the
         file format, e.g. x00. Note: the version of the file format is not
         tied to the version of the numpy package.
         The next 2 bytes form a little-endian unsigned short int: the length
         of the header data HEADER_LEN.
         6 + 1 + 1 + 2 + headerLen
         */
        let bodyData = (ptr + 10 + headerLen).bindMemory(to: Float32.self, capacity: bodyLen)
        let bodyBuff = UnsafeBufferPointer<Float32>.init(start: bodyData, count: bodyLen)

        var values = [] as [Float32]

        bodyBuff.forEach { (v) in
            values.append(v)
        }

        var result: MPSImage

        switch shape.count {
        case 2:
            let width = shape[0]
            let height = shape[1]
            result =  ShaderRegistry.getDevice().MakeMPSImage(width: width,
                                                              height: height,
                                                              pixelFormat: destinationPixelFormat,
                                                              values: values)
        case 4:
            let channelsOut = shape[0]
            let height = shape[1]
            let width = shape[2]
            let channelsIn = shape[3]

            result =  ShaderRegistry.getDevice().MakeMPSImage(width: width,
                                                              height: height,
                                                              featureChannels: channelsIn,
                                                              pixelFormat: destinationPixelFormat,
                                                              values: values)

        default:
            fatalError("Unknown shape dimensions: \(shape)")
        }
        
        return result
    }
}
