import UIKit
import MetalPerformanceShaders
import Accelerate

extension UIImage {
    static func MPSImageToUIImage(image: MPSImage, orientation: UIImageOrientation) -> UIImage {
        let texture = image.texture
        let bytesPerPixel = image.pixelSize
        let bytesPerRow = bytesPerPixel * texture.width
        var imageBytes = [UInt8](repeating: 0, count: texture.width * texture.height * bytesPerPixel)
        let region = MTLRegionMake2D(0, 0, texture.width, texture.height)
        texture.getBytes(&imageBytes, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)

        let providerRef = CGDataProvider(data: NSData(bytes: &imageBytes, length: imageBytes.count * MemoryLayout<UInt8>.size))
        let bitmapInfo = CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue)
        let imageRef = CGImage(width: texture.width, height: texture.height, bitsPerComponent: 8, bitsPerPixel: bytesPerPixel * 8, bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bitmapInfo, provider: providerRef!, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!

        return UIImage(cgImage: imageRef, scale: 0, orientation: orientation)
    }

    static func MTLTextureToUIImage(texture: MTLTexture, orientation: UIImageOrientation) -> UIImage {
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

    func createMTLTextureForDevice(device: MTLDevice,
                                   pixelFormat: MTLPixelFormat = .rgba8Unorm) -> MTLTexture {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let ciContext = CIContext.init(mtlDevice: device)

        var image = cgImage
        if (image == nil) {
            let ciImage = CIImage(image: self)
            image = ciContext.createCGImage(ciImage!, from: ciImage!.extent)
        }

        let width = image!.width
        let height = image!.height
        let bounds = CGRect(x: 0, y: 0, width: width, height: height)
        let rowBytes = MTLPixelFormat.rgba8Unorm.bytesPerRow(width)
        let pixelArea = width * height * pixelFormat.pixelCount

        let context = CGContext(data: nil,
                                width: width,
                                height: height,
                                bitsPerComponent: 8,
                                bytesPerRow: rowBytes,
                                space: colorSpace,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        context!.clear(bounds)
        context!.draw(image!, in: bounds)

        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: pixelFormat,
                                                                         width: width,
                                                                         height: height,
                                                                         mipmapped: false)
        let texture = device.makeTexture(descriptor: textureDescriptor)

        switch pixelFormat {
        case .rgba8Unorm:
            texture.replace(region: MTLRegionMake2D(0, 0, width, height),
                            mipmapLevel: 0,
                            withBytes: context!.data!,
                            bytesPerRow: pixelFormat.bytesPerRow(width))
        case .rgba16Float:
            let imageBuffer = UnsafeBufferPointer<UInt8>(start: context!.data!.bindMemory(to: UInt8.self,
                                                                                          capacity: pixelArea),
                                                         count: pixelArea)
            let imageFloats = imageBuffer.enumerated().map { [unowned self] (idx, e) in
                Float32(e)
            }

            let imageFloat16 = Conversions.float32toFloat16(imageFloats)

            texture.replace(region: MTLRegionMake2D(0, 0, width, height),
                            mipmapLevel: 0,
                            withBytes: imageFloat16,
                            bytesPerRow: pixelFormat.bytesPerRow(width))
        case .rgba32Float:
            let imageBuffer = UnsafeBufferPointer<UInt8>(start: context!.data!.bindMemory(to: UInt8.self,
                                                                                          capacity: pixelArea),
                                                         count: pixelArea)
            let imageFloats = imageBuffer.enumerated().map { [unowned self] (idx, e) in
                Float32(e)
            }

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
