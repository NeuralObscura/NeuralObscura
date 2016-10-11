//
//  CommandEncoderBaseTest.swift
//  NeuralObscura
//
//  Created by Edward Knox on 10/9/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class ConvolutionLayerTests: CommandEncoderBaseTest {
    
    func testConvolution() {
        // Build test model params
        let testWeights: [Float] = [1, 1, 1, 0, 0, 0, 1, 1, 1]
        var testWeightsPtr = UnsafeMutablePointer<Float>.allocate(capacity: testWeights.count)
        for (idx, e) in testWeights.enumerated() {
            testWeightsPtr[idx] = e;
        }
        let w = UnsafePointer<Float>(testWeightsPtr);
        
        var testBias: Float = 0
        var testBiasPtr = UnsafeMutablePointer<Float>.allocate(capacity: 1)
        testBiasPtr.pointee = testBias
        let b = UnsafePointer<Float>(testBiasPtr)
        
        let conv = ConvolutionLayer(
            device: device,
            kernelSize: 3,
            channelsIn: 1,
            channelsOut: 1,
            w: w,
            b: b,
            useTemporary: true)
        
        // Create a test image
        var imgValues: [UInt8] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        var imgDataMem = UnsafeMutablePointer<UInt8>.allocate(capacity: imgValues.count)
        var imgDataMutPtr = UnsafeMutablePointer<UInt8>(imgDataMem)
        for (idx, value) in imgData.enumerated() {
            imgDataMutPtr[idx] = value
        }
        var imgDataPtr: UnsafePointer<UInt8> = UnsafePointer<UInt8>(imgDataMutPtr)
        // var imgData: CFData = CFDataCreateWithBytesNoCopy(
        //     <#T##allocator: CFAllocator!##CFAllocator!#>,
        //     <#T##bytes: UnsafePointer<UInt8>!##UnsafePointer<UInt8>!#>,
        //     <#T##length: CFIndex##CFIndex#>,
        //     <#T##bytesDeallocator: CFAllocator!##CFAllocator!#>)
        var imgData: CFData = CFDataCreate(<#T##allocator: CFAllocator!##CFAllocator!#>, imgDataPtr, <#T##length: CFIndex##CFIndex#>)
        var imgDataProvider: CGDataProvider = CGDataProvider(data: imgData)
        // var imgDataProvider: CGDataProvider = CGDataProvider(
        //     dataInfo: <#T##UnsafeMutableRawPointer?#>,
        //     data: <#T##UnsafeRawPointer#>,
        //     size: <#T##Int#>,
        //     releaseData: <#T##CGDataProviderReleaseDataCallback##CGDataProviderReleaseDataCallback##(UnsafeMutableRawPointer?, UnsafeRawPointer, Int) -> Void#>)
        var img: CGImage = CGImage(
            width: <#T##Int#>,
            height: <#T##Int#>,
            bitsPerComponent: <#T##Int#>,
            bitsPerPixel: <#T##Int#>,
            bytesPerRow: <#T##Int#>,
            space: <#T##CGColorSpace#>,
            bitmapInfo: <#T##CGBitmapInfo#>,
            provider: <#T##CGDataProvider#>,
            decode: <#T##UnsafePointer<CGFloat>?#>,
            shouldInterpolate: <#T##Bool#>,
            intent: <#T##CGColorRenderingIntent#>)
        var texture = UIImage(cgImage: img).createMTLTextureForDevice(device: device)
        var testImage: MPSImage = MPSImage(texture: texture, featureChannels: 1)
        
        // Execute test subject
        var testOutput = conv.execute(commandBuffer: commandBuffer, sourceImage: testImage)
        
        // Create expected output image
        let expectedOutput: MPSImage
        
        assertEqual(testOutput, expectedOutput)
    }
    
}
