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
        /* Create our CommandEncoder */
        let w = MemoryParameterBuffer([0, 0, 0,
                                       0, 1, 0,
                                       0, 0, 0])
        let b = MemoryParameterBuffer(0)
        let conv = ConvolutionLayer(
            device: device,
            kernelSize: 3,
            channelsIn: 1,
            channelsOut: 1,
            w: w,
            b: b,
            relu: false,
            padding: 1,
            debug: true)
        
        let testImgBytes = [0, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 0, 0,
                            0, 0, 0, 0] as [UInt8]
        let expImgBytes =  [0, 0, 0, 0,
                            0, 0, 0, 0,
                            0, 0, 1, 0,
                            0, 0, 0, 0] as [UInt8]
        
        /* Create an input test image */
        let testTextureDesc = MTLTextureDescriptor()
        testTextureDesc.textureType = .type2D
        testTextureDesc.width = 4
        testTextureDesc.height = 4
        testTextureDesc.pixelFormat = .r8Unorm
        // testTextureDesc.resourceOptions
        // testTextureDesc.cpuCacheMode
        // testTextureDesc.usage
        // testTextureDesc.storageMode
        let testTexture = device.makeTexture(descriptor: testTextureDesc)
        testTexture.replace(
            region: MTLRegionMake2D(0, 0, testTexture.width, testTexture.height),
            mipmapLevel: 0,
            withBytes: testImgBytes,
            bytesPerRow: testTexture.width * MemoryLayout<UInt8>.stride)
        let testImg = MPSImage(texture: testTexture, featureChannels: 1)
        
        /* Create an expected test image */
        let expTextureDesc = MTLTextureDescriptor()
        expTextureDesc.textureType = .type2D
        expTextureDesc.width = 4
        expTextureDesc.height = 4
        expTextureDesc.pixelFormat = .r8Unorm
        // expTextureDesc.resourceOptions
        // expTextureDesc.cpuCacheMode
        // expTextureDesc.usage
        // expTextureDesc.storageMode
        let expTexture = device.makeTexture(descriptor: expTextureDesc)
        expTexture.replace(
            region: MTLRegionMake2D(0, 0, expTexture.width, expTexture.height),
            mipmapLevel: 0,
            withBytes: expImgBytes,
            bytesPerRow: expTexture.width * MemoryLayout<UInt8>.stride)
        let expImg = MPSImage(texture: expTexture, featureChannels: 1)
        
        /* Run our test */
        let outputImg = conv.execute(commandBuffer: commandBuffer, sourceImage: testImg)
        
        print("Input image:")
        print(testImg)
        
        print("Output image:")
        print(outputImg)
        
        print("Expected image:")
        print(expImg)
        
        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
    
}
