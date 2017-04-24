//
//  MTLDeviceExtensionsTests.swift
//  NeuralObscura
//
//  Created by Edward Knox on 4/23/17.
//  Copyright © 2017 Paul Bergeron. All rights reserved.
//

import Foundation
import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class MTLDeviceExtensionsTests: CommandEncoderBaseTest {
    func testMakeMPSImage2dValuesSingleFeatureChannel() {
        /* featureChannels = 1 */
        let values = [[1.0], [2.0], [3.0], [4.0]] as [[Float32]]
        let valuesFlattened = [1.0, 2.0, 3.0, 4.0] as [Float32]
        let width = 2
        let height = 2
        
        let img = device.MakeMPSImage(
            width: width,
            height: height,
            featureChannels: 1,
            pixelFormat: .r16Float,
            textureType: .type2DArray,
            values: values)
        
        let textureDesc = MTLTextureDescriptor()
        textureDesc.textureType = .type2D
        textureDesc.width = width
        textureDesc.height = height
        textureDesc.pixelFormat = .r16Float
        textureDesc.arrayLength = 1
        let texture = device.makeTexture(descriptor: textureDesc)
        let bytes: Array<UInt16> = Conversions.float32toFloat16(valuesFlattened)
        texture.fill(bytes, slice: 0)
        let expImg = MPSImage(texture: texture, featureChannels: 1)
        
        XCTAssertEqual(img, expImg)
    }
    
    func testMakeMPSImage2dValuesTwoFeatureChannels() {
        /* featureChannels = 2 */
        let values = [[1.0, 5.0],
                      [2.0, 6.0],
                      [3.0, 7.0],
                      [4.0, 8.0]] as [[Float32]]
        let valuesFlattened = [1.0, 5.0,
                               2.0, 6.0,
                               3.0, 7.0,
                               4.0, 8.0] as [Float32]
        let width = 2
        let height = 2
        
        let img = device.MakeMPSImage(
            width: width,
            height: height,
            featureChannels: 2,
            pixelFormat: .rg16Float,
            textureType: .type2DArray,
            values: values)
        
        let textureDesc = MTLTextureDescriptor()
        textureDesc.textureType = .type2D
        textureDesc.width = width
        textureDesc.height = height
        textureDesc.pixelFormat = .rg16Float
        textureDesc.arrayLength = 1
        let texture = device.makeTexture(descriptor: textureDesc)
        let bytes: Array<UInt16> = Conversions.float32toFloat16(valuesFlattened)
        texture.fill(bytes, slice: 0)
        let expImg = MPSImage(texture: texture, featureChannels: 2)
        
        XCTAssertEqual(img, expImg)
    }
    
    func testMakeMPSImage2dValuesMultipleFeatureChannels() {
        /* featureChannels = 2 */
        let values = [[1.0, 5.0, 9.0, 13.0],
                      [2.0, 6.0, 10.0, 14.0],
                      [3.0, 7.0, 11.0, 15.0],
                      [4.0, 8.0, 12.0, 16.0]] as [[Float32]]
        let valuesFlattened = [1.0, 5.0, 9.0, 13.0,
                               2.0, 6.0, 10.0, 14.0,
                               3.0, 7.0, 11.0, 15.0,
                               4.0, 8.0, 12.0, 16.0] as [Float32]
        let width = 2
        let height = 2
        
        let img = device.MakeMPSImage(
            width: width,
            height: height,
            featureChannels: 4,
            pixelFormat: .rgba16Float,
            textureType: .type2DArray,
            values: values)
        
        let textureDesc = MTLTextureDescriptor()
        textureDesc.textureType = .type2D
        textureDesc.width = width
        textureDesc.height = height
        textureDesc.pixelFormat = .rgba16Float
        textureDesc.arrayLength = 1
        let texture = device.makeTexture(descriptor: textureDesc)
        let bytes: Array<UInt16> = Conversions.float32toFloat16(valuesFlattened)
        texture.fill(bytes, slice: 0)
        let expImg = MPSImage(texture: texture, featureChannels: 4)
        
        XCTAssertEqual(img, expImg)
    }
    
    func testMakeMPSImage1dValues() {
        
    }
    
    func testCreateTexture2D() {
    }
    
    func testMakeMTLTexture() {
    }
}
