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
    func testMakeMPSImageSingleFeatureChannel() {
        let width = 2
        let height = 2
        let values = [1,2,
                      3,4]
        
        let img = device.makeMPSImage(width: width,
                                      height: height,
                                      values: values)
        
        let valuesPadded = [1,0,0,0,
                            2,0,0,0,
                            3,0,0,0,
                            4,0,0,0]
        let valuesConverted = Conversions.float32toFloat16(valuesPadded.map { (val) -> Float32 in
            Float32(val)
        })
        let textureDesc = MTLTextureDescriptor()
        textureDesc.textureType = .type2DArray
        textureDesc.width = width
        textureDesc.height = height
        textureDesc.pixelFormat = .rgba16Float
        textureDesc.arrayLength = 1
        let texture = device.makeTexture(descriptor: textureDesc)
        texture.fill(valuesConverted, slice: 0)
        let expImg = MPSImage(texture: texture, featureChannels: 4)
        
        XCTAssertEqual(img, expImg)
    }
}
