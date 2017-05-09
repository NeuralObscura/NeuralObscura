//
//  RGBAToBRGALayerTests.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 11/26/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class RGBAToBRGALayerTests: CommandEncoderBaseTest {
    func testRGBAToBRGALayer() {
        let textureDesc = MTLTextureDescriptor()
        textureDesc.textureType = .type2D
        textureDesc.width = 2
        textureDesc.height = 2
        textureDesc.pixelFormat = .rgba16Float

        let testImg = MPSImage(texture:
            device.makeMTLTexture(textureDesc: textureDesc,
                                  values: [1.0,4.0,3.0,2.0,
                                           2.0,3.0,4.0,1.0,
                                           3.0,2.0,2.0,3.0,
                                           4.0,1.0,1.0,4.0]),
                               featureChannels: 4)


        let rgbaToBRGA = RGBAToBRGALayer()

        let outputImg = rgbaToBRGA.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()

        let expImg = MPSImage(texture:
            device.makeMTLTexture(textureDesc: textureDesc,
                                  values: [3.0,2.0,2.0,3.0,
                                           1.0,4.0,3.0,2.0,
                                           2.0,3.0,4.0,1.0,
                                           4.0,1.0,1.0,4.0]),
                               featureChannels: 4)

        XCTAssertEqual(outputImg, expImg)
    }
}
