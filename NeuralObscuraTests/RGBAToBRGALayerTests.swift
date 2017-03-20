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
        let testImg = device.MakeMPSImage(width: 2,
                                              height: 2,
                                              featureChannels: 4,
                                              pixelFormat: testTextureFormatRGBA,
                                              textureType: .type2D,
                                              values: [[1,2,3,4], [4,3,2,1],
                                                       [4,3,2,1], [1,2,3,4]] as [[Float32]])

        /* Create our CommandEncoder*/
        let tanhAdj = RGBAToBRGALayer()

        /* Run our test */
        let outputImg = tanhAdj.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()

        let expImg = device.MakeMPSImage(width: 2,
                                             height: 2,
                                             featureChannels: 4,
                                             pixelFormat: testTextureFormatRGBA,
                                             textureType: .type2D,
                                             values: [[3,1,2,4], [2,4,3,1],
                                                      [2,4,3,1], [3,1,2,4]] as [[Float32]])

        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
}
