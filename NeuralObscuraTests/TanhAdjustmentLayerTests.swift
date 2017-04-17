//
//  TanhAdjustmentLayerTests.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 11/6/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class TanhAdjustmentLayerTests: CommandEncoderBaseTest {
    func testTanhAdjustmentLayer() {
        let testImg = device.MakeMPSImage(width: 2,
                                              height: 2,
                                              featureChannels: 4,
                                              pixelFormat: testTextureFormatRGBA,
                                              textureType: .type2D,
                                              values: [-25, -25, -25, -25, -25, -25, -25, -25,
                                                       -25, -25, -25, -25, -25, -25, -25, -25] as [Float32])

        /* Create our CommandEncoder*/
        let tanhAdj = TanhAdjustmentLayer()

        /* Run our test */
        let outputImg = tanhAdj.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()

        let expImg = device.MakeMPSImage(width: 2,
                                             height: 2,
                                             featureChannels: 4,
                                             pixelFormat: testTextureFormatRGBA,
                                             textureType: .type2D,
                                             values: [[0,0,0,0], [0,0,0,0],
                                                      [0,0,0,0], [0,0,0,0]] as [[Float32]])

        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
}
