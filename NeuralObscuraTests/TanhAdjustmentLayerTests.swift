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
        // textureType is 2DArray since the number of feature channels at this
        // stage in processing is 3, RGB, as this is meant as a cleanup step of
        // the final image.
        let testImg = device.MakeTestMPSImage(width: 2,
                                              height: 2,
                                              featureChannels: 3,
                                              pixelFormat: MTLPixelFormat.rgba16Float,
                                              textureType: .type2D,
                                              values: [-25,-25,-25, -25,-25,-25,
                                                       -25,-25,-25, -25,-25,-25] as [Float32])

        /* Create our CommandEncoder*/
        let tanhAdj = TanhAdjustmentLayer()

        /* Run our test */
        let outputImg = tanhAdj.execute(commandBuffer: commandBuffer, sourceImage: testImg)

        let expImg = device.MakeTestMPSImage(width: 2,
                                             height: 2,
                                             featureChannels: 4,
                                             pixelFormat: MTLPixelFormat.rgba16Float,
                                             textureType: .type2D,
                                             values: [[0,0,0,255], [0,0,0,255],
                                                      [0,0,0,255], [0,0,0,255]] as [[Float32]])

        print(outputImg)
        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
}
