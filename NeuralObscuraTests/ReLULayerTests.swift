//
//  ReLULayerTests.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 11/8/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class ReLULayerTests: CommandEncoderBaseTest {

    func testReLULayer() {
        let testImg = device.MakeMPSImage(width: 2,
                                          height: 2,
                                          values: [-9, -1,
                                                   1, 2] as [Float32])
        /* Create our CommandEncoder */
        let relu = ReLULayer().chain(MPSImageVariable(testImg))

        /* Run our test */
        let outputImg = relu.forward(commandBuffer: commandBuffer)
        execute()

        let expImg = device.MakeMPSImage(width: 2,
                                         height: 2,
                                         pixelFormat: testTextureFormatR,
                                         values: [0, 0,
                                                  1, 2] as [Float32])

        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
}
