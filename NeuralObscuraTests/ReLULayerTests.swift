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
        let testImg = device.MakeTestMPSImage(width: 2, height: 2, values: [-9, -1,
                                                                            1, 2] as [Float32])
        /* Create our CommandEncoder */
        let relu = ReLULayer()

        /* Run our test */
        let outputImg = relu.execute(commandBuffer: commandBuffer, sourceImage: testImg)


        let expImg = device.MakeTestMPSImage(width: 2, height: 2, values: [0, 0,
                                                                           1, 2] as [Float32])

        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
}
