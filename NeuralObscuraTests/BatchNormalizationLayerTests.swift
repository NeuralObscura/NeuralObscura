//
//  BatchNormalizationLayerTests.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 11/4/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class BatchNormalizationLayerTests: CommandEncoderBaseTest {

    func testOneFeatureBatchNormalization() {
        let testImg = device.MakeTestMPSImage(width: 2, height: 2, values: [1, 1,
                                                                            1, 1])

        /* Create our CommandEncoder */
        let gamma_pb = MemoryParameterBuffer([2])
        let beta_pb = MemoryParameterBuffer([1])
        let bn = BatchNormalizationLayer(channelsIn: 1,
                                         beta: beta_pb,
                                         gamma: gamma_pb)

        /* Run our test */
        let outputImg = bn.execute(commandBuffer: commandBuffer, sourceImage: testImg)

        print(outputImg)

        let expImg = device.MakeTestMPSImage(width: 2, height: 2, values: [3, 3,
                                                                           3, 3])

        /* Verify the result */
        //XCTAssertEqual(outputImg, expImg)
    }
}
