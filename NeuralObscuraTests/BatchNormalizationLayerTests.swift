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
        let testImg = device.makeMPSImage(width: 2, height: 2, values: [1.0, 1.0,
                                                                        1.0, 1.0])
        /* Create our CommandEncoder */
        let gamma_pb = MemoryParameterBuffer([2,0,0,0])
        let beta_pb = MemoryParameterBuffer([1,0,0,0])
        let mean_pb = MemoryParameterBuffer([0,0,0,0])
        let stddev_pb = MemoryParameterBuffer([1,0,0,0])
        let bn = BatchNormalizationLayer(beta: beta_pb,
                                         gamma: gamma_pb,
                                         mean: mean_pb,
                                         stddev: stddev_pb)

        /* Run our test */
        let outputImg = bn.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()

        let expImg = device.makeMPSImage(width: 2, height: 2, values: [3, 3, 3, 3])

        /* Verify the result */
        XCTAssert(outputImg.isLossyEqual(image: expImg, precision: 1))
    }

    func testMultipleFeatureBatchNormalization() {
        let testImg = device.makeMPSImage(width: 2, height: 2, values: [[1,4,3,2],
                                                                        [2,3,4,1],
                                                                        [3,2,2,4],
                                                                        [4,1,1,4]])

        /* Create our CommandEncoder*/
        let gamma_pb = MemoryParameterBuffer([3,2,2,3])
        let beta_pb = MemoryParameterBuffer([1,0,1,-1])
        let mean_pb = MemoryParameterBuffer([0,0,0,0])
        let stddev_pb = MemoryParameterBuffer([1,1,1,1])

        let bn = BatchNormalizationLayer(beta: beta_pb, gamma: gamma_pb, mean: mean_pb, stddev: stddev_pb)

        /* Run our test */
        let outputImg = bn.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()

        let expImg = device.makeMPSImage(width: 2,
                                         height: 2,
                                         values: [[4,13,10,7],
                                                  [4,6,8,2],
                                                  [7,5,5,9],
                                                  [11,2,2,11]])

        /* Verify the result */
        XCTAssert(outputImg.isLossyEqual(image: expImg, precision: 2))
    }
}
