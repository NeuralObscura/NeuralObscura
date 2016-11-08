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
                                                                            1, 1] as [Float32])
        /* Create our CommandEncoder */
        let gamma_pb = MemoryParameterBuffer([2])
        let beta_pb = MemoryParameterBuffer([1])
        let bn = BatchNormalizationLayer(channelsIn: 1,
                                         beta: beta_pb,
                                         gamma: gamma_pb)

        /* Run our test */
        let outputImg = bn.execute(commandBuffer: commandBuffer, sourceImage: testImg)


        let expImg = device.MakeTestMPSImage(width: 2, height: 2, values: [3, 3,
                                                                           3, 3] as [Float32])

        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }

    func testMultipleFeatureBatchNormalization() {
        let testImg = device.MakeTestMPSImage(width: 2,
                                              height: 2,
                                              featureChannels: 4,
                                              pixelFormat: MTLPixelFormat.rgba32Float,
                                              values: [[1,2,3,4], [4,3,2,1],
                                                       [3,4,2,1], [2,1,3,4]] as [[Float32]])

        /* Create our CommandEncoder*/
        let gamma_pb = MemoryParameterBuffer([3,2,2,3])
        let beta_pb = MemoryParameterBuffer([1,0,1,-1])

        let bn = BatchNormalizationLayer(channelsIn: 1,
                                         beta: beta_pb,
                                         gamma: gamma_pb)

        /* Run our test */
        let outputImg = bn.execute(commandBuffer: commandBuffer, sourceImage: testImg)

        let expImg = device.MakeTestMPSImage(width: 2,
                                             height: 2,
                                             featureChannels: 4,
                                             pixelFormat: MTLPixelFormat.rgba32Float,
                                             values: [[4,4,7,11], [13,6,5,2],
                                                      [10,8,5,2], [7,2,7,11]] as [[Float32]])


        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
}
