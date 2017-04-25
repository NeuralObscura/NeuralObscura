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

    func testBatchNormGroundTruth() throws {
        let testUrl = Bundle(for: type(of: self))
            .url(forResource: "batch_norm_input", withExtension: "npy", subdirectory: "testdata")!
        let testImg = MPSImage.loadFromNumpy(testUrl)
        let w_pb = FileParameterBuffer(modelName: "composition", rawFileName: "c1_W")
        let b_pb = FileParameterBuffer(modelName: "composition", rawFileName: "c1_b")

        let conv = ConvolutionLayer(
            kernelSize: 9,
            channelsIn: 3,
            channelsOut: 32,
            w: w_pb,
            b: b_pb,
            relu: true,
            padding: 4)

        /* Create our CommandEncoder*/
        let gamma = FileParameterBuffer(modelName: "composition", rawFileName: "b1_gamma")
        let beta = FileParameterBuffer(modelName: "composition", rawFileName: "b1_beta")
        
        let bn =  BatchNormalizationLayer(channelsIn: 32,
                                          beta: beta,
                                          gamma: gamma,
                                          testMode: false)
        /* Run our test */
        let outputImg = bn.chain(conv.chain(MPSImageVariable(testImg))).forward(commandBuffer: commandBuffer)
        execute()

        let expUrl = Bundle(for: type(of: self))
            .url(forResource: "batch_norm_expected_output", withExtension: "npy", subdirectory: "testdata")!
        let expImg = MPSImage.loadFromNumpy(expUrl)
        XCTAssert(outputImg.isLossyEqual(image: expImg, precision: 2))
    }

    func testOneFeatureBatchNormalization() {
        let testImg = device.makeMPSImage(width: 2, height: 2, values: [1.0, 1.0, 1.0, 1.0])
        /* Create our CommandEncoder */
        let gamma_pb = MemoryParameterBuffer([2])
        let beta_pb = MemoryParameterBuffer([1])
        let mean_pb = MemoryParameterBuffer([0])
        let stddev_pb = MemoryParameterBuffer([1])
        let bn = BatchNormalizationLayer(channelsIn: 1,
                                         beta: beta_pb,
                                         gamma: gamma_pb,
                                         mean: mean_pb,
                                         stddev: stddev_pb,
                                         testMode: true)

        /* Run our test */
        let outputImg = bn.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()


        let expImg = device.makeMPSImage(width: 2, height: 2, values: [3, 3, 3, 3])

        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
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

        let bn = BatchNormalizationLayer(channelsIn: 1,
                                         beta: beta_pb,
                                         gamma: gamma_pb,
                                         mean: mean_pb,
                                         stddev: stddev_pb,
                                         testMode: true)

        /* Run our test */
        let outputImg = bn.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()

        let expImg = device.makeMPSImage(width: 2,
                                         height: 2,
                                         values: [[4,4,7,11],
                                                  [13,6,5,2],
                                                  [10,8,5,2],
                                                  [7,2,7,11]])


        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
}
