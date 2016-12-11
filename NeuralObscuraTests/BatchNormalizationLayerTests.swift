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

    func testBatchNormGroundTruth() {
        let featureChannelsIn = 3
        let featureChannelsOut = 32
        let debug2ImagePath = Bundle.main.path(forResource: "debug2", ofType: "png")!
        let image = UIImage.init(contentsOfFile: debug2ImagePath)!
        let inputMtlTexture = image.createMTLTextureForDevice(device: ShaderRegistry.getDevice())

        let testImg = MPSImage(texture: inputMtlTexture, featureChannels: featureChannelsIn)

        let w_pb = FileParameterBuffer(modelName: "composition", rawFileName: "c1_W")
        let b_pb = FileParameterBuffer(modelName: "composition", rawFileName: "c1_b")

        let conv = ConvolutionLayer(
            kernelSize: 9,
            channelsIn: featureChannelsIn,
            channelsOut: featureChannelsOut,
            w: w_pb,
            b: b_pb,
            relu: true,
            padding: 4,
            debug: true)

        /* Create our CommandEncoder*/
        let gamma = FileParameterBuffer(modelName: "composition", rawFileName: "b1_gamma")
        let beta = FileParameterBuffer(modelName: "composition", rawFileName: "b1_beta")
        let bn =  BatchNormalizationLayer(channelsIn: featureChannelsOut,
                                          beta: beta,
                                          gamma: gamma,
                                          testMode: false)
        /* Run our test */
        let outputImg = bn.chain(conv).execute(commandBuffer: commandBuffer, sourceImage: testImg)

        /* Verify the result */
        let expImg = [-0.45, 1.27, -0.07, 0.82, -0.45, -0.28, -0.07, -0.41,
                      1.55, 0.85, -0.07, -0.07, -0.45, -0.57, -0.07, -0.41,
                      -1.05, -0.03, -0.09, -0.53, -0.66, -0.03, -0.09, -0.53,
                      1.14, -0.03, -0.09, -0.53, 1.11, -0.03, -0.09, 1.13,
                      -0.07, 0.19, 1.67, -0.01, -0.07, 0.19, -1.13, -0.61,
                      -0.07, 0.19, 1.18, 0.86, -0.07, 0.19, -1.31, -0.61,
                      -0.00, -0.30, 0.65, 1.52, -0.00, 0.13, 0.29, 0.46,
                      -0.00, -0.35, -0.16, -0.79, -0.00, -0.35, -0.52, -0.79,
                      0.04, -0.71, 0.48, 0.04, 0.04, -0.42, -0.17, 0.04,
                      0.04, 0.53, 0.36, 1.23, 0.04, 0.67, -0.30, 0.04,
                      0.10, -0.20, 0.20, 0.03, 0.10, -0.20, 0.92, 0.03,
                      0.10, -0.20, -0.23, 0.03, 0.10, -0.20, -0.23, 0.03,
                      -1.30, 1.00, 0.48, -0.16, 0.37, 0.02, -0.31, -0.16,
                      0.09, -0.12, 0.06, 0.67, 1.86, -0.76, -0.52, -0.16,
                      -0.36, -0.06, 0.54, 0.09, 0.58, -0.39, -0.64, 0.09,
                      -0.63, 0.08, 0.64, 0.09, 0.50, -0.35, -0.44, 0.09] as [Float32]

        XCTAssert(outputImg.isLossyEqual(expImg, percision: 2))
    }

    func testOneFeatureBatchNormalization() {
        let testImg = device.MakeTestMPSImage(width: 2, height: 2, values: [1, 1,
                                                                            1, 1] as [Float32])
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
                                              pixelFormat: MTLPixelFormat.rgba16Float,
                                              values: [[1,2,3,4], [4,3,2,1],
                                                       [3,4,2,1], [2,1,3,4]] as [[Float32]])

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
        let outputImg = bn.execute(commandBuffer: commandBuffer, sourceImage: testImg)

        let expImg = device.MakeTestMPSImage(width: 2,
                                             height: 2,
                                             featureChannels: 4,
                                             pixelFormat: MTLPixelFormat.rgba16Float,
                                             values: [[4,4,7,11], [13,6,5,2],
                                                      [10,8,5,2], [7,2,7,11]] as [[Float32]])


        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
}
