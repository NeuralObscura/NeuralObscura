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
        let inputMtlTexture = image.createMTLTextureForDevice(device: ShaderRegistry.getDevice(), pixelFormat: .rgba32Float)

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
        let expImg = [-0.53, 1.41, -0.07, 0.94, -0.53, -0.37, -0.07, -0.47,
                      1.78, 0.93, -0.07, -0.07, -0.53, -0.71, -0.07, -0.47,
                      -1.23, -0.03, -0.09, -0.60, -0.78, -0.03, -0.09, -0.60,
                      1.30, -0.03, -0.09, -0.60, 1.26, -0.03, -0.09, 1.32,
                      -0.07, 0.19, 1.91, 0.01, -0.07, 0.19, -1.33, -0.69,
                      -0.07, 0.19, 1.35, 1.01, -0.07, 0.19, -1.53, -0.69,
                      -0.00, -0.31, 0.74, 1.75, -0.00, 0.18, 0.33, 0.52,
                      -0.00, -0.37, -0.19, -0.93, -0.00, -0.37, -0.62, -0.93,
                      0.04, -0.82, 0.54, -0.01, 0.04, -0.48, -0.21, -0.01,
                      0.04, 0.61, 0.40, 1.37, 0.04, 0.77, -0.36, -0.01,
                      0.10, -0.20, 0.21, 0.03, 0.10, -0.20, 1.04, 0.03,
                      0.10, -0.20, -0.29, 0.03, 0.10, -0.20, -0.29, 0.03,
                      -1.54, 1.15, 0.56, -0.19, 0.39, 0.02, -0.35, -0.19,
                      0.06, -0.14, 0.08, 0.76, 2.11, -0.88, -0.59, -0.19, 
                      -0.41, -0.04, 0.63, 0.09, 0.66, -0.42, -0.75, 0.09,
                      -0.73, 0.12, 0.74, 0.09, 0.57, -0.38, -0.51, 0.09] as [Float32]

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
                                              pixelFormat: MTLPixelFormat.rgba32Float,
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
                                             pixelFormat: MTLPixelFormat.rgba32Float,
                                             values: [[4,4,7,11], [13,6,5,2],
                                                      [10,8,5,2], [7,2,7,11]] as [[Float32]])


        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
}
