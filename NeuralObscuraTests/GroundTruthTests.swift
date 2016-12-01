//
//  GroundTruthTests.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 11/15/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class GroundTruthTests: CommandEncoderBaseTest {

    func testGroundTruthConv() {
        let debug2ImagePath = Bundle.main.path(forResource: "debug2", ofType: "png")!
        let image = UIImage.init(contentsOfFile: debug2ImagePath)!
        let inputMtlTexture = image.createMTLTextureForDevice(device: ShaderRegistry.getDevice())

        let testImg = MPSImage(texture: inputMtlTexture, featureChannels: 3)

        let w_pb = FileParameterBuffer(modelName: "composition", rawFileName: "c1_W")
        let b_pb = FileParameterBuffer(modelName: "composition", rawFileName: "c1_b")

        let conv = ConvolutionLayer(
            kernelSize: 9,
            channelsIn: 3,
            channelsOut: 32,
            w: w_pb,
            b: b_pb,
            relu: false,
            padding: 4,
            debug: true)

        let expImg = [-105.94, 665.00, -432.00, 150.88, -285.75, 294.25, -395.25, -270.00,
                      138.00, 565.00, -267.50, 42.12, -84.38, 223.12, -219.88, -342.50,
                      168.50, -337.75, -463.75, -416.75, 226.50, -129.75, -395.00, -45.66,
                      493.00, -425.75, -299.75, -272.00, 488.75, -236.38, -313.75, 21.45,
                      -5.04, -96.69, 369.75, 92.56, -242.12, -128.00, 22.17, -243.38,
                      -360.00, -6.68, 309.50, 225.38, -544.00, -105.88, -10.25, -76.62,
                      -396.75, 16.75, 233.50, 349.00, -457.75, 154.25, 198.38, 189.25,
                      -267.00, -183.50, 153.25, -93.00, -282.25, -133.38, 116.88, -217.25,
                      -180.75, 125.06, 499.50, -70.94, -253.12, 153.00, 353.00, -93.00,
                      -248.12, 244.00, 473.50, 11.94, -304.50, 257.25, 323.00, -42.06,
                      -366.00, -91.12, 130.25, -20.70, -119.56, -211.00, 348.25, -254.12,
                      -343.25, -365.75, -109.50, -51.28, -119.94, -342.25, -9.80, -246.38,
                      -43.44, 540.00, 454.75, -194.12, 254.38, 240.25, 149.38, -158.88,
                      211.50, 196.50, 292.25, 22.62, 481.25, -12.75, 68.06, -13.76, 
                      96.81, 209.50, 349.50, -333.75, 394.25, -21.98, -13.04, -298.75,  
                      9.54, 301.50, 378.25, -361.00, 370.25, 23.80, 60.12, -413.25] as [Float32]

        /* Run our test */
        let outputImg = conv.execute(commandBuffer: commandBuffer, sourceImage: testImg)
        /* Verify the result */
        XCTAssert(outputImg.isLossyEqual(expImg, percision: 2))
    }

    func testGroundTruthConvRelu() {
        let debug2ImagePath = Bundle.main.path(forResource: "debug2", ofType: "png")!
        let image = UIImage.init(contentsOfFile: debug2ImagePath)!
        let inputMtlTexture = image.createMTLTextureForDevice(device: ShaderRegistry.getDevice())

        let testImg = MPSImage(texture: inputMtlTexture, featureChannels: 3)

        let w_pb = FileParameterBuffer(modelName: "composition", rawFileName: "c1_W")
        let b_pb = FileParameterBuffer(modelName: "composition", rawFileName: "c1_b")

        let conv = ConvolutionLayer(
            kernelSize: 9,
            channelsIn: 3,
            channelsOut: 32,
            w: w_pb,
            b: b_pb,
            relu: true,
            padding: 4,
            debug: true)

        let expImg = [0, 665.00, 0, 150.88, 0, 294.25, 0, 0,
                      138.00, 565.00, 0, 42.12, 0, 223.12, 0, 0,
                      168.50, 0, 0, 0, 226.50, 0, 0, 0,
                      493.00, 0, 0, 0, 488.75, 0, 0, 21.45,
                      0, 0, 369.75, 92.56, 0, 0, 22.17, 0,
                      0, 0, 309.50, 225.38, 0, 0, 0, 0,
                      0, 16.75, 233.50, 349.00, 0, 154.25, 198.38, 189.25,
                      0, 0, 153.25, 0, 0, 0, 116.88, 0,
                      0, 125.06, 499.50, 0, 0, 153.00, 353.00, 0,
                      0, 244.00, 473.50, 11.94, 0, 257.25, 323.00, 0,
                      0, 0, 130.25, 0, 0, 0, 348.25, 0,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0, 540.00, 454.75, 0, 254.38, 240.25, 149.38, 0,
                      211.50, 196.50, 292.25, 22.62, 481.25, 0, 68.06, 0,
                      96.81, 209.50, 349.50, 0, 394.25, 0, 0, 0,
                      9.54, 301.50, 378.25, 0, 370.25, 23.80, 60.12, 0] as [Float32]

        /* Run our test */
        let outputImg = conv.execute(commandBuffer: commandBuffer, sourceImage: testImg)
        /* Verify the result */
        XCTAssert(outputImg.isLossyEqual(expImg, percision: 2))
    }

}
