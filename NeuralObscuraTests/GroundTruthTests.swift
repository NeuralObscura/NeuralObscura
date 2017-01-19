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
        let inputMtlTexture = ShaderRegistry.getDevice().MakeMTLTexture(uiImage: image, pixelFormat: testTextureFormatRGBA)

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

        let expImg = [-106.00, 665.06, -432.24, 150.99, -285.92, 294.37, -395.31,
                      -270.16, 138.08, 565.08, -267.65, 42.14, -84.40, 223.20,
                      -219.99, -342.51, 168.58, -337.76, -463.75, -416.94,
                      226.53, -129.81, -395.01, -45.66, 493.14, -425.82, -299.82,
                      -272.22, 488.95, -236.46, -313.79, 21.47, -5.04, -96.70,
                      369.82, 92.56, -242.16, -128.02, 22.18, -243.49, -360.08,
                      -6.69, 309.63, 225.41, -544.36, -105.91, -10.26, -76.66,
                      -396.92, 16.76, 233.61, 349.09, -457.99, 154.35, 198.47,
                      189.36, -267.07, -183.53, 153.35, -93.01, -282.31, -133.38,
                      116.91, -217.36, -180.87, 125.10, 499.73, -70.97, -253.17,
                      153.04, 353.00, -93.06, -248.23, 244.07, 473.69, 11.94,
                      -304.70, 257.48, 323.10, -42.08, -366.12, -91.18, 130.37,
                      -20.71, -119.59, -211.09, 348.34, -254.16, -343.31,
                      -365.86, -109.51, -51.30, -119.98, -342.32, -9.80, -246.38,
                      -43.46, 540.04, 454.95, -194.25, 254.38, 240.34, 149.42,
                      -158.97, 211.57, 196.50, 292.48, 22.63, 481.36, -12.75,
                      68.09, -13.76, 96.83, 209.54, 349.51, -333.89, 394.45,
                      -21.99, -13.05, -298.77, 9.54, 301.70, 378.38, -361.03,
                      370.32, 23.80, 60.13, -413.34] as [Float32]

        /* Run our test */
        let outputImg = conv.execute(commandBuffer: commandBuffer, sourceImage: testImg)
        /* Verify the result */
        XCTAssert(outputImg.isLossyEqual(expImg, precision: -1))
    }

    func testGroundTruthConvRelu() {
        let debug2ImagePath = Bundle.main.path(forResource: "debug2", ofType: "png")!
        let image = UIImage.init(contentsOfFile: debug2ImagePath)!
        let inputMtlTexture = ShaderRegistry.getDevice().MakeMTLTexture(uiImage: image, pixelFormat: testTextureFormatRGBA)

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

        let expImg = [0, 665.06, 0, 150.99, 0, 294.37, 0,
                      0, 138.08, 565.08, 0, 42.14, 0, 223.20,
                      0, 0, 168.58, 0, 0, 0,
                      226.53, 0, 0, 0, 493.14, 0, 0,
                      0, 488.95, 0, 0, 21.47, 0, 0,
                      369.82, 92.56, 0, 0, 22.18, 0, 0,
                      0, 309.63, 225.41, 0, 0, 0, 0,
                      0, 16.76, 233.61, 349.09, 0, 154.35, 198.47,
                      189.36, 0, 0, 153.35, 0, 0, 0,
                      116.91, 0, 0, 125.10, 499.73, 0, 0,
                      153.04, 353.00, 0, 0, 244.07, 473.69, 11.94,
                      0, 257.48, 323.10, 0, 0, 0, 130.37,
                      0, 0, 0, 348.34, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 540.04, 454.95, 0, 254.38, 240.34, 149.42,
                      0, 211.57, 196.50, 292.48, 22.63, 481.36, 0,
                      68.09, 0, 96.83, 209.54, 349.51, 0, 394.45,
                      0, 0, 0, 9.54, 301.70, 378.38, 0,
                      370.32, 23.80, 60.13, 0] as [Float32]

        /* Run our test */
        let outputImg = conv.execute(commandBuffer: commandBuffer, sourceImage: testImg)

        /* Verify the result */
        XCTAssert(outputImg.isLossyEqual(expImg, precision: -1))
    }
    
    func testGroundTruthDeconv() {
        let testUrl = Bundle(for: type(of: self))
            .url(forResource: "deconv-test-data", withExtension: "npy", subdirectory: "testdata")!
        let testImg = MPSImage.loadFromNumpy(testUrl, destinationPixelFormat: testTextureFormatRGBA)

        let w_pb = FileParameterBuffer(modelName: "composition", rawFileName: "d1_W")
        let b_pb = FileParameterBuffer(modelName: "composition", rawFileName: "d1_b")

        let deconv = DeconvolutionLayer(
            kernelSize: 9,
            channelsIn: 3,
            channelsOut: 32,
            w: w_pb,
            b: b_pb,
            relu: true,
            padding: 4,
            debug: true)

        let outputImg = deconv.execute(commandBuffer: commandBuffer, sourceImage: testImg)

        let expUrl = Bundle(for: type(of: self))
            .url(forResource: "deconv-ground-truth", withExtension: "npy", subdirectory: "testdata")!
        let expImg = MPSImage.loadFromNumpy(expUrl, destinationPixelFormat: testTextureFormatRGBA)

        XCTAssertEqual(outputImg, expImg)
    }
}
