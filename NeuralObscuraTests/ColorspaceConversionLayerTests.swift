//
//  ColorspaceConversionLayerTests.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 6/5/17.
//  Copyright © 2017 Paul Bergeron. All rights reserved.
//

import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class ColorspaceConversionLayerTests: CommandEncoderBaseTest {

    func testColorspaceConversion() {
        let testUrl = Bundle(for: type(of: self))
            .url(forResource: "debug", withExtension: "png", subdirectory: "testdata")!
        let debugImageData = try! Data(contentsOf: testUrl)
        let image = UIImage.init(data: debugImageData)!
        let testTex = image.toMTLTexture(device: device)
        let testImg = MPSImage(texture: testTex, featureChannels: 3)

        let colorspaceConversion = ColorspaceConversionLayer(
            sourceColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
            destinationColorSpace: CGColorSpace(name: CGColorSpace.linearSRGB)!)
        let outputImg = colorspaceConversion.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()

        let expImg = device.makeMPSImage(
            width: 4,
            height: 4,
            values:
            [[1.00, 0.00, 0.00, 1.00,
                0.00, 0.00, 0.00, 0.00,
                0.00, 0.00, 0.00, 0.00,
                1.00, 0.00, 0.00, 1.00],
             [0.00, 1.00, 0.00, 0.00,
                1.00, 0.00, 1.00, 0.00,
                0.00, 1.00, 0.00, 1.00,
                0.00, 0.00, 1.00, 0.00],
             [0.00, 0.00, 1.00, 0.00,
                0.00, 1.00, 0.00, 1.00,
                1.00, 0.00, 1.00, 0.00,
                0.00, 1.00, 0.00, 0.00]])

        XCTAssert(outputImg.isLossyEqual(image: expImg, precision: 0))
    }

    func testImageLoad() {
        let testUrl = Bundle(for: type(of: self))
            .url(forResource: "tubingen", withExtension: "jpg", subdirectory: "testdata")!
        let testImg = UIImage.init(contentsOfFile: testUrl.path)!.toMPSImage(device: device)

        let colorspaceConversion = ColorspaceConversionLayer(
            sourceColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
            destinationColorSpace: CGColorSpace(name: CGColorSpace.linearSRGB)!)

        let outImg = colorspaceConversion
            .chain(MPSImageVariable(testImg))
            .forward(commandBuffer: commandBuffer)
        execute()

        let expUrl = Bundle(for: type(of: self))
            .url(forResource: "tubingen", withExtension: "npy", subdirectory: "testdata")!
        let expImg = MPSImage.fromNumpy(expUrl)

        XCTAssert(outImg.isLossyEqual(image: expImg, precision: 0))
    }

}
