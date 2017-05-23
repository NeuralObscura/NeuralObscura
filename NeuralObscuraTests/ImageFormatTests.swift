//
//  ImageFormatTests.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 11/9/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class ImageFormatTests: CommandEncoderBaseTest {

    func testImageFormatMatchesExpectationsBGRA8Unorm() {
        let expImg = [0.0, 0.0, 255.0, 255.0,
                      0.0, 255.0, 0.0, 255.0,
                      255.0, 0.0, 0.0, 255.0,
                      0.0, 0.0, 255.0, 255.0,
                      0.0, 255.0, 0.0, 255.0,
                      255.0, 0.0, 0.0, 255.0,
                      0.0, 255.0, 0.0, 255.0,
                      255.0, 0.0, 0.0, 255.0,
                      255.0, 0.0, 0.0, 255.0,
                      0.0, 255.0, 0.0, 255.0,
                      255.0, 0.0, 0.0, 255.0,
                      0.0, 255.0, 0.0, 255.0,
                      0.0, 0.0, 255.0, 255.0,
                      255.0, 0.0, 0.0, 255.0,
                      0.0, 255.0, 0.0, 255.0,
                      0.0, 0.0, 255.0, 255.0] as [Float32]


        let debugImageUrl = Bundle(for: type(of: self))
            .url(forResource: "debug", withExtension: "png", subdirectory: "testdata")!
        let debugImageData = try! Data(contentsOf: debugImageUrl)
        let image = UIImage.init(data: debugImageData)!
        let inputMtlTexture = image.toMTLTexture(device: device)


        let outputImg = MPSImage(texture: inputMtlTexture, featureChannels: 4)


        XCTAssert(outputImg.isLossyEqual(values: expImg, precision: 3))
    }

    func testUIImageToTestBGRA8Unorm() {
        let expImg = [0, 0, 255, 255,
                      0, 0, 200, 255,
                      0, 0, 150, 255,
                      0, 0, 100, 255] as [Float32]

        let debug2ImageUrl = Bundle(for: type(of: self))
            .url(forResource: "debug2", withExtension: "png", subdirectory: "testdata")!
        let debug2ImageData = try! Data(contentsOf: debug2ImageUrl)
        let image = UIImage.init(data: debug2ImageData)!
        let inputMtlTexture = image.toMTLTexture(device: device)
        let outputImg = MPSImage(texture: inputMtlTexture, featureChannels: 4)

        XCTAssert(outputImg.isLossyEqual(values: expImg, precision: 3))
    }
}
