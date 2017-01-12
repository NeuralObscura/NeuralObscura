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
    func testImageFormatMatchesExpectationsRGBA8Unorm() {
        let mergedChannelsAlpha = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 0, 0, 255],
                                   [0, 255, 0, 255], [0, 0, 255, 255], [0, 255, 0, 255], [0, 0, 255, 255],
                                   [0, 0, 255, 255], [0, 255, 0, 255], [0, 0, 255, 255], [0, 255, 0, 255],
                                   [255, 0, 0, 255], [0, 0, 255, 255], [0, 255, 0, 255], [255, 0, 0, 255]] as [[Float32]]

        let expImg = device.MakeMPSImage(width: 4,
                                             height: 4,
                                             featureChannels: 3,
                                             pixelFormat: .rgba8Unorm,
                                             values: mergedChannelsAlpha)

        let imagePath = Bundle.main.path(forResource: "debug", ofType: "png")!
        let image = UIImage.init(contentsOfFile: imagePath)!
        let inputMtlTexture = device.MakeMTLTexture(uiImage: image)

        let testImg = MPSImage(texture: inputMtlTexture, featureChannels: 3)

        XCTAssertEqual(testImg, expImg)
    }

    func testUIImageToTestRGBA8Unorm() {
        let debug2RawValues = [[255, 0, 0, 255], [200, 0, 0, 255],
                               [150, 0, 0, 255], [100, 0, 0, 255]] as [[Float32]]

        let expImg = device.MakeMPSImage(width: 2,
                                             height: 2,
                                             featureChannels: 3,
                                             pixelFormat: .rgba8Unorm,
                                             values: debug2RawValues)

        let debug2ImagePath = Bundle.main.path(forResource: "debug2", ofType: "png")!
        let image = UIImage.init(contentsOfFile: debug2ImagePath)!
        let inputMtlTexture = device.MakeMTLTexture(uiImage: image)
        let outputImg = MPSImage(texture: inputMtlTexture, featureChannels: 3)

        XCTAssertEqual(outputImg, expImg)
    }

    func testImageFormatMatchesExpectationsRGBA16Float() {
        let mergedChannelsAlpha = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 0, 0, 255],
                                   [0, 255, 0, 255], [0, 0, 255, 255], [0, 255, 0, 255], [0, 0, 255, 255],
                                   [0, 0, 255, 255], [0, 255, 0, 255], [0, 0, 255, 255], [0, 255, 0, 255],
                                   [255, 0, 0, 255], [0, 0, 255, 255], [0, 255, 0, 255], [255, 0, 0, 255]] as [[Float32]]

        let expImg = device.MakeMPSImage(width: 4,
                                             height: 4,
                                             featureChannels: 3,
                                             pixelFormat: .rgba16Float,
                                             values: mergedChannelsAlpha)

        let imagePath = Bundle.main.path(forResource: "debug", ofType: "png")!
        let image = UIImage.init(contentsOfFile: imagePath)!
        let inputMtlTexture = ShaderRegistry.getDevice().MakeMTLTexture(uiImage: image, pixelFormat: .rgba16Float)

        let testImg = MPSImage(texture: inputMtlTexture, featureChannels: 3)

        XCTAssertEqual(testImg, expImg)
    }

    func testUIImageToTestRGBA16Float() {
        let debug2RawValues = [[255, 0, 0, 255], [200, 0, 0, 255],
                               [150, 0, 0, 255], [100, 0, 0, 255]] as [[Float32]]

        let expImg = device.MakeMPSImage(width: 2,
                                             height: 2,
                                             featureChannels: 3,
                                             pixelFormat: .rgba16Float,
                                             values: debug2RawValues)

        let debug2ImagePath = Bundle.main.path(forResource: "debug2", ofType: "png")!
        let image = UIImage.init(contentsOfFile: debug2ImagePath)!
        let inputMtlTexture = ShaderRegistry.getDevice().MakeMTLTexture(uiImage: image, pixelFormat: .rgba16Float)
        let outputImg = MPSImage(texture: inputMtlTexture, featureChannels: 3)
        
        XCTAssertEqual(outputImg, expImg)
    }

    func testImageFormatMatchesExpectationsRGBA32Float() {
        let mergedChannelsAlpha = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 0, 0, 255],
                                   [0, 255, 0, 255], [0, 0, 255, 255], [0, 255, 0, 255], [0, 0, 255, 255],
                                   [0, 0, 255, 255], [0, 255, 0, 255], [0, 0, 255, 255], [0, 255, 0, 255],
                                   [255, 0, 0, 255], [0, 0, 255, 255], [0, 255, 0, 255], [255, 0, 0, 255]] as [[Float32]]

        let expImg = device.MakeMPSImage(width: 4,
                                             height: 4,
                                             featureChannels: 3,
                                             pixelFormat: .rgba32Float,
                                             values: mergedChannelsAlpha)

        let imagePath = Bundle.main.path(forResource: "debug", ofType: "png")!
        let image = UIImage.init(contentsOfFile: imagePath)!
        let inputMtlTexture = ShaderRegistry.getDevice().MakeMTLTexture(uiImage: image, pixelFormat: .rgba32Float)

        let testImg = MPSImage(texture: inputMtlTexture, featureChannels: 3)

        XCTAssertEqual(testImg, expImg)
    }

    func testUIImageToTestRGBA32Float() {
        let debug2RawValues = [[255, 0, 0, 255], [200, 0, 0, 255],
                               [150, 0, 0, 255], [100, 0, 0, 255]] as [[Float32]]

        let expImg = device.MakeMPSImage(width: 2,
                                             height: 2,
                                             featureChannels: 3,
                                             pixelFormat: .rgba32Float,
                                             values: debug2RawValues)

        let debug2ImagePath = Bundle.main.path(forResource: "debug2", ofType: "png")!
        let image = UIImage.init(contentsOfFile: debug2ImagePath)!
        let inputMtlTexture = ShaderRegistry.getDevice().MakeMTLTexture(uiImage: image, pixelFormat: .rgba32Float)
        let outputImg = MPSImage(texture: inputMtlTexture, featureChannels: 3)

        XCTAssertEqual(outputImg, expImg)
    }
}
