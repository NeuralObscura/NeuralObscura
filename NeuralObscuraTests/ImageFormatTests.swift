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
    func testImageFormatMatchesExpectations() {
        let mergedChannelsAlpha = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 0, 0, 255],
                                   [0, 255, 0, 255], [0, 0, 255, 255], [0, 255, 0, 255], [0, 0, 255, 255],
                                   [0, 0, 255, 255], [0, 255, 0, 255], [0, 0, 255, 255], [0, 255, 0, 255],
                                   [255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 0, 0, 255]] as [[Float32]]

        let expImg = device.MakeTestMPSImage(width: 4,
                                             height: 4,
                                             featureChannels: 3,
                                             pixelFormat: MTLPixelFormat.rgba8Unorm,
                                             values: mergedChannelsAlpha)

        let imagePath = Bundle.main.path(forResource: "debug", ofType: "png")!
        let image = UIImage.init(contentsOfFile: imagePath)!
        let inputMtlTexture = image.createMTLTextureForDevice(device: ShaderRegistry.getDevice())

        let testImg = MPSImage(texture: inputMtlTexture, featureChannels: 3)

        XCTAssertEqual(testImg, expImg)
    }
}
