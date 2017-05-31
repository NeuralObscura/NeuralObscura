//
//  BGRAToBRGALayerTests.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 11/26/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class UnormToHalfLayerTests: CommandEncoderBaseTest {
    func testUnormToHalfLayer() {

        let testUrl = Bundle(for: type(of: self))
                .url(forResource: "debug", withExtension: "png", subdirectory: "testdata")!
        let debugImageData = try! Data(contentsOf: testUrl)
        let image = UIImage.init(data: debugImageData)!
        let testTex = image.toMTLTexture(device: device)
        let testImg = MPSImage(texture: testTex, featureChannels: 4)

        let unorm_to_half = UnormToHalfLayer()
        let outputImg = unorm_to_half.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()

        let expImg = device.makeMPSImage(
                width: 4,
                height: 4,
                values:
            [[255, 0, 0, 255,
              0, 0, 0, 0,
              0, 0, 0, 0,
              255, 0, 0, 255],
             [0, 255, 0, 0,
              255, 0, 255, 0,
              0, 255, 0, 255,
              0, 0, 255, 0],
             [0, 0, 255, 0,
              0, 255, 0, 255,
              255, 0, 255, 0,
              0, 255, 0, 0],
             [0, 0, 0, 0,
              0, 0, 0, 0,
              0, 0, 0, 0,
              0, 0, 0, 0]])

        XCTAssertEqual(outputImg, expImg)
    }
}
