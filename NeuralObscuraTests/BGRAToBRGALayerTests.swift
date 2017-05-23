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

class BGRAToBRGALayerTests: CommandEncoderBaseTest {
    func testBGRAToBRGALayer() {

        let testUrl = Bundle(for: type(of: self))
                .url(forResource: "debug", withExtension: "png", subdirectory: "testdata")!
        let testTex = try! textureLoader.newTexture(withContentsOf: testUrl)
        let testImg = MPSImage(texture: testTex, featureChannels: 4)

        let bgraToBRGA = BGRAToBRGALayer()
        let outputImg = bgraToBRGA.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()

        let expImg = device.makeMPSImage(
                width: 4,
                height: 4,
                values:
                [[255,   0,   0, 255,
                    0,   0,   0,   0,
                    0,   0,   0,   0,
                  255,   0,   0, 255],
                 [  0, 255,   0,   0,
                  255,   0, 255,   0,
                    0, 255,   0, 255,
                    0,   0, 255,   0],
                 [  0,   0, 255,   0,
                    0, 255,   0, 255,
                  255,   0, 255,   0,
                    0, 255,   0,   0]])

        XCTAssertEqual(outputImg, expImg)
    }
}
