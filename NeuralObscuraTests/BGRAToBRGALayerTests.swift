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
        let debugImageData = try! Data(contentsOf: testUrl)
        let image = UIImage.init(data: debugImageData)!
        let testTex = image.toMTLTexture(device: device)
        let testImg = MPSImage(texture: testTex, featureChannels: 4)

        let bgraToBRGA = BGRAToBRGALayer()
        let outputImg = bgraToBRGA.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()

        let expImg = device.makeMPSImage(
                width: 4,
                height: 4,
                values:
            [[1, 0, 0, 1,
              0, 0, 0, 0,
              0, 0, 0, 0,
              1, 0, 0, 1],
             [0, 0, 1, 0,
              0, 1, 0, 1,
              1, 0, 1, 0,
              0, 1, 0, 0],
             [0, 1, 0, 0,
              1, 0, 1, 0,
              0, 1, 0, 1,
              0, 0, 1, 0],
             [1, 1, 1, 1,
              1, 1, 1, 1,
              1, 1, 1, 1,
              1, 1, 1, 1]])

        print(testImg)
        print(outputImg)
        print(expImg)
        print("-----------------------------")
        XCTAssertEqual(outputImg, expImg)
    }
}
