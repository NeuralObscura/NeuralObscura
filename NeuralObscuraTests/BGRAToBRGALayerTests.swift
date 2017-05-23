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
    func testRGBAToBRGALayer() {
        let textureDesc = MTLTextureDescriptor()
        textureDesc.textureType = .type2D
        textureDesc.width = 2
        textureDesc.height = 2
        textureDesc.pixelFormat = .rgba16Float

        let testUrl = Bundle(for: type(of: self))
                .url(forResource: "debug", withExtension: "png", subdirectory: "testdata")!
        let testTex = try! textureLoader.newTexture(withContentsOf: testUrl)
        let testImg = MPSImage(texture: testTex, featureChannels: 4)

        let bgraToBRGA = BGRAToBRGALayer()
        let outputImg = bgraToBRGA.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()

        let expImg = MPSImage(texture:
            device.makeMTLTexture(
                    textureDesc: textureDesc,
                    values: [1.0,3.0,4.0,2.0,
                             2.0,4.0,3.0,1.0,
                             3.0,2.0,2.0,3.0,
                             4.0,1.0,1.0,4.0]),
                    featureChannels: 4)

        XCTAssertEqual(outputImg, expImg)
    }
}
