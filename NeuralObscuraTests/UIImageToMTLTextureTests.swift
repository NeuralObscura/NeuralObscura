//
//  UIImageToMTLTextureTests.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 11/16/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class UIImageToMTLTextureTests: CommandEncoderBaseTest {

    func testUIImageToTest() {
        let debug2RawValues = [[255, 0, 0, 255], [200, 0, 0, 255],
                               [150, 0, 0, 255], [100, 0, 0, 255]] as [[Float32]]

        let expImg = device.MakeTestMPSImage(width: 2,
                                             height: 2,
                                             featureChannels: 3,
                                             pixelFormat: .rgba16Float,
                                             values: debug2RawValues)

        let debug2ImagePath = Bundle.main.path(forResource: "debug2", ofType: "png")!
        let image = UIImage.init(contentsOfFile: debug2ImagePath)!
        let inputMtlTexture = image.createMTLTextureForDevice(device: ShaderRegistry.getDevice())
        let outputImg = MPSImage(texture: inputMtlTexture, featureChannels: 3)

        XCTAssertEqual(outputImg, expImg)
    }

}
