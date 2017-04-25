//
//  MPSImageExtensionsTests.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 1/18/17.
//  Copyright © 2017 Paul Bergeron. All rights reserved.
//

import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class MPSImageExtensionsTests: CommandEncoderBaseTest {

//    func testLoadFromNumpy() {
//        let url = Bundle(for: type(of: self))
//            .url(forResource: "test_loadFromNumpy_data", withExtension: "npy", subdirectory: "testdata")!
//        let outputImg = MPSImage.loadFromNumpy(url)
//
//        let expImg = [1.1, 2.2, 3.3, 4.4] as [Float32]
//
//        XCTAssert(outputImg.isLossyEqual(values: expImg, precision: 2))
//    }

    func testFloat32ToString() {
        let debug2ImageUrl = Bundle(for: type(of: self))
            .url(forResource: "debug2", withExtension: "png", subdirectory: "testdata")!
        let debug2ImageData = try! Data(contentsOf: debug2ImageUrl)
        let image = UIImage.init(data: debug2ImageData)!
        let inputMtlTexture = ShaderRegistry.getDevice().makeMTLTexture(uiImage: image,
                                                                        pixelFormat: .rgba32Float)

        let testImg = MPSImage(texture: inputMtlTexture, featureChannels: 3)

        let expString = "255.00 200.00 150.00 100.00 \n0.00 0.00 0.00 0.00 \n0.00 0.00 0.00 0.00 \n255.00 255.00 255.00 255.00 \n"

        XCTAssertEqual(testImg.Float32ToString(), expString)
    }

    func testFloat16ToString() {
        let debug2ImageUrl = Bundle(for: type(of: self))
            .url(forResource: "debug2", withExtension: "png", subdirectory: "testdata")!
        let debug2ImageData = try! Data(contentsOf: debug2ImageUrl)
        let image = UIImage.init(data: debug2ImageData)!
        let inputMtlTexture = ShaderRegistry.getDevice().makeMTLTexture(uiImage: image,
                                                                        pixelFormat: .rgba16Float)

        let testImg = MPSImage(texture: inputMtlTexture, featureChannels: 3)

        let expString = "255.00 200.00 150.00 100.00 \n0.00 0.00 0.00 0.00 \n0.00 0.00 0.00 0.00 \n255.00 255.00 255.00 255.00 \n"

        XCTAssertEqual(testImg.Float16ToString(), expString)
    }
}
