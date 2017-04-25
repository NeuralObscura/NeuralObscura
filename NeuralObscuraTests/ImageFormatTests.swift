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
//    func testImageFormatMatchesExpectationsRGBA8Unorm() {
//        let mergedChannelsAlpha = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 1],
//                                   [0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1],
//                                   [0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1],
//                                   [1, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1]]
//        // TODO: Finish converting this, I'm too tired to do it right now
//        let expImg = device.makeMPSImage(width: 4,
//                                         height: 4,
//                                         values: mergedChannelsAlpha)
////            [[1,0,0,1,
////                                                   0,0,0,0,
////                                                   0,0,0,0,
////                                                   0,0,0,0],
////                                                  [0,1,0,0,
////                                                   1,0,0,0,
////                                                   0,0,0,0,
////                                                   0,0,0,0],
////                                                  [0,0,1,0,
////                                                   0,0,0,0,
////                                                   0,0,0,0,
////                                                   0,0,0,0],
////                                                  [1,1,1,1,
////                                                   1,0,0,0,
////                                                   0,0,0,0,
////                                                   0,0,0,0]])
//        
//        let debugImageUrl = Bundle(for: type(of: self))
//            .url(forResource: "debug", withExtension: "png", subdirectory: "testdata")!
//        let debugImageData = try! Data(contentsOf: debugImageUrl)
//        let image = UIImage.init(data: debugImageData)!
//        let inputMtlTexture = device.makeMTLTexture(uiImage: image)
//
//        let testImg = MPSImage(texture: inputMtlTexture, featureChannels: 3)
//
//        XCTAssertEqual(testImg, expImg)
//    }
//
//    func testUIImageToTestRGBA8Unorm() {
//        let debug2RawValues = [[255, 0, 0, 255], [200, 0, 0, 255],
//                               [150, 0, 0, 255], [100, 0, 0, 255]] as [[Float32]]
//
//        // TODO: convert
////                                             featureChannels: 3,
//        let expImg = device.makeMPSImage(width: 2,
//                                             height: 2,
//                                             values: [[255, 0, 0, 255], [200, 0, 0, 255],
//                                                      [150, 0, 0, 255], [100, 0, 0, 255]])
//
//        let debug2ImageUrl = Bundle(for: type(of: self))
//            .url(forResource: "debug2", withExtension: "png", subdirectory: "testdata")!
//        let debug2ImageData = try! Data(contentsOf: debug2ImageUrl)
//        let image = UIImage.init(data: debug2ImageData)!
//        let inputMtlTexture = device.makeMTLTexture(uiImage: image)
//        let outputImg = MPSImage(texture: inputMtlTexture, featureChannels: 3)
//
//        XCTAssertEqual(outputImg, expImg)
//    }
//
//    func testImageFormatMatchesExpectationsRGBA16Float() {
//        let mergedChannelsAlpha = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 0, 0, 255],
//                                   [0, 255, 0, 255], [0, 0, 255, 255], [0, 255, 0, 255], [0, 0, 255, 255],
//                                   [0, 0, 255, 255], [0, 255, 0, 255], [0, 0, 255, 255], [0, 255, 0, 255],
//                                   [255, 0, 0, 255], [0, 0, 255, 255], [0, 255, 0, 255], [255, 0, 0, 255]] as [[Float32]]
//
////                                             featureChannels: 3,
//        let expImg = device.makeMPSImage(width: 4,
//                                             height: 4,
//                                             values: mergedChannelsAlpha)
//
//        let debugImageUrl = Bundle(for: type(of: self))
//            .url(forResource: "debug", withExtension: "png", subdirectory: "testdata")!
//        let debugImageData = try! Data(contentsOf: debugImageUrl)
//        let image = UIImage.init(data: debugImageData)!
//        let inputMtlTexture = ShaderRegistry.getDevice().makeMTLTexture(uiImage: image, pixelFormat: .rgba16Float)
//
//        let testImg = MPSImage(texture: inputMtlTexture, featureChannels: 3)
//
//        XCTAssertEqual(testImg, expImg)
//    }
//
//    func testUIImageToTestRGBA16Float() {
//        let debug2RawValues = [[255, 0, 0, 255], [200, 0, 0, 255],
//                               [150, 0, 0, 255], [100, 0, 0, 255]] as [[Float32]]
//
////                                             featureChannels: 3,
//        let expImg = device.makeMPSImage(width: 2,
//                                             height: 2,
//                                             values: debug2RawValues)
//
//        let debugImageUrl = Bundle(for: type(of: self))
//            .url(forResource: "debug2", withExtension: "png", subdirectory: "testdata")!
//        let debugImageData = try! Data(contentsOf: debugImageUrl)
//        let image = UIImage.init(data: debugImageData)!
//        let inputMtlTexture = ShaderRegistry.getDevice().makeMTLTexture(uiImage: image, pixelFormat: .rgba16Float)
//        let outputImg = MPSImage(texture: inputMtlTexture, featureChannels: 3)
//        
//        XCTAssertEqual(outputImg, expImg)
//    }
//
//    func testImageFormatMatchesExpectationsRGBA32Float() {
//        let mergedChannelsAlpha = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 0, 0, 255],
//                                   [0, 255, 0, 255], [0, 0, 255, 255], [0, 255, 0, 255], [0, 0, 255, 255],
//                                   [0, 0, 255, 255], [0, 255, 0, 255], [0, 0, 255, 255], [0, 255, 0, 255],
//                                   [255, 0, 0, 255], [0, 0, 255, 255], [0, 255, 0, 255], [255, 0, 0, 255]] as [[Float32]]
//
////                                         featureChannels: 3,
//        let expImg = device.makeMPSImage(width: 4,
//                                         height: 4,
//                                         values: mergedChannelsAlpha)
//
//        let debugImageUrl = Bundle(for: type(of: self))
//            .url(forResource: "debug", withExtension: "png", subdirectory: "testdata")!
//        let debugImageData = try! Data(contentsOf: debugImageUrl)
//        let image = UIImage.init(data: debugImageData)!
//        let inputMtlTexture = ShaderRegistry.getDevice().makeMTLTexture(uiImage: image, pixelFormat: .rgba32Float)
//
//        let testImg = MPSImage(texture: inputMtlTexture, featureChannels: 3)
//
//        XCTAssertEqual(testImg, expImg)
//    }
//
//    func testUIImageToTestRGBA32Float() {
//        let debug2RawValues = [[255, 0, 0, 255], [200, 0, 0, 255],
//                               [150, 0, 0, 255], [100, 0, 0, 255]] as [[Float32]]
//
////                                         featureChannels: 3,
//        let expImg = device.makeMPSImage(width: 2,
//                                         height: 2,
//                                         values: debug2RawValues)
//
//        let debug2ImageUrl = Bundle(for: type(of: self))
//            .url(forResource: "debug2", withExtension: "png", subdirectory: "testdata")!
//        let debug2ImageData = try! Data(contentsOf: debug2ImageUrl)
//        let image = UIImage.init(data: debug2ImageData)!
//        let inputMtlTexture = ShaderRegistry.getDevice().makeMTLTexture(uiImage: image, pixelFormat: .rgba32Float)
//        let outputImg = MPSImage(texture: inputMtlTexture, featureChannels: 3)
//
//        XCTAssertEqual(outputImg, expImg)
//    }
}
