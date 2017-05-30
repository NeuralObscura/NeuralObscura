//
//  IncrementalModelGroundTruthTests.swift
//  NeuralObscura
//
//  Created by Edward Knox on 5/18/17.
//  Copyright © 2017 Paul Bergeron. All rights reserved.
//

import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class IncrementalModelGroundTruthTests: CommandEncoderBaseTest {
    
    func testStage0() {
        let testUrl = Bundle(for: type(of: self))
            .url(forResource: "tubingen", withExtension: "jpg", subdirectory: "testdata")!
//        let testImg = UIImage.init(contentsOfFile: testUrl.path)!.toMPSImage(device: device)
        let options: [String : NSObject] = [
            MTKTextureLoaderOptionTextureUsage: MTLTextureUsage.shaderRead.rawValue as NSNumber,
            MTKTextureLoaderOptionAllocateMipmaps: false as NSNumber,
            MTKTextureLoaderOptionGenerateMipmaps: false as NSNumber,
            MTKTextureLoaderOptionSRGB: false as NSNumber,
            MTKTextureLoaderOptionTextureStorageMode: MTLStorageMode.shared.rawValue as NSNumber
        ]
        let texture = try! textureLoader.newTexture(withContentsOf: testUrl, options: options)
        let testImg = MPSImage.init(texture: texture, featureChannels: 4)
        
        let outImg = BGRAToBRGALayer()
            .chain(MPSImageVariable(testImg))
            .forward(commandBuffer: commandBuffer)
        execute()
        
        let expUrl = Bundle(for: type(of: self))
            .url(forResource: "inc_gt_0", withExtension: "npy", subdirectory: "testdata")!
        let expImg = MPSImage.fromNumpy(expUrl)
        
        XCTAssert(outImg.isLossyEqual(image: expImg, precision: -1))
        print(outImg)
        print(expImg)
    }
}
