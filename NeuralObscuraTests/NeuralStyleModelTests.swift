//
//  CommandEncoderBaseTest.swift
//  NeuralObscura
//
//  Created by Edward Knox on 10/9/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class NeuralStyleModelTests: CommandEncoderBaseTest {
    
    override func setUp() {
        super.setUp()
        print("==== BEGIN TEST OUTPUT ====")
    }
    
    func testCorrectness() {
        let model = NeuralStyleModel(modelName: "composition", debug: true)
        let image = UIImage.init(contentsOfFile: Bundle.main.path(forResource: "tubingen", ofType: "jpg")!)!
        let texture = image.createMTLTextureForDevice(device: ShaderRegistry.getDevice())
        let mpsImage = MPSImage(texture: texture, featureChannels: 3)
        model.execute(commandQueue: commandQueue, sourceImage: mpsImage)
        print(DebugFrameStorage.getFrames().count)
        for frame in DebugFrameStorage.getFrames() {
            print(frame)
        }
    }
    
    override func tearDown() {
        print("==== END TEST OUTPUT ====")
        super.tearDown()
    }
    
}
