
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

class CommandEncoderBaseTest: XCTestCase {
    var device: MTLDevice!
    var textureLoader: MTKTextureLoader!
    var commandQueue: MTLCommandQueue!
    var commandBuffer: MTLCommandBuffer!
    
    override func setUp() {
        super.setUp()
        device = ShaderRegistry.getDevice()
        assert(MPSSupportsMTLDevice(device) == true)
        textureLoader = MTKTextureLoader(device: device)
        commandQueue = device.makeCommandQueue()
        commandBuffer = commandQueue.makeCommandBuffer()
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
}
