//
//  ShaderRegistry.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 11/4/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import MetalPerformanceShaders

class ShaderRegistry {
    private let device: MTLDevice = MTLCreateSystemDefaultDevice()!
    private let library: MTLLibrary = MTLCreateSystemDefaultDevice()!.newDefaultLibrary()!
    private var registry = [String: MTLComputePipelineState]()
    static let sharedInstance = ShaderRegistry()

    private init() {}

    static func getOrLoad(name: String) -> MTLComputePipelineState {
        if let shader = sharedInstance.registry[name] {
            return shader
        } else {
            do {
                let shaderFunc = sharedInstance.library.makeFunction(name: name)!
                sharedInstance.registry[name] = try sharedInstance.device.makeComputePipelineState(function: shaderFunc)
            } catch {
                fatalError("Unable to load shader: \(name)")
            }

            return sharedInstance.registry[name]!
        }
    }
}
