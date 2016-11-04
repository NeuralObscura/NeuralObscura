//
//  ShaderRegistry.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 11/4/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import MetalPerformanceShaders

class ShaderRegistry {
    private var device: MTLDevice?
    private var library: MTLLibrary?
    private var registry = [String: MTLComputePipelineState]()
    static let sharedInstance = ShaderRegistry()

    private init() {}

    static func getOrLoad(name: String) -> MTLComputePipelineState {
        if let shader = sharedInstance.registry[name] {
            return shader
        } else {
            do {
                let shaderFunc = getLibrary().makeFunction(name: name)!
                sharedInstance.registry[name] = try getDevice().makeComputePipelineState(function: shaderFunc)
            } catch {
                fatalError("Unable to load shader: \(name)")
            }

            return sharedInstance.registry[name]!
        }
    }

    static func getDevice() -> MTLDevice {
        if let d = sharedInstance.device {
            return d
        } else {
            sharedInstance.device = MTLCreateSystemDefaultDevice()
            guard MPSSupportsMTLDevice(sharedInstance.device) else {
                fatalError("Error: Metal Performance Shaders not supported on this device")
            }
            return sharedInstance.device!
        }
    }

    static func getLibrary() -> MTLLibrary {
        if let l = sharedInstance.library {
            return l
        } else {
            sharedInstance.library = getDevice().newDefaultLibrary()
            return sharedInstance.library!
        }
    }
}
