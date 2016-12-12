//
//  DebugImageRegistry.swift
//  NeuralObscura
//
//  Created by Edward Knox on 12/11/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import MetalPerformanceShaders

class DebugFrameStorage {
    static let sharedInstance = DebugFrameStorage()
    
    private var frames = [MPSImage]()
    
    private init() {}
    
    static func reset() {
        sharedInstance.frames = [MPSImage]()
    }
    
    static func registerFrame(_ frame: MPSImage) {
        sharedInstance.frames.append(frame)
    }
    
    static func getFrames() -> [MPSImage] {
        return sharedInstance.frames
    }
}
