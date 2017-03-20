//
//  Variables.swift
//  NeuralObscura
//
//  Created by Edward Knox on 3/11/17.
//  Copyright © 2017 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders


// class MPSImageVariable: AnyCommandEncoder<MPSImage> {
//     private class _MPSImageVariable: CommandEncoder {
//         private let value: () -> MPSImage
//         
//         init(_ value: @escaping () -> MPSImage) {
//             self.value = value
//         }
//         
//         func forward(commandBuffer: MTLCommandBuffer) -> MPSImage {
//             return value()
//         }
//         
//         func registerConsumer() {}
//     }
//     
//     private var value: MPSImage?
//     
//     convenience init(_ value: MPSImage) {
//         self.init()
//         self.value = value
//     }
//     
//     init() {
//         super.init(_MPSImageVariable({ [weak value] () -> MPSImage in value! }))
//     }
//     
//     func setValue(_ value: MPSImage) {
//         self.value = value
//     }
// }

class MPSImageVariable: AnyCommandEncoder<MPSImage> {
    override init() {
        super.init()
    }
    
    init(_ value: MPSImage) {
        super.init()
        _value = value
    }
    
    func setValue(_ value: MPSImage) {
        _value = value
    }
}

class MTLBufferVariable: AnyCommandEncoder<MTLBuffer> {
    override init() {
        super.init()
    }
    
    init(_ value: MTLBuffer) {
        super.init()
        _value = value
    }
    
    func setValue(_ value: MTLBuffer) {
        _value = value
    }
}
