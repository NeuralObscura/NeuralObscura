//
//  Conversions.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 11/8/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Accelerate

class Conversions {
    static func float32toFloat16(_ values: [Float32]) -> [UInt16] {
        var input = values
        var inputBuffer = vImage_Buffer(data: &input, height: 1, width: UInt(values.count), rowBytes: values.count * 4)
        var output = [UInt16](repeating: 0, count: values.count)
        var outputBuffer = vImage_Buffer(data: &output, height: 1, width: UInt(values.count), rowBytes: values.count * 2)

        if vImageConvert_PlanarFtoPlanar16F(&inputBuffer, &outputBuffer, 0) != kvImageNoError {
            fatalError("Couldn't convert from float32 to float16")
        }
        return output
    }

    static func float16toFloat32(_ values: [UInt16]) -> [Float32] {
        var input = values
        var inputBuffer = vImage_Buffer(data: &input, height: 1, width: UInt(values.count), rowBytes: values.count * 2)
        var output = [Float32](repeating: 0, count: values.count)
        var outputBuffer = vImage_Buffer(data: &output, height: 1, width: UInt(values.count), rowBytes: values.count * 4)

        if vImageConvert_Planar16FtoPlanarF(&inputBuffer, &outputBuffer, 0) != kvImageNoError {
            fatalError("Couldn't convert from float16 to float32")
        }
        return output
    }
}
