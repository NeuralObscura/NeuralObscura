//
//  CommandEncoder.swift
//  NeuralObscura
//
//  Created by Edward Knox on 9/29/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

protocol UnaryChain {
    mutating func chain(_ top: CommandEncoder) -> CommandEncoder
}

protocol BinaryChain {
    mutating func chain(_ topA: CommandEncoder, _ topB: CommandEncoder) -> CommandEncoder
}
