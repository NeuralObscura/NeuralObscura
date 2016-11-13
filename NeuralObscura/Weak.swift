//
//  Weak.swift
//  NeuralObscura
//
//  Created by Edward Knox on 11/12/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation


public class Weak<T: AnyObject>: NSObject {
    public private(set) weak var value: T?
    
    public init(_ value: T?) {
        self.value = value
    }
}
