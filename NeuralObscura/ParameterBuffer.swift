//
//  StyleModel.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 9/20/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

protocol ParameterBuffer {
    var pointer: UnsafeMutablePointer<Float> { get }
    var length: Int { get }
    var count: Int { get }
}

class MemoryParameterBuffer: ParameterBuffer {
    var pointer: UnsafeMutablePointer<Float>
    var count: Int
    var length: Int
    
    init(_ values: [Float]) {
        self.count = values.count
        self.length = self.count * MemoryLayout<Float>.size
        self.pointer = values.withUnsafeBufferPointer({ (buf) -> UnsafeMutablePointer<Float> in
            let pointer = UnsafeMutablePointer<Float>.allocate(capacity: buf.count)
            for (i, e) in buf.enumerated() {
                pointer[i] = e
            }
            return pointer
        })
    }
    
    init(_ values: Float...) {
        self.count = 1
        self.length = self.count * MemoryLayout<Float>.size
        self.pointer = UnsafeMutablePointer<Float>.allocate(capacity: 1)
        for (i,e) in values.enumerated() {
            pointer[i] = e
        }
    }
    
    deinit {
        pointer.deallocate(capacity: count)
    }
}


class FileParameterBuffer: ParameterBuffer {
    let rawFileName: String
    let modelName: String
    
    var pointer: UnsafeMutablePointer<Float>
    var count: Int
    var length: Int
    
    private var fd: CInt!
    private var hdr: UnsafeMutableRawPointer!
    private var fileSize: UInt64 = 0
    
    
    init(modelName: String, rawFileName: String) {
        self.modelName = modelName
        self.rawFileName = rawFileName
        
        let path = Bundle.main.path(forResource: rawFileName, ofType: "dat", inDirectory: self.modelName + "_model_data")
        
        assert(path != nil, "Error: failed to find file \(rawFileName)")
        
        do {
            let attr = try FileManager.default.attributesOfItem(atPath: path!)
            
            if let fsize = attr[FileAttributeKey.size] as? NSNumber {
                length = fsize.intValue
            } else {
                fatalError("Failed to get a size attribute from path: \(path)")
            }
        } catch {
            fatalError("Error: \(error)")
        }
        
        // open file descriptors in read-only mode to parameter files
        let fd = open( path!, O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)
        
        assert(fd != -1, "Error: failed to open file at \""+path!+"\"  errno = \(errno)\n")
        
        let hdr = mmap(nil, length, PROT_READ, MAP_FILE | MAP_SHARED, fd, 0)
        
        count = length / MemoryLayout<Float>.size
        pointer = hdr!.bindMemory(to: Float.self, capacity: count)
        
        if pointer == UnsafeMutablePointer<Float>(bitPattern: -1) {
            fatalError("Error: mmap failed, errno = \(errno)")
        }
    }
    
    deinit {
        print("deinit \(self) \(modelName) \(rawFileName)")
        
        if let hdr = hdr {
            let result = munmap(hdr, Int(fileSize))
            assert(result == 0, "Error: munmap failed, errno = \(errno)")
        }
        if let fd = fd {
            close(fd)
        }
    }
}
