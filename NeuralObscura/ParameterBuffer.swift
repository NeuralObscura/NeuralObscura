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
    func pointer() -> UnsafeMutablePointer<Float>
    func lengthInBytes() -> Int
}

class MemoryParameterBuffer: ParameterBuffer {
    private var ptr: UnsafeMutablePointer<Float>!
    private var count: Int!
    private let length: Int
    
    init( _ values: [Float]) {
        self.count = values.count
        self.length = self.count * MemoryLayout<Float>.size
        self.ptr = values.withUnsafeBufferPointer({ (buf) -> UnsafeMutablePointer<Float> in
            let ptr = UnsafeMutablePointer<Float>.allocate(capacity: buf.count)
            for (i, e) in buf.enumerated() {
                ptr[i] = e
            }
            return ptr
        })
    }
    
    init(_ value: Float) {
        self.count = 1
        self.length = self.count * MemoryLayout<Float>.size
        self.ptr = UnsafeMutablePointer<Float>.allocate(capacity: 1)
        self.ptr.pointee = value
    }
    
    deinit {
        ptr!.deallocate(capacity: count)
    }
    
    func pointer() -> UnsafeMutablePointer<Float> {
        return ptr
    }

    func lengthInBytes() -> Int {
        return length
    }
}


class FileParameterBuffer: ParameterBuffer {
    private var fd: CInt!
    private var hdr: UnsafeMutableRawPointer!
    private var ptr: UnsafeMutablePointer<Float>!
    private var fileSize: UInt64 = 0
    var length: Int
    let modelName: String
    let rawFileName: String
    
    init(modelName: String, rawFileName: String) {
        self.modelName = modelName
        self.rawFileName = rawFileName
        self.length = 0
        
        loadRawFile(rawFileName: self.rawFileName)
    }
    
    private func loadRawFile(rawFileName: String) {
        let path = Bundle.main.path(forResource: rawFileName, ofType: "dat", inDirectory: self.modelName + "_model_data")
        
        assert(path != nil, "Error: failed to find file \(rawFileName)")
        
        do {
            let attr = try FileManager.default.attributesOfItem(atPath: path!)
            
            if let fsize = attr[FileAttributeKey.size] as? NSNumber {
                fileSize = fsize.uint64Value
            } else {
                fatalError("Failed to get a size attribute from path: \(path)")
            }
        } catch {
            fatalError("Error: \(error)")
        }

        self.length = Int(fileSize)
        
        // open file descriptors in read-only mode to parameter files
        let fd = open( path!, O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)
        
        assert(fd != -1, "Error: failed to open file at \""+path!+"\"  errno = \(errno)\n")
        
        let hdr = mmap(nil, Int(fileSize), PROT_READ, MAP_FILE | MAP_SHARED, fd, 0)
        
        print("Opened: \(path!) (\(fileSize))")
        
        let floatCount = Int(fileSize) / MemoryLayout<Float>.size
        ptr = hdr!.bindMemory(to: Float.self, capacity: floatCount)
        
        if ptr == UnsafeMutablePointer<Float>(bitPattern: -1) {
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
    
    func pointer() -> UnsafeMutablePointer<Float> {
        return ptr
    }

    func lengthInBytes() -> Int {
        return length
    }

}
