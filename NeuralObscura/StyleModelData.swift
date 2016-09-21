//
//  StyleModel.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 9/20/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class StyleModelData {
    private var fd: CInt!
    private var hdr: UnsafeMutableRawPointer!
    private var ptr: UnsafeMutablePointer<Float>!
    private var fileSize: UInt64 = 0
    let modelName: String
    let rawFileName: String

    init(modelName: String, rawFileName: String) {
        self.modelName = modelName
        self.rawFileName = rawFileName
    }

    private func loadRawFile(rawFileName: String) {
        let path = Bundle.main.path( forResource: rawFileName, ofType: "dat")
        do {
            let attr = try FileManager.default.attributesOfItem(atPath: path!)

            if let fsize = attr[FileAttributeKey.size] as? UInt64 {
                fileSize = fsize
            } else {
                print("Failed to get a size attribute from path: \(path)")
            }
        } catch {
            print("Error: \(error)")
        }

        // open file descriptors in read-only mode to parameter files
        let fd = open( path!, O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)

        assert(fd != -1, "Error: failed to open output file at \""+path!+"\"  errno = \(errno)\n")

        let hdr = mmap(nil, Int(fileSize), PROT_READ, MAP_FILE | MAP_SHARED, fd, 0)

        print("Opened: "+path!)

        let numBytes = Int(fileSize) / MemoryLayout<Float>.size
        ptr = hdr!.bindMemory(to: Float.self, capacity: numBytes)
        if ptr == UnsafeMutablePointer<Float>(bitPattern: -1) {
            print("Error: mmap failed, errno = \(errno)")
        }
    }

    deinit{
        print("deinit \(self)")

        if let hdr = hdr {
            let result = munmap(hdr, Int(fileSize))
            assert(result == 0, "Error: munmap failed, errno = \(errno)")
        }
        if let fd = fd {
            close(fd)
        }
    }

}
