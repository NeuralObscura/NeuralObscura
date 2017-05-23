//
//  MTLPixelFormat.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 11/7/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//
import MetalPerformanceShaders

extension MTLPixelFormat {
    
    var channelCount: Int {
        switch self {
        case .rgba8Unorm:
            return 4
        case .r8Unorm:
            return 1
        case .rgba32Float:
            return 4
        case .rgba16Float:
            return 4
        case .r32Float:
            return 1
        case .r16Float:
            return 1
        case .bgra8Unorm_srgb:
            return 4
        case .bgra8Unorm:
            return 4
        default:
            fatalError("Unknown MTLPixelFormat: \(self)")
        }
    }

    var sizeOfDataType: Int {
        switch self {
        case .rgba8Unorm:
            return MemoryLayout<UInt8>.size
        case .r8Unorm:
            return MemoryLayout<UInt8>.size
        case .rgba32Float:
            return MemoryLayout<Float32>.size
        case .r32Float:
            return MemoryLayout<Float32>.size
        case .r16Float:
            return ExpectedFloat16Size
        case .rgba16Float:
            return ExpectedFloat16Size
        case .bgra8Unorm_srgb:
            return MemoryLayout<UInt8>.size
        case .bgra8Unorm:
            return MemoryLayout<UInt8>.size
        default:
            fatalError("Unknown MTLPixelFormat: \(self)")
        }
    }

    func featureChannelsToSlices(_ featureChannels: Int) -> Int {
        return Int(ceil(Float(featureChannels) / Float(self.channelCount)))
    }

    func typedSize(width: Int, height: Int) -> Int {
        return width * height * self.channelCount
    }

    func featureChannels(_ arrayLength: Int) -> Int {
        return arrayLength * self.channelCount
    }

    public var description: String {
        switch self {
        case .a8Unorm:
            return "a8Unorm"
        case .r8Unorm:
            return "r8Unorm"
        case .r8Unorm_srgb:
            return "r8Unorm_srgb"
        case .r8Snorm:
            return "r8Snorm"
        case .r8Uint:
            return "r8Uint"
        case .r8Sint:
            return "r8Sint"
        case .r16Unorm:
            return "r16Unorm"
        case .r16Snorm:
            return "r16Snorm"
        case .r16Uint:
            return "r16Uint"
        case .r16Sint:
            return "r16Sint"
        case .r16Float:
            return "r16Float"
        case .rg8Unorm:
            return "rg8Unorm"
        case .rg8Unorm_srgb:
            return "rg8Unorm_srgb"
        case .rg8Snorm:
            return "rg8Snorm"
        case .rg8Uint:
            return "rg8Uint"
        case .rg8Sint:
            return "rg8Sint"
        case .b5g6r5Unorm:
            return "b5g6r5Unorm"
        case .a1bgr5Unorm:
            return "a1bgr5Unorm"
        case .abgr4Unorm:
            return "abgr4Unorm"
        case .bgr5A1Unorm:
            return "bgr5A1Unorm"
        case .r32Uint:
            return "r32Uint"
        case .r32Sint:
            return "r32Sint"
        case .r32Float:
            return "r32Float"
        case .rg16Unorm:
            return "rg16Unorm"
        case .rg16Snorm:
            return "rg16Snorm"
        case .rg16Uint:
            return "rg16Uint"
        case .rg16Sint:
            return "rg16Sint"
        case .rg16Float:
            return "rg16Float"
        case .rgba8Unorm:
            return "rgba8Unorm"
        case .rgba8Unorm_srgb:
            return "rgba8Unorm_srgb"
        case .rgba8Snorm:
            return "rgba8Snorm"
        case .rgba8Uint:
            return "rgba8Uint"
        case .rgba8Sint:
            return "rgba8Sint"
        case .bgra8Unorm:
            return "bgra8Unorm"
        case .bgra8Unorm_srgb:
            return "bgra8Unorm_srgb"
        case .rgb10a2Unorm:
            return "rgb10a2Unorm"
        case .rgb10a2Uint:
            return "rgb10a2Uint"
        case .rg11b10Float:
            return "rg11b10Float"
        case .rgb9e5Float:
            return "rgb9e5Float"
        case .bgr10_xr:
            return "bgr10_xr"
        case .bgr10_xr_srgb:
            return "bgr10_xr_srgb"
        case .rg32Uint:
            return "rg32Uint"
        case .rg32Sint:
            return "rg32Sint"
        case .rg32Float:
            return "rg32Float"
        case .rgba16Unorm:
            return "rgba16Unorm"
        case .rgba16Snorm:
            return "rgba16Snorm"
        case .rgba16Uint:
            return "rgba16Uint"
        case .rgba16Sint:
            return "rgba16Sint"
        case .rgba16Float:
            return "rgba16Float"
        case .BGRA10_XR:
            return "BGRA10_XR"
        case .bgra10_XR_sRGB:
            return "bgra10_XR_sRGB"
        case .rgba32Uint:
            return "rgba32Uint"
        case .rgba32Sint:
            return "rgba32Sint"
        case .rgba32Float:
            return "rgba32Float"
        case .pvrtc_rgb_2bpp:
            return "pvrtc_rgb_2bpp"
        case .pvrtc_rgb_2bpp_srgb:
            return "pvrtc_rgb_2bpp_srgb"
        case .pvrtc_rgb_4bpp:
            return "pvrtc_rgb_4bpp"
        case .pvrtc_rgb_4bpp_srgb:
            return "pvrtc_rgb_4bpp_srgb"
        case .pvrtc_rgba_2bpp:
            return "pvrtc_rgba_2bpp"
        case .pvrtc_rgba_2bpp_srgb:
            return "pvrtc_rgba_2bpp_srgb"
        case .pvrtc_rgba_4bpp:
            return "pvrtc_rgba_4bpp"
        case .pvrtc_rgba_4bpp_srgb:
            return "pvrtc_rgba_4bpp_srgb"
        case .eac_r11Unorm:
            return "eac_r11Unorm"
        case .eac_r11Snorm:
            return "eac_r11Snorm"
        case .eac_rg11Unorm:
            return "eac_rg11Unorm"
        case .eac_rg11Snorm:
            return "eac_rg11Snorm"
        case .eac_rgba8:
            return "eac_rgba8"
        case .eac_rgba8_srgb:
            return "eac_rgba8_srgb"
        case .etc2_rgb8:
            return "etc2_rgb8"
        case .etc2_rgb8_srgb:
            return "etc2_rgb8_srgb"
        case .etc2_rgb8a1:
            return "etc2_rgb8a1"
        case .etc2_rgb8a1_srgb:
            return "etc2_rgb8a1_srgb"
        case .astc_4x4_srgb:
            return "astc_4x4_srgb"
        case .astc_5x4_srgb:
            return "astc_5x4_srgb"
        case .astc_5x5_srgb:
            return "astc_5x5_srgb"
        case .astc_6x5_srgb:
            return "astc_6x5_srgb"
        case .astc_6x6_srgb:
            return "astc_6x6_srgb"
        case .astc_8x5_srgb:
            return "astc_8x5_srgb"
        case .astc_8x6_srgb:
            return "astc_8x6_srgb"
        case .astc_8x8_srgb:
            return "astc_8x8_srgb"
        case .astc_10x5_srgb:
            return "astc_10x5_srgb"
        case .astc_10x6_srgb:
            return "astc_10x6_srgb"
        case .astc_10x8_srgb:
            return "astc_10x8_srgb"
        case .astc_10x10_srgb:
            return "astc_10x10_srgb"
        case .astc_12x10_srgb:
            return "astc_12x10_srgb"
        case .astc_12x12_srgb:
            return "astc_12x12_srgb"
        case .astc_4x4_ldr:
            return "astc_4x4_ldr"
        case .astc_5x4_ldr:
            return "astc_5x4_ldr"
        case .astc_5x5_ldr:
            return "astc_5x5_ldr"
        case .astc_6x5_ldr:
            return "astc_6x5_ldr"
        case .astc_6x6_ldr:
            return "astc_6x6_ldr"
        case .astc_8x5_ldr:
            return "astc_8x5_ldr"
        case .astc_8x6_ldr:
            return "astc_8x6_ldr"
        case .astc_8x8_ldr:
            return "astc_8x8_ldr"
        case .astc_10x5_ldr:
            return "astc_10x5_ldr"
        case .astc_10x6_ldr:
            return "astc_10x6_ldr"
        case .astc_10x8_ldr:
            return "astc_10x8_ldr"
        case .astc_10x10_ldr:
            return "astc_10x10_ldr"
        case .astc_12x10_ldr:
            return "astc_12x10_ldr"
        case .astc_12x12_ldr:
            return "astc_12x12_ldr"
        case .gbgr422:
            return "gbgr422"
        case .bgrg422:
            return "bgrg422"
        case .depth32Float:
            return "depth32Float"
        case .stencil8:
            return "stencil8"
        case .depth32Float_stencil8:
            return "stencil8"
        case .x32_stencil8:
            return "stencil8"
        default:
            return "unknown"
        }
    }
}
