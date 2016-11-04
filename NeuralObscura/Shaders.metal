#include <metal_stdlib>

using namespace metal;

kernel void identity(texture2d<float, access::read> inTexture [[texture(0)]],
                     texture2d<float, access::write> outTexture [[texture(1)]],
                     uint2 gid [[thread_position_in_grid]]) {
    float4 inColor = inTexture.read(gid);
    float4 outColor = inColor;
    outTexture.write(outColor, gid);
}

/* Batch Normalization
 *
 * Formula: output = (gamma * input) + beta
 *
 * Thread grouping must be that each thread is a single feature channel.
 * i.e. gid.z == feature channel to operate on.
 * 
 * Expects 4 channels per pixel (RGBA)
 */
kernel void batch_normalization(texture2d_array<float, access::read> inTexture [[texture(0)]],
                                texture2d_array<float, access::write> outTexture [[texture(1)]],
                                const device float* gamma [[ buffer(2) ]],
                                const device float* beta [[ buffer(3) ]],
                                uint3 gid [[thread_position_in_grid]]) {
    float4 inColor = inTexture.read(gid.xy, gid.z);
    float4 outColor = inColor;

    outTexture.write(outColor, gid.xy, gid.z);
}
