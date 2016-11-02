#include <metal_stdlib>

using namespace metal;

kernel void identity(texture2d<float, access::read> inTexture [[texture(0)]],
                     texture2d<float, access::write> outTexture [[texture(1)]],
                     uint2 gid [[thread_position_in_grid]]) {
    float4 inColor = inTexture.read(gid);
    float4 outColor = inColor;
    outTexture.write(outColor, gid);
}

kernel void addOne(texture2d_array<float, access::read> inTexture [[texture(0)]],
                   texture2d_array<float, access::write> outTexture [[texture(1)]],
                   uint3 gid [[thread_position_in_grid]]) {
    float4 inColor = inTexture.read(gid.xy, gid.z);
    float4 outColor = inColor;

    outTexture.write(outColor, gid.xy, gid.z);
}

