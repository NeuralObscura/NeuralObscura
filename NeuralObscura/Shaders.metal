#include <metal_stdlib>

using namespace metal;

kernel void identity(texture2d<float, access::read> inTexture [[texture(0)]],
                     texture2d<float, access::write> outTexture [[texture(1)]],
                     uint2 gid [[thread_position_in_grid]]) {
    float4 inColor = inTexture.read(gid);
    float4 outColor = inColor;
    outTexture.write(outColor, gid);
}

/* ReLU (Rectifier Activation)
 *
 * Formula: output[i] = max(0, input[i])
 */
kernel void rectifier_linear(texture2d_array<float, access::read> inTexture [[texture(0)]],
                             texture2d_array<float, access::write> outTexture [[texture(1)]],
                             uint3 gid [[thread_position_in_grid]]) {
    float4 input = inTexture.read(gid.xy, gid.z);
    float4 output = float4(fmax(0.0, input.r), fmax(0.0, input.g), fmax(0.0, input.b), fmax(0.0, input.a));
    outTexture.write(output, gid.xy, gid.z);
}


/* Batch Normalization
 *
 * Formula: output = (gamma * input) + beta
 */
kernel void batch_normalization(texture2d_array<float, access::read> inTexture [[texture(0)]],
                                texture2d_array<float, access::write> outTexture [[texture(1)]],
                                const device float* gamma [[ buffer(2) ]],
                                const device float* beta [[ buffer(3) ]],
                                uint3 gid [[thread_position_in_grid]]) {
    float4 input = inTexture.read(gid.xy, gid.z);
    float4 output = float4(input.r * gamma[gid.z], input.g * gamma[gid.z+1], input.b * gamma[gid.z+2], input.a * gamma[gid.z+3]);
    output = float4(output.r + beta[gid.z], output.g + beta[gid.z+1], output.b + beta[gid.z+2], output.a + beta[gid.z+3]);
    outTexture.write(output, gid.xy, gid.z);
}

/* Deconvolution Interpixel Stride
 *
 * Formula: output[i * stride][j * stride] = input[i][j]
 */
kernel void deconvolution_interpixel_stride(texture2d_array<float, access::read> inTexture [[texture(0)]],
                                texture2d_array<float, access::write> outTexture [[texture(1)]],
                                const device uint* stride [[ buffer(2) ]],
                                uint3 gid [[thread_position_in_grid]]) {
    float4 outColor = inTexture.read(gid.xy, gid.z);
    uint2 outLoc = uint2(gid.x * *stride, gid.y * *stride);
    outTexture.write(outColor, outLoc, gid.z);
}

/* Add two matrices together
 *
 * Formula: output = input1 + input2
 */
kernel void add(texture2d_array<float, access::read> inTexture1 [[texture(0)]],
                texture2d_array<float, access::read> inTexture2 [[texture(1)]],
                texture2d_array<float, access::write> outTexture [[texture(2)]],
                uint3 gid [[thread_position_in_grid]]) {
    float4 input1 = inTexture1.read(gid.xy, gid.z);
    float4 input2 = inTexture2.read(gid.xy, gid.z);
    float4 output = input1 + input2;
    outTexture.write(output, gid.xy, gid.z);
}

/* Tanh cleanup and adjustment. To be used at the end of style processing
 *
 * Formula: output = (tanh(input)+1)*127.5
 */
kernel void tanh_adjustment(texture2d<float, access::read> inTexture [[texture(0)]],
                            texture2d<float, access::write> outTexture [[texture(1)]],
                            uint3 gid [[thread_position_in_grid]]) {
    float4 input = inTexture.read(gid.xy, gid.z);
    float4 output = (tanh(input) + 1) * 127.5;
    output.a = 255.0; // max opacity
    outTexture.write(output, gid.xy, gid.z);
}
