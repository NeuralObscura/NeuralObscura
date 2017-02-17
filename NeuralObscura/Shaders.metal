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


/* Batch Normalization (Test Mode)
 *
 * Formula: output = (gamma * ((input - mean) / stddev)) + beta
 */
kernel void batch_normalization(texture2d_array<float, access::read> inTexture [[texture(0)]],
                                texture2d_array<float, access::write> outTexture [[texture(1)]],
                                const device float* gamma [[ buffer(2) ]],
                                const device float* beta [[ buffer(3) ]],
                                const device float* mean [[ buffer(4) ]],
                                const device float* stddev [[ buffer(5) ]],
                                uint3 gid [[thread_position_in_grid]]) {
    float4 input = inTexture.read(gid.xy, gid.z);
    float4 output = float4(input.r - mean[gid.z],
                           input.g - mean[gid.z+1],
                           input.b - mean[gid.z+2],
                           input.a - mean[gid.z+3]);
    output = float4(output.r / stddev[gid.z],
                    output.g / stddev[gid.z+1],
                    output.b / stddev[gid.z+2],
                    output.a / stddev[gid.z+3]);
    output = float4(output.r * gamma[gid.z],
                    output.g * gamma[gid.z+1],
                    output.b * gamma[gid.z+2],
                    output.a * gamma[gid.z+3]);
    output = float4(output.r + beta[gid.z],
                    output.g + beta[gid.z+1],
                    output.b + beta[gid.z+2],
                    output.a + beta[gid.z+3]);
    outTexture.write(output, gid.xy, gid.z);
}


/* Batch Normalization (Non-test Mode)
 *
 * Formula: output = (gamma * ((input - mean) / stddev)) + beta
 * Thread Group configuration:
 *         let threadsPerGroup = MTLSizeMake(1, 1, channelsIn / sourceImage.texture.pixelFormat.pixelCount)
 *         let threadGroups = MTLSizeMake(1, 1, 1)
 */
kernel void batch_normalization_nt(texture2d_array<float, access::read> inTexture [[texture(0)]],
                                   texture2d_array<float, access::write> outTexture [[texture(1)]],
                                   const device float* gamma [[ buffer(2) ]],
                                   const device float* beta [[ buffer(3) ]],
                                   uint3 gid [[thread_position_in_grid]]) {
    uint height = inTexture.get_height();
    uint width = inTexture.get_width();
    float4 sum = float4(0.0, 0.0, 0.0, 0.0);
    uint slot = (gid.z * 4); // where in gamma & beta to read from

    for(uint j = 0; j < width; j++) {
        for(uint i = 0; i < height; i++) {
            sum += inTexture.read(uint2(j, i), gid.z);
        }
    }

    float4 mean = sum / float4(width*height);
    float4 vari = float4(0.0, 0.0, 0.0, 0.0);
    for(uint j = 0; j < width; j++) {
        for(uint i = 0; i < height; i++) {
            vari += (inTexture.read(uint2(j, i), gid.z) - mean) *
            (inTexture.read(uint2(j, i), gid.z) - mean);
        }
    }
    vari /= (width*height);
    float4 stddev = sqrt(vari) + 0.0000001; // Prevent divide by 0 (chainer constant)

    for(uint j = 0; j < width; j++) {
        for(uint i = 0; i < height; i++) {
            float4 x_hat = (inTexture.read(uint2(j, i), gid.z) - mean) / stddev;
            float4 y = stddev;

            y[0] = (gamma[slot] * x_hat[0]) + beta[slot];
            y[1] = (gamma[slot+1] * x_hat[1]) + beta[slot+1];
            y[2] = (gamma[slot+2] * x_hat[2]) + beta[slot+2];
            y[3] = (gamma[slot+3] * x_hat[3]) + beta[slot+3];

            outTexture.write(y, uint2(j, i), gid.z);
        }
    }
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
    outTexture.write(output, gid.xy, gid.z);
}


/* RGBA -> BRGA
 *
 */
kernel void rgba_to_brga(texture2d<float, access::read> inTexture [[texture(0)]],
                        texture2d<float, access::write> outTexture [[texture(1)]],
                        uint3 gid [[thread_position_in_grid]]) {
    float4 input = inTexture.read(gid.xy, gid.z);
    float4 output = float4(input[2], input[0], input[1], input[3]);
    outTexture.write(output, gid.xy, gid.z);
}

struct _2d_shape {
    uint a;
    uint b;
}

struct _2d_index {
    uint x;
    uint y;
}

struct _3d_shape {
    uint a;
    uint b;
    uint c;
}

struct _3d_index {
    uint x;
    uint y;
    uint z;
}

struct _4d_shape {
    uint a;
    uint b;
    uint c;
    uint d;
}

struct _4d_index {
    uint x;
    uint y;
    uint z;
    uint w;
}

struct _5d_shape {
    uint a;
    uint b;
    uint e;
    uint d;
    uint e;
}

struct _5d_index {
    uint x;
    uint y;
    uint z;
    uint w;
    uint v;
}

uint _4d_index_to_1d_index(_4d_shape shape, weights_index_t index) {
    return  index.x * (shape.b * shape.c * shape.d)
          + index.y * (shape.c * shape.d)
          + index.z * (shape.d)
          + index.w;
}

uint _5d_index_to_1d_index(_5d_shape shape, _5d_index index) {
    return index.x * (shape.b * shape.c * shape.d * shape.e)
         + index.y * (shape.c shape.d * shape.e)
         + index.z * (shape.d * shape.e)
         + index.w * (shape.e)
         + index.v;
}

_2d_index _1d_index_to_2d_index(_2d_shape shape, uint index) {
    return (_2d_index) { index / shape.b, index % shape.b }
}

_3d_index _1d_index_to_3d_index(_3d_shape shape, uint index) {
    uint x = index / (shape.b * shape.c);
    uint y = (index % (shape.b * shape.c)) / (shape.c);
    uint z = index % shape.c;
    return (_3d_index) { x, y, z };
}


/* Tensor dot of the feature map and the weights array over the c_in axis 
 *
 * input featureMap has dimensions (c_in, h, w)
 * input weights has size c_out * kh * kw * c_in
 * gid[0] takes values from 0..(c_out * kh * kw)
 * gid[1] takes values from 0..(h * w)
 *
 * output buffer has size c_out * kh * kw * h * w
 */
kernel void deconvolution_v2_tensordot(texture2d_array<float, access::read> featureMap [[texture(0)]],
                                       float* output [[buffer(1) ]],
                                       const device float* weights [[ buffer(2) ]],
                                       const device uint* weightsShapeParam [[ buffer(3) ]],
                                       uint2 gid [[thread_position_in_grid]]) {
    uint nc_out = *weightsShapeParam;
    uint nkh = *(weightsShapeParam + 1);
    uint nkw = *(weightsShapeParam + 2);
    uint nc_in = *(kernelShape + 3);
    _4d_shape weightsShape = { nc_out, nkh, nkw, nc_in };
    
    uint nh = featureMap.get_height();
    uint nw = featureMap.get_width();
    
    /* gid[1] is a 1d index into a 3d array with shape (nc_out, nkh, nkw) */
    _3d_shape rightKernelShape = { nc_out, nkh, nkw };
    _3d_index rightKernelIndex = _1d_index_to_3d_index(kernelShape, gid[1])
    
    /* rightKernelIndex has the form { x: c_out, y: kh, z: kw } */
    uint c_out = rightKernelIndex.x;
    uint kh = rightKernelIndex.y;
    uint kw = rightKernelIndex.z;
    
    /* gid[0] is a 1d index into a 2d array with shape (nh, nw) */
    _2d_shape locShape = { nh, nw };
    _2d_index locIndex = _1d_index_to_2d_index(locShape, gid[0]);
    
    /* locIndex has the form { x: h, y: w } */
    uint h = locIndex.x;
    uint w = locIndex.y;
    
    _5d_shape outputShape = { nc_out, nkh, nkw, nh, nw };
    _5d_index outputIndex = { c_out, kh, kw, h, w };
    
    float acc = 0;
    for (uint c_in = 0; c_in < nc_in; c_in++) {
        _4d_index weightsIndex = { c_out, kh, kw, c_in };
        float weightValue = weights[_4d_index_to_1d_index(weightsShape, weightsIndex)];
        float featureMapValue = featureMap.read(uint2(w, h), c_in / 4)[c_in % 4];
        acc += weightValue * featureMapValue;
    }
    output[_5d_index_to_1d_index(outputShape, outputIndex)] = acc;
}
