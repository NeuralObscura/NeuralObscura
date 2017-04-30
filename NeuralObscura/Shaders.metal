#include <metal_stdlib>
#include <metal_texture>

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
kernel void batch_normalization(texture2d_array<half, access::read> inTexture [[texture(0)]],
                                texture2d_array<half, access::write> outTexture [[texture(1)]],
                                const device float* gamma [[ buffer(2) ]],
                                const device float* beta [[ buffer(3) ]],
                                const device float* mean [[ buffer(4) ]],
                                const device float* stddev [[ buffer(5) ]],
                                uint3 gid [[thread_position_in_grid]]) {
    uint buffer_idx = gid.z * 4;
    half4 input = inTexture.read(gid.xy, gid.z);
    float no_zero_divide = 0.0000001;
    half4 output = half4(input.r - mean[buffer_idx],
                         input.g - mean[buffer_idx+1],
                         input.b - mean[buffer_idx+2],
                         input.a - mean[buffer_idx+3]);
    output = half4(output.r / (stddev[buffer_idx] + no_zero_divide),
                   output.g / (stddev[buffer_idx+1] + no_zero_divide),
                   output.b / (stddev[buffer_idx+2] + no_zero_divide),
                   output.a / (stddev[buffer_idx+3] + no_zero_divide));
    output = half4(output.r * gamma[buffer_idx],
                   output.g * gamma[buffer_idx+1],
                   output.b * gamma[buffer_idx+2],
                   output.a * gamma[buffer_idx+3]);
    output = half4(output.r + beta[buffer_idx],
                   output.g + beta[buffer_idx+1],
                   output.b + beta[buffer_idx+2],
                   output.a + beta[buffer_idx+3]);
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
    uint slot = (gid.z * 4); // where in gamma & beta to read from

    
    float4 sum = float4(0.0, 0.0, 0.0, 0.0);
    float4 error = float4(0.0, 0.0, 0.0, 0.0);
    for(uint j = 0; j < width; j++) {
        for(uint i = 0; i < height; i++) {
            float4 y = inTexture.read(uint2(j, i), gid.z);
            float4 t = sum + y;
            error = (t - sum) - y;
            sum = t;
        }
    }

    float4 mean = sum / float4(width*height);
    float4 vari = float4(0.0, 0.0, 0.0, 0.0);
    error = float4(0.0, 0.0, 0.0, 0.0);
    for(uint j = 0; j < width; j++) {
        for(uint i = 0; i < height; i++) {
            float4 y = (inTexture.read(uint2(j, i), gid.z) - mean) * (inTexture.read(uint2(j, i), gid.z) - mean);
            float4 t = vari + y;
            error = (t - vari) - y;
            vari = t;
        }
    }
    vari /= (width*height);
    float4 stddev = sqrt(vari) + 0.0000001; // Prevent divide by 0 (chainer constant)

    /* TODO: Implement numerical error correction here. */
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

/* Add two matrices together
 *
 * Formula: output = input1 + input2
 */
kernel void add(texture2d_array<half, access::read> inTexture1 [[texture(0)]],
                texture2d_array<half, access::read> inTexture2 [[texture(1)]],
                texture2d_array<half, access::write> outTexture [[texture(2)]],
                uint3 gid [[thread_position_in_grid]]) {
    half4 input1 = inTexture1.read(gid.xy, gid.z);
    half4 input2 = inTexture2.read(gid.xy, gid.z);
    half4 output = input1 + input2;
    outTexture.write(output, gid.xy, gid.z);
}


/* Tanh cleanup and adjustment. To be used at the end of style processing
 *
 * Formula: output = (tanh(input)+1)*127.5
 */
kernel void tanh_adjustment(texture2d_array<half, access::read> inTexture [[texture(0)]],
                            texture2d_array<half, access::write> outTexture [[texture(1)]],
                            uint3 gid [[thread_position_in_grid]]) {
    half4 input = inTexture.read(gid.xy, gid.z);
    half4 output = (tanh(input) + 1) * 127.5;
    outTexture.write(output, gid.xy, gid.z);
}


/* RGBA -> BRGA
 *
 */
kernel void rgba_to_brga(texture2d_array<half, access::read> inTexture [[texture(0)]],
                         texture2d_array<half, access::write> outTexture [[texture(1)]],
                         uint3 gid [[thread_position_in_grid]]) {
    half4 input = inTexture.read(gid.xy, gid.z);
    half4 output = half4(input[2], input[0], input[1], input[3]);
    outTexture.write(output, gid.xy, gid.z);
}

struct _2d_shape {
    uint a;
    uint b;
};

struct _2d_index {
    uint x;
    uint y;
};

struct _3d_shape {
    uint a;
    uint b;
    uint c;
};

struct _3d_index {
    uint x;
    uint y;
    uint z;
};

struct _4d_shape {
    uint a;
    uint b;
    uint c;
    uint d;
};

struct _4d_index {
    uint x;
    uint y;
    uint z;
    uint w;
};

struct _5d_shape {
    uint a;
    uint b;
    uint c;
    uint d;
    uint e;
};

struct _5d_index {
    uint x;
    uint y;
    uint z;
    uint w;
    uint v;
};

uint _2d_index_to_1d_index(uint shape, _2d_index index);
uint _2d_index_to_1d_index(uint shape, _2d_index index) {
    return shape * index.y
         + index.x;
}

uint _4d_index_to_1d_index(_4d_shape shape, _4d_index index);
uint _4d_index_to_1d_index(_4d_shape shape, _4d_index index) {
    return index.x * (shape.b * shape.c * shape.d)
         + index.y * (shape.c * shape.d)
         + index.z * (shape.d)
         + index.w;
}

uint _5d_index_to_1d_index(_5d_shape shape, _5d_index index);
uint _5d_index_to_1d_index(_5d_shape shape, _5d_index index) {
    return index.x * (shape.b * shape.c * shape.d * shape.e)
         + index.y * (shape.c * shape.d * shape.e)
         + index.z * (shape.d * shape.e)
         + index.w * (shape.e)
         + index.v;
}

_2d_index _1d_index_to_2d_index(_2d_shape shape, uint index);
_2d_index _1d_index_to_2d_index(_2d_shape shape, uint index) {
    return (_2d_index) { index / shape.b, index % shape.b };
}

_3d_index _1d_index_to_3d_index(_3d_shape shape, uint index);
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
                                       device float* output [[buffer(1)]],
                                       const device float* weights [[ buffer(2) ]],
                                       const device uint* weightsShapeParam [[ buffer(3) ]],
                                       uint2 position [[thread_position_in_grid]]) {
    uint nc_out = weightsShapeParam[0];
    uint nkh = weightsShapeParam[1];
    uint nkw = weightsShapeParam[2];
    uint nc_in = weightsShapeParam[3];
    uint nh = featureMap.get_height();
    uint nw = featureMap.get_width();
    
    /* Early return if thread position is outside output range. */
    if (position.x > featureMap.get_height() * featureMap.get_width() - 1 || position.y > nc_out * nkh * nkw - 1) {
        return;
    }
    
    _4d_shape weightsShape = { nc_in, nkh, nkw, nc_out };
    uint nslices = featureMap.get_array_size();
    
    /* position[0] is a 1d index into a 2d array with shape (nh, nw) */
    _2d_shape locShape = { nh, nw };
    _2d_index locIndex = _1d_index_to_2d_index(locShape, position[0]);
    /* locIndex has the form { x: h, y: w } */
    uint h = locIndex.x;
    uint w = locIndex.y;

    /* position[1] is a 1d index into a 3d array with shape (nc_out, nkh, nkw) */
    _3d_shape rightKernelShape = { nc_out, nkh, nkw };
    _3d_index rightKernelIndex = _1d_index_to_3d_index(rightKernelShape, position[1]);
    /* rightKernelIndex has the form { x: c_out, y: kh, z: kw } */
    uint c_out = rightKernelIndex.x;
    uint kh = rightKernelIndex.y;
    uint kw = rightKernelIndex.z;
    
    _5d_shape outputShape = { nc_out, nkh, nkw, nh, nw };
    _5d_index outputIndex = { c_out, kh, kw, h, w };
    
    float sum = 0;
    float error = 0;
    for (uint slice = 0; slice < nslices; ++slice) {
        float4 weightValues;
        uint c_in_base = slice * 4;
        for (uint c_in_offset = 0; c_in_offset < 4; c_in_offset++) {
            _4d_index weightsIndex = { c_in_base + c_in_offset, kh, kw, c_out };
            uint index = _4d_index_to_1d_index(weightsShape, weightsIndex);
            weightValues[c_in_offset] = weights[index];
        }
        float4 featureMapValues = featureMap.read(uint2(w, h), slice);
        float termGroup = dot(weightValues, featureMapValues);
        /* https://en.wikipedia.org/wiki/Kahan_summation_algorithm */
        float y = termGroup - error;
        float t = sum + y;
        error = (t - sum) - y;
        sum = t;
    }
    output[_5d_index_to_1d_index(outputShape, outputIndex)] = sum;
}

kernel void col2im(const device float* input [[ buffer (0) ]],
                   texture2d_array<float, access::write> outTexture [[texture(1)]],
                   const device uint* input_dim [[buffer(2)]],
                   uint3 id [[thread_position_in_grid]]) {

    //uint nc_out = input_dim[0];
    uint nh = input_dim[1];
    uint nw = input_dim[2];
    //uint inputSize = input_dim[3];
    uint nkh = input_dim[4];
    uint nkw = input_dim[5];
    uint destRowWidth = input_dim[6];
    uint s = input_dim[7];
    uint p = input_dim[8];

    int i = id.x;
    int sy = s;
    int sx = s;
    int ph = p;
    int pw = p;
    int kh = nkh;
    int kw = nkw;
    int out_h = nh;
    int out_w = nw;
    int h = nh;
    int w = nw;


    int c0 = i / (h * w);
    int y  = i / w % h;
    int x  = i % w;


    float val = 0.0;
    for (int ky = 0; ky < kh; ++ky) {
        int out_y = (y + ph - ky * 1);
        if (0 > out_y || out_y >= out_h * sy) continue;
        if (out_y % sy != 0) continue;
        out_y /= sy;
        for (int kx = 0; kx < kw; ++kx) {
            int out_x = (x + pw - kx * 1);
            if (0 > out_x || out_x >= out_w * sx) continue;
            if (out_x % sx != 0) continue;
            out_x /= sx;
            int k = out_y + out_h * (kx + kw * (ky + kh * c0));
            val = val + input[out_x + out_w * k];
        }
    }
    uint dest_y = i / destRowWidth;
    uint dest_x = i % destRowWidth;
    uint2 idx = {dest_x, dest_y};
    outTexture.write(val, idx, c0);
}
