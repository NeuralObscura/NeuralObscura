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
 */
kernel void batch_normalization_nt(texture2d_array<float, access::read> inTexture [[texture(0)]],
                                   texture2d_array<float, access::write> outTexture [[texture(1)]],
                                   const device float* gammas [[ buffer(2) ]],
                                   const device float* betas [[ buffer(3) ]],
                                   const device float* means [[ buffer(4) ]],
                                   const device float* stddevs [[ buffer(5) ]],
                                   uint3 gid [[thread_position_in_grid]]) {
    uint height = inTexture.get_height();
    uint width = inTexture.get_width();
    uint slot = (gid.z * 4);

    float4 working_means = float4(means[slot], means[slot+1], means[slot+2], means[slot+3]);
    float4 working_stddevs = float4(stddevs[slot], stddevs[slot+1], stddevs[slot+2], stddevs[slot+3]);
    float4 working_betas = float4(betas[slot], betas[slot+1], betas[slot+2], betas[slot+3]);
    float4 working_gammas = float4(gammas[slot], gammas[slot+1], gammas[slot+2], gammas[slot+3]);
    
    /* TODO: Implement numerical error correction here. */
    for(uint j = 0; j < width; ++j) {
        for(uint i = 0; i < height; ++i) {
            float4 x_hat = (inTexture.read(uint2(j, i), gid.z) - working_means) / working_stddevs;
            float4 y = working_gammas * x_hat + working_betas;
            outTexture.write(y, uint2(j, i), gid.z);
        }
    }
}

/* Produces two buffers one containing channel-wide mean, the other with channel-wide stddev. */
/* TODO: This runs on a single GPU core per slice... not optimal. */
kernel void mean_std_dev(texture2d_array<float, access::read> inTexture [[texture(0)]],
                         device packed_float4* means [[buffer(1)]],
                         device packed_float4* stddevs [[buffer(2)]],
                         uint3 gid [[thread_position_in_grid]]) {
    uint height = inTexture.get_height();
    uint width = inTexture.get_width();
    int count = width * height;
    
    float4 sum = float4(0,0,0,0);
    float4 error = float4(0,0,0,0);
    for(uint j = 0; j < width; ++j) {
        for(uint i = 0; i < height; ++i) {
            float4 y = inTexture.read(uint2(j, i), gid.z) - error;
            float4 t = sum + y;
            error = (t - sum) - y;
            sum = t;
        }
    }
    float4 mean = sum / count;
    
    float4 vari = float4(0.0, 0.0, 0.0, 0.0);
    error = float4(0.0, 0.0, 0.0, 0.0);
    for(uint j = 0; j < width; ++j) {
        for(uint i = 0; i < height; ++i) {
            float4 y = pow(inTexture.read(uint2(j, i), gid.z) - mean, 2) - error;
            float4 t = vari + y;
            error = (t - vari) - y;
            vari = t;
        }
    }
    vari /= (count - 1);
    float4 stddev = sqrt(vari);
    if (stddev[0] == 0) { stddev[0] = 1; }
    if (stddev[1] == 0) { stddev[1] = 1; }
    if (stddev[2] == 0) { stddev[2] = 1; }
    if (stddev[3] == 0) { stddev[3] = 1; }

    means[gid.z] = packed_float4(mean);
    stddevs[gid.z] = packed_float4(stddev);
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

/* BGRA -> BRGA
 *
 */
kernel void bgra_to_brga(texture2d<half, access::read> inTexture [[texture(0)]],
                         texture2d<half, access::write> outTexture [[texture(1)]],
                         uint3 pos [[thread_position_in_grid]]) {
    
    if (pos.x > outTexture.get_width() - 1 || pos.y > outTexture.get_height() - 1) {
        return;
    }

    half4 input = inTexture.read(pos.xy, pos.z);
    half4 output = half4(input[0], input[2], input[1], input[3]);
    outTexture.write(output, pos.xy, pos.z);
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
kernel void tensordot(texture2d_array<half, access::read> featureMap [[texture(0)]],
                      device half* output [[buffer(1)]],
                      const device half* weights [[ buffer(2) ]],
                      const device uint* weightsShapeParam [[ buffer(3) ]],
                      uint2 pos [[thread_position_in_grid]]) {
    /* TODO: Structify these parameters */
    uint nc_out = weightsShapeParam[0];
    uint nkh = weightsShapeParam[1];
    uint nkw = weightsShapeParam[2];
    uint nc_in = weightsShapeParam[3];
    uint nh = featureMap.get_height();
    uint nw = featureMap.get_width();
    
    /* Early return if thread pos is outside output range. */
    if (pos.x > featureMap.get_height() * featureMap.get_width() - 1 || pos.y > nc_out * nkh * nkw - 1) {
        return;
    }
    
    _4d_shape weightsShape = { nc_out, nkh, nkw, nc_in };
    uint nslices = featureMap.get_array_size();
    
    /* pos[0] is a 1d index into a 2d array with shape (nh, nw) */
    _2d_shape locShape = { nh, nw };
    _2d_index locIndex = _1d_index_to_2d_index(locShape, pos[0]);
    /* locIndex has the form { x: h, y: w } */
    uint h = locIndex.x;
    uint w = locIndex.y;

    /* pos[1] is a 1d index into a 3d array with shape (nc_out, nkh, nkw) */
    _3d_shape rightKernelShape = { nc_out, nkh, nkw };
    _3d_index rightKernelIndex = _1d_index_to_3d_index(rightKernelShape, pos[1]);
    /* rightKernelIndex has the form { x: c_out, y: kh, z: kw } */
    uint c_out = rightKernelIndex.x;
    uint kh = rightKernelIndex.y;
    uint kw = rightKernelIndex.z;
    
    _5d_shape outputShape = { nc_out, nkh, nkw, nh, nw };
    _5d_index outputIndex = { c_out, kh, kw, h, w };
    
    half sum = 0;
    half error = 0;
    for (uint slice = 0; slice < nslices; ++slice) {
        half4 weightValues;
        uint c_in_base = slice * 4;
        for (uint c_in_offset = 0; c_in_offset < 4; ++c_in_offset) {
            _4d_index weightsIndex = { c_out, kh, kw, c_in_base + c_in_offset };
            uint index = _4d_index_to_1d_index(weightsShape, weightsIndex);
            weightValues[c_in_offset] = weights[index];
        }
        half4 featureMapValues = featureMap.read(uint2(w, h), slice);
        half termGroup = dot(weightValues, featureMapValues);
        /* https://en.wikipedia.org/wiki/Kahan_summation_algorithm */
        half y = termGroup - error;
        half t = sum + y;
        error = (t - sum) - y;
        sum = t;
    }
    output[_5d_index_to_1d_index(outputShape, outputIndex)] = sum;
}

kernel void col2im(const device half* input [[ buffer (0) ]],
                   texture2d_array<half, access::write> outTexture [[texture(1)]],
                   const device uint* input_dim [[buffer(2)]],
                   uint3 gid [[thread_position_in_grid]]) {

    if (gid.x > outTexture.get_width() - 1 || gid.y > outTexture.get_height() - 1) {
        return;
    }
    
    uint nh = input_dim[0];
    uint nw = input_dim[1];
    uint nc_out = input_dim[2];
    uint k = input_dim[3];
    uint s = input_dim[4];
    uint p = input_dim[5];
    
    _5d_shape inputShape = { nc_out, k, k, nh, nw };
    half4 vals = half4(0, 0, 0, 0);
    half4 errors = half4(0, 0, 0, 0);
    for (uint ky = 0; ky < k; ++ky) {
        /* This can/will result in underflow, but we keep it because it reduces
         conversions needed, and we catch underflow in the next condition. */
        uint y = (gid.y + p - ky); 
        if (y >= nh * s) continue;
        if (y % s != 0) continue;
        y /= s;
        for (uint kx = 0; kx < k; ++kx) {
            /* This can/will result in underflow, but we keep it because it reduces
             conversions needed, and we catch underflow in the next condition. */
            uint x = (gid.x + p - kx);
            if (x >= nw * s) continue;
            if (x % s != 0) continue;
            x /= s;

            for (uint c_offset = 0; c_offset < 4; ++c_offset) {
                _5d_index inputIndex = { gid.z * 4 + c_offset, ky, kx, y, x };
                half term = input[_5d_index_to_1d_index(inputShape, inputIndex)];
                half y = term - errors[c_offset];
                half t = vals[c_offset] + y;
                errors[c_offset] = (t - vals[c_offset]) - y;
                vals[c_offset] = t;
            }
        }
    }
    outTexture.write(vals, gid.xy, gid.z);
}

kernel void add_bias(texture2d_array<half, access::read> inTexture [[texture(0)]],
                     texture2d_array<half, access::write> outTexture [[texture(1)]],
                     const device half* biases [[buffer(2)]],
                     uint3 gid [[thread_position_in_grid]]) {
    
    if (gid.x > outTexture.get_width() - 1 || gid.y > outTexture.get_height() - 1) {
        return;
    }
    
    uint c_base = gid.z * 4;
    half4 workingVals = inTexture.read(gid.xy, gid.z);
    half4 workingBiases = half4(biases[c_base], biases[c_base + 1], biases[c_base + 2], biases[c_base + 3]);
    outTexture.write(workingVals + workingBiases, gid.xy, gid.z);
}

kernel void deconvolution_interpixel_stride(texture2d_array<half, access::read> inTexture [[texture(0)]],
                                            texture2d_array<half, access::write> outTexture [[texture(1)]],
                                            const device uint* stride [[ buffer(2) ]],
                                            uint3 pos [[thread_position_in_grid]]) {
    half4 outColor = inTexture.read(pos.xy, pos.z);
    uint2 outLoc = uint2(pos.x * *stride, pos.y * *stride);
    outTexture.write(outColor, outLoc, pos.z);
}
