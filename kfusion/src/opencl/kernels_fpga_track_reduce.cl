/*

 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#define NUM_WI_1 200
#define NUM_WI_2 200
#define NUM_WI_3 200

#define BATCHSIZE_1 (320*240)/NUM_WI_1
#define BATCHSIZE_2 (160*120)/NUM_WI_2
#define BATCHSIZE_3 (80*60)/NUM_WI_3

/************** TYPES ***************/

#define INVALID -2

typedef struct sTrackData {
    int result;
    float error;
    float J[6];
} TrackData;

typedef struct sMatrix4 {
    float4 data[4];
} Matrix4;

/************** FUNCTIONS ***************/

inline float3 Mat4TimeFloat3(const float4 M0, const float4 M1, const float4 M2, float3 v) {
    return (float3)(
            dot((float3)(M0.x, M0.y, M0.z), v)
                    + M0.w,
            dot((float3)(M1.x, M1.y, M1.z), v)
                    + M1.w,
            dot((float3)(M2.x, M2.y, M2.z), v)
                    + M2.w);
}

inline float3 myrotate(const float4 M0, const float4 M1, const float4 M2, const float3 v) {
    return (float3)(dot((float3)(M0.x, M0.y, M0.z), v),
            dot((float3)(M1.x, M1.y, M1.z), v),
            dot((float3)(M2.x, M2.y, M2.z), v));
}

__attribute__((max_work_group_size(320*240)))
__kernel void trackKernel (
        __global __write_only TrackData * restrict output,
        const uint2 outputSize,
        __global __read_only const float * restrict inVertex,// float3
        const uint2 inVertexSize,
        __global __read_only const float * restrict inNormal,// float3
        const uint2 inNormalSize,
        __global __read_only const float * restrict refVertex,// float3
        const uint2 refVertexSize,
        __global __read_only const float * restrict refNormal,// float3
        const uint2 refNormalSize,
        const float4 Ttrack0, const float4 Ttrack1, const float4 Ttrack2, const float4 Ttrack3,
        const float4 view0, const float4 view1, const float4 view2, const float4 view3,
        const float dist_threshold,
        const float normal_threshold
) {
    const size_t pixelX = get_global_id(0);
    const size_t pixelY = get_global_id(1);

    if(pixelX >= inVertexSize.x || pixelY >= inVertexSize.y ) {return;}

    float3 inNormalPixel = vload3(pixelX + inNormalSize.x * pixelY,inNormal);

    if(inNormalPixel.x == INVALID ) {
        output[pixelX + outputSize.x * pixelY].result = -1;
        return;
    }

    float3 inVertexPixel = vload3(pixelX + inVertexSize.x * pixelY,inVertex);
    const float3 projectedVertex = Mat4TimeFloat3 (Ttrack0, Ttrack1, Ttrack2, inVertexPixel);
    const float3 projectedPos = Mat4TimeFloat3 (view0, view1, view2, projectedVertex);
    const float2 projPixel = (float2) ( projectedPos.x / projectedPos.z + 0.5f, projectedPos.y / projectedPos.z + 0.5f);

    if(projPixel.x < 0 || projPixel.x > refVertexSize.x-1 || projPixel.y < 0 || projPixel.y > refVertexSize.y-1 ) {
        output[pixelX + outputSize.x * pixelY].result = -2;
        return;
    }

    const uint2 refPixel = (uint2) (projPixel.x, projPixel.y);
    const float3 referenceNormal = vload3(refPixel.x + refNormalSize.x * refPixel.y,refNormal);

    if(referenceNormal.x == INVALID) {
        output[pixelX + outputSize.x * pixelY].result = -3;
        return;
    }

    const float3 diff = vload3(refPixel.x + refVertexSize.x * refPixel.y,refVertex) - projectedVertex;
    const float3 projectedNormal = myrotate(Ttrack0, Ttrack1, Ttrack2, inNormalPixel);

    if(length(diff) > dist_threshold ) {
        output[pixelX + outputSize.x * pixelY].result = -4;
        return;
    }
    if(dot(projectedNormal, referenceNormal) < normal_threshold) {
        output[pixelX + outputSize.x * pixelY] .result = -5;
        return;
    }

    output[pixelX + outputSize.x * pixelY].result = 1;
    output[pixelX + outputSize.x * pixelY].error = dot(referenceNormal, diff);

    vstore3(referenceNormal,0,(output[pixelX + outputSize.x * pixelY].J));
    vstore3(cross(projectedVertex, referenceNormal),1,(output[pixelX + outputSize.x * pixelY].J));

}

__attribute__((reqd_work_group_size(NUM_WI_1,1,1)))
__kernel void reduceKernel1 (
        __global __write_only float * restrict out,
        __global __read_only const TrackData * restrict J
) {
    uint threadIdx = get_global_id(0);
    uint globalSize = get_global_size(0);

    float sums[32];
    uint i, k;

    #pragma unroll
    for(i = 0; i < 32; ++i) {
        sums[i] = 0.0f;
    }

    for(i = 0; i < BATCHSIZE_1; i++) {
        const TrackData row = J[threadIdx + i*globalSize];
        if(row.result < 1) {
            if (row.result == -4) {
                sums[29] += 1;
            } else if (row.result == -5) {
                sums[30] += 1;
            } else if (row.result > -4) {
                sums[31] += 1;
            }
        } else {
            // Error part
            //sums[0] += row.error * row.error;
            sums[0] = mad(row.error, row.error, sums[0]);

            // JTe part
            #pragma unroll
            for(k = 0; k < 6; ++k) {
                sums[k+1] = mad(row.error, row.J[k], sums[k+1]);
            }

            sums[7] = mad(row.J[0], row.J[0], sums[7]);
            sums[8] = mad(row.J[0], row.J[1], sums[8]);
            sums[9] = mad(row.J[0], row.J[2], sums[9]);
            sums[10] = mad(row.J[0], row.J[3], sums[10]);
            sums[11] = mad(row.J[0], row.J[4], sums[11]);
            sums[12] = mad(row.J[0], row.J[5], sums[12]);

            sums[13] = mad(row.J[1], row.J[1], sums[13]);
            sums[14] = mad(row.J[1], row.J[2], sums[14]);
            sums[15] = mad(row.J[1], row.J[3], sums[15]);
            sums[16] = mad(row.J[1], row.J[4], sums[16]);
            sums[17] = mad(row.J[1], row.J[5], sums[17]);

            sums[18] = mad(row.J[2], row.J[2], sums[18]);
            sums[19] = mad(row.J[2], row.J[3], sums[19]);
            sums[20] = mad(row.J[2], row.J[4], sums[20]);
            sums[21] = mad(row.J[2], row.J[5], sums[21]);

            sums[22] = mad(row.J[3], row.J[3], sums[22]);
            sums[23] = mad(row.J[3], row.J[4], sums[23]);
            sums[24] = mad(row.J[3], row.J[5], sums[24]);

            sums[25] = mad(row.J[4], row.J[4], sums[25]);
            sums[26] = mad(row.J[4], row.J[5], sums[26]);

            sums[27] = mad(row.J[5], row.J[5], sums[27]);
            
            // extra info here
            sums[28] += 1;
        }
    }

    #pragma unroll
    for(i = 0; i < 32; i++) {
        out[i+threadIdx*32] = sums[i];
    }
}

__attribute__((reqd_work_group_size(NUM_WI_2,1,1)))
__kernel void reduceKernel2 (
        __global __write_only float * restrict out,
        __global __read_only const TrackData * restrict J
) {
    uint threadIdx = get_global_id(0);
    uint globalSize = get_global_size(0);

    float sums[32];
    uint i, k;

    #pragma unroll
    for(i = 0; i < 32; ++i) {
        sums[i] = 0.0f;
    }

    for(i = 0; i < BATCHSIZE_2; i++) {
        const TrackData row = J[threadIdx + i*globalSize];
        if(row.result < 1) {
            if (row.result == -4) {
                sums[29] += 1;
            } else if (row.result == -5) {
                sums[30] += 1;
            } else if (row.result > -4) {
                sums[31] += 1;
            }
        } else {
            // Error part
            //sums[0] += row.error * row.error;
            sums[0] = mad(row.error, row.error, sums[0]);

            // JTe part
            #pragma unroll
            for(k = 0; k < 6; ++k) {
                sums[k+1] = mad(row.error, row.J[k], sums[k+1]);
            }

            sums[7] = mad(row.J[0], row.J[0], sums[7]);
            sums[8] = mad(row.J[0], row.J[1], sums[8]);
            sums[9] = mad(row.J[0], row.J[2], sums[9]);
            sums[10] = mad(row.J[0], row.J[3], sums[10]);
            sums[11] = mad(row.J[0], row.J[4], sums[11]);
            sums[12] = mad(row.J[0], row.J[5], sums[12]);

            sums[13] = mad(row.J[1], row.J[1], sums[13]);
            sums[14] = mad(row.J[1], row.J[2], sums[14]);
            sums[15] = mad(row.J[1], row.J[3], sums[15]);
            sums[16] = mad(row.J[1], row.J[4], sums[16]);
            sums[17] = mad(row.J[1], row.J[5], sums[17]);

            sums[18] = mad(row.J[2], row.J[2], sums[18]);
            sums[19] = mad(row.J[2], row.J[3], sums[19]);
            sums[20] = mad(row.J[2], row.J[4], sums[20]);
            sums[21] = mad(row.J[2], row.J[5], sums[21]);

            sums[22] = mad(row.J[3], row.J[3], sums[22]);
            sums[23] = mad(row.J[3], row.J[4], sums[23]);
            sums[24] = mad(row.J[3], row.J[5], sums[24]);

            sums[25] = mad(row.J[4], row.J[4], sums[25]);
            sums[26] = mad(row.J[4], row.J[5], sums[26]);

            sums[27] = mad(row.J[5], row.J[5], sums[27]);
            
            // extra info here
            sums[28] += 1;
        }
    }

    #pragma unroll
    for(i = 0; i < 32; i++) {
        out[i+threadIdx*32] = sums[i];
    }
}

__attribute__((reqd_work_group_size(NUM_WI_3,1,1)))
__kernel void reduceKernel3 (
        __global __write_only float * restrict out,
        __global __read_only const TrackData * restrict J
) {
    uint threadIdx = get_global_id(0);
    uint globalSize = get_global_size(0);

    float sums[32];
    uint i, k;

    #pragma unroll
    for(i = 0; i < 32; ++i) {
        sums[i] = 0.0f;
    }

    for(i = 0; i < BATCHSIZE_3; i++) {
        const TrackData row = J[threadIdx + i*globalSize];
        if(row.result < 1) {
            if (row.result == -4) {
                sums[29] += 1;
            } else if (row.result == -5) {
                sums[30] += 1;
            } else if (row.result > -4) {
                sums[31] += 1;
            }
        } else {
            // Error part
            //sums[0] += row.error * row.error;
            sums[0] = mad(row.error, row.error, sums[0]);

            // JTe part
            #pragma unroll
            for(k = 0; k < 6; ++k) {
                sums[k+1] = mad(row.error, row.J[k], sums[k+1]);
            }

            sums[7] = mad(row.J[0], row.J[0], sums[7]);
            sums[8] = mad(row.J[0], row.J[1], sums[8]);
            sums[9] = mad(row.J[0], row.J[2], sums[9]);
            sums[10] = mad(row.J[0], row.J[3], sums[10]);
            sums[11] = mad(row.J[0], row.J[4], sums[11]);
            sums[12] = mad(row.J[0], row.J[5], sums[12]);

            sums[13] = mad(row.J[1], row.J[1], sums[13]);
            sums[14] = mad(row.J[1], row.J[2], sums[14]);
            sums[15] = mad(row.J[1], row.J[3], sums[15]);
            sums[16] = mad(row.J[1], row.J[4], sums[16]);
            sums[17] = mad(row.J[1], row.J[5], sums[17]);

            sums[18] = mad(row.J[2], row.J[2], sums[18]);
            sums[19] = mad(row.J[2], row.J[3], sums[19]);
            sums[20] = mad(row.J[2], row.J[4], sums[20]);
            sums[21] = mad(row.J[2], row.J[5], sums[21]);

            sums[22] = mad(row.J[3], row.J[3], sums[22]);
            sums[23] = mad(row.J[3], row.J[4], sums[23]);
            sums[24] = mad(row.J[3], row.J[5], sums[24]);

            sums[25] = mad(row.J[4], row.J[4], sums[25]);
            sums[26] = mad(row.J[4], row.J[5], sums[26]);

            sums[27] = mad(row.J[5], row.J[5], sums[27]);
            
            // extra info here
            sums[28] += 1;
        }
    }

    #pragma unroll
    for(i = 0; i < 32; i++) {
        out[i+threadIdx*32] = sums[i];
    }
}
