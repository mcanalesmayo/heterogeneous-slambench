/*

 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

/************** TYPES ***************/

#include "track_utils.h"

#define INVALID -2

typedef struct sTrackData {
    int result;
    float error;
    float J[6];
} TrackData;

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

__attribute__((reqd_work_group_size(X_LEVEL_TRACK_1,Y_LEVEL_TRACK_1,1)))
__kernel void trackKernel1 (
        __global __write_only TrackData * restrict output,
        __global __read_only const float * restrict inVertex,// float3
        __global __read_only const float * restrict inNormal,// float3
        __global __read_only const float * restrict refVertex,// float3
        __global __read_only const float * restrict refNormal,// float3
        const float4 Ttrack0, const float4 Ttrack1, const float4 Ttrack2,
        const float4 view0, const float4 view1, const float4 view2,
        const float dist_threshold,
        const float normal_threshold
) {
    const size_t pixelX = get_global_id(0);
    const size_t pixelY = get_global_id(1);

    float3 inNormalPixel = vload3(pixelX + X_LEVEL_TRACK_1 * pixelY, inNormal);

    if(inNormalPixel.x == INVALID ) {
        output[pixelX + X_COMPUTATION_SIZE * pixelY].result = -1;
        return;
    }

    float3 inVertexPixel = vload3(pixelX + X_LEVEL_TRACK_1 * pixelY, inVertex);
    const float3 projectedVertex = Mat4TimeFloat3(Ttrack0, Ttrack1, Ttrack2, inVertexPixel);
    const float3 projectedPos = Mat4TimeFloat3(view0, view1, view2, projectedVertex);
    const float2 projPixel = (float2) (projectedPos.x / projectedPos.z + 0.5f, projectedPos.y / projectedPos.z + 0.5f);

    if(projPixel.x < 0 || projPixel.x > X_COMPUTATION_SIZE-1 || projPixel.y < 0 || projPixel.y > Y_COMPUTATION_SIZE-1 ) {
        output[pixelX + X_COMPUTATION_SIZE * pixelY].result = -2;
        return;
    }

    const uint2 refPixel = (uint2) (projPixel.x, projPixel.y);
    const float3 referenceNormal = vload3(refPixel.x + X_COMPUTATION_SIZE * refPixel.y, refNormal);

    if(referenceNormal.x == INVALID) {
        output[pixelX + X_COMPUTATION_SIZE * pixelY].result = -3;
        return;
    }

    const float3 diff = vload3(refPixel.x + X_COMPUTATION_SIZE * refPixel.y, refVertex) - projectedVertex;
    const float3 projectedNormal = myrotate(Ttrack0, Ttrack1, Ttrack2, inNormalPixel);

    if(length(diff) > dist_threshold ) {
        output[pixelX + X_COMPUTATION_SIZE * pixelY].result = -4;
        return;
    }
    if(dot(projectedNormal, referenceNormal) < normal_threshold) {
        output[pixelX + X_COMPUTATION_SIZE * pixelY] .result = -5;
        return;
    }

    output[pixelX + X_COMPUTATION_SIZE * pixelY].result = 1;
    output[pixelX + X_COMPUTATION_SIZE * pixelY].error = dot(referenceNormal, diff);

    vstore3(referenceNormal, 0, (output[pixelX + X_COMPUTATION_SIZE * pixelY].J));
    vstore3(cross(projectedVertex, referenceNormal), 1, (output[pixelX + X_COMPUTATION_SIZE * pixelY].J));
}

__attribute__((reqd_work_group_size(X_LEVEL_TRACK_2,Y_LEVEL_TRACK_2,1)))
__kernel void trackKernel2 (
        __global __write_only TrackData * restrict output,
        __global __read_only const float * restrict inVertex,// float3
        __global __read_only const float * restrict inNormal,// float3
        __global __read_only const float * restrict refVertex,// float3
        __global __read_only const float * restrict refNormal,// float3
        const float4 Ttrack0, const float4 Ttrack1, const float4 Ttrack2,
        const float4 view0, const float4 view1, const float4 view2,
        const float dist_threshold,
        const float normal_threshold
) {
    const size_t pixelX = get_global_id(0);
    const size_t pixelY = get_global_id(1);

    float3 inNormalPixel = vload3(pixelX + X_LEVEL_TRACK_2 * pixelY, inNormal);

    if(inNormalPixel.x == INVALID ) {
        output[pixelX + X_COMPUTATION_SIZE * pixelY].result = -1;
        return;
    }

    float3 inVertexPixel = vload3(pixelX + X_LEVEL_TRACK_2 * pixelY, inVertex);
    const float3 projectedVertex = Mat4TimeFloat3(Ttrack0, Ttrack1, Ttrack2, inVertexPixel);
    const float3 projectedPos = Mat4TimeFloat3(view0, view1, view2, projectedVertex);
    const float2 projPixel = (float2) (projectedPos.x / projectedPos.z + 0.5f, projectedPos.y / projectedPos.z + 0.5f);

    if(projPixel.x < 0 || projPixel.x > X_COMPUTATION_SIZE-1 || projPixel.y < 0 || projPixel.y > Y_COMPUTATION_SIZE-1 ) {
        output[pixelX + X_COMPUTATION_SIZE * pixelY].result = -2;
        return;
    }

    const uint2 refPixel = (uint2) (projPixel.x, projPixel.y);
    const float3 referenceNormal = vload3(refPixel.x + X_COMPUTATION_SIZE * refPixel.y, refNormal);

    if(referenceNormal.x == INVALID) {
        output[pixelX + X_COMPUTATION_SIZE * pixelY].result = -3;
        return;
    }

    const float3 diff = vload3(refPixel.x + X_COMPUTATION_SIZE * refPixel.y, refVertex) - projectedVertex;
    const float3 projectedNormal = myrotate(Ttrack0, Ttrack1, Ttrack2, inNormalPixel);

    if(length(diff) > dist_threshold ) {
        output[pixelX + X_COMPUTATION_SIZE * pixelY].result = -4;
        return;
    }
    if(dot(projectedNormal, referenceNormal) < normal_threshold) {
        output[pixelX + X_COMPUTATION_SIZE * pixelY] .result = -5;
        return;
    }

    output[pixelX + X_COMPUTATION_SIZE * pixelY].result = 1;
    output[pixelX + X_COMPUTATION_SIZE * pixelY].error = dot(referenceNormal, diff);

    vstore3(referenceNormal, 0, (output[pixelX + X_COMPUTATION_SIZE * pixelY].J));
    vstore3(cross(projectedVertex, referenceNormal), 1, (output[pixelX + X_COMPUTATION_SIZE * pixelY].J));
}

__attribute__((reqd_work_group_size(X_LEVEL_TRACK_3,Y_LEVEL_TRACK_3,1)))
__kernel void trackKernel3 (
        __global __write_only TrackData * restrict output,
        __global __read_only const float * restrict inVertex,// float3
        __global __read_only const float * restrict inNormal,// float3
        __global __read_only const float * restrict refVertex,// float3
        __global __read_only const float * restrict refNormal,// float3
        const float4 Ttrack0, const float4 Ttrack1, const float4 Ttrack2,
        const float4 view0, const float4 view1, const float4 view2,
        const float dist_threshold,
        const float normal_threshold
) {
    const size_t pixelX = get_global_id(0);
    const size_t pixelY = get_global_id(1);

    float3 inNormalPixel = vload3(pixelX + X_LEVEL_TRACK_3 * pixelY, inNormal);

    if(inNormalPixel.x == INVALID ) {
        output[pixelX + X_COMPUTATION_SIZE * pixelY].result = -1;
        return;
    }

    float3 inVertexPixel = vload3(pixelX + X_LEVEL_TRACK_3 * pixelY, inVertex);
    const float3 projectedVertex = Mat4TimeFloat3(Ttrack0, Ttrack1, Ttrack2, inVertexPixel);
    const float3 projectedPos = Mat4TimeFloat3(view0, view1, view2, projectedVertex);
    const float2 projPixel = (float2) (projectedPos.x / projectedPos.z + 0.5f, projectedPos.y / projectedPos.z + 0.5f);

    if(projPixel.x < 0 || projPixel.x > X_COMPUTATION_SIZE-1 || projPixel.y < 0 || projPixel.y > Y_COMPUTATION_SIZE-1 ) {
        output[pixelX + X_COMPUTATION_SIZE * pixelY].result = -2;
        return;
    }

    const uint2 refPixel = (uint2) (projPixel.x, projPixel.y);
    const float3 referenceNormal = vload3(refPixel.x + X_COMPUTATION_SIZE * refPixel.y, refNormal);

    if(referenceNormal.x == INVALID) {
        output[pixelX + X_COMPUTATION_SIZE * pixelY].result = -3;
        return;
    }

    const float3 diff = vload3(refPixel.x + X_COMPUTATION_SIZE * refPixel.y, refVertex) - projectedVertex;
    const float3 projectedNormal = myrotate(Ttrack0, Ttrack1, Ttrack2, inNormalPixel);

    if(length(diff) > dist_threshold ) {
        output[pixelX + X_COMPUTATION_SIZE * pixelY].result = -4;
        return;
    }
    if(dot(projectedNormal, referenceNormal) < normal_threshold) {
        output[pixelX + X_COMPUTATION_SIZE * pixelY] .result = -5;
        return;
    }

    output[pixelX + X_COMPUTATION_SIZE * pixelY].result = 1;
    output[pixelX + X_COMPUTATION_SIZE * pixelY].error = dot(referenceNormal, diff);

    vstore3(referenceNormal, 0, (output[pixelX + X_COMPUTATION_SIZE * pixelY].J));
    vstore3(cross(projectedVertex, referenceNormal), 1, (output[pixelX + X_COMPUTATION_SIZE * pixelY].J));
}