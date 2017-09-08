/*

 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

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

inline float3 Mat4TimeFloat3(const float4 M0, const float4 M1, const float4 M2, const float4 M3, float3 v) {
    Matrix4 M;
    M.data[0] = M0;
    M.data[1] = M1;
    M.data[2] = M2;
    M.data[3] = M3;
    return (float3)(
            dot((float3)(M.data[0].x, M.data[0].y, M.data[0].z), v)
                    + M.data[0].w,
            dot((float3)(M.data[1].x, M.data[1].y, M.data[1].z), v)
                    + M.data[1].w,
            dot((float3)(M.data[2].x, M.data[2].y, M.data[2].z), v)
                    + M.data[2].w);
}

inline float3 myrotate(const float4 M0, const float4 M1, const float4 M2, const float4 M3, const float3 v) {
    Matrix4 M;
    M.data[0] = M0;
    M.data[1] = M1;
    M.data[2] = M2;
    M.data[3] = M3;
    return (float3)(dot((float3)(M.data[0].x, M.data[0].y, M.data[0].z), v),
            dot((float3)(M.data[1].x, M.data[1].y, M.data[1].z), v),
            dot((float3)(M.data[2].x, M.data[2].y, M.data[2].z), v));
}

// inVertex iterate
__kernel void trackKernel (
        __global TrackData * output,
        const uint2 outputSize,
        __global const float * inVertex,// float3
        const uint2 inVertexSize,
        __global const float * inNormal,// float3
        const uint2 inNormalSize,
        __global const float * refVertex,// float3
        const uint2 refVertexSize,
        __global const float * refNormal,// float3
        const uint2 refNormalSize,
        const float4 Ttrack0, const float4 Ttrack1, const float4 Ttrack2, const float4 Ttrack3,
        const float4 view0, const float4 view1, const float4 view2, const float4 view3,
        const float dist_threshold,
        const float normal_threshold
) {
    Matrix4 Ttrack;
    Ttrack.data[0] = Ttrack0;
    Ttrack.data[1] = Ttrack1;
    Ttrack.data[2] = Ttrack2;
    Ttrack.data[3] = Ttrack3;

    Matrix4 view;
    view.data[0] = view0;
    view.data[1] = view1;
    view.data[2] = view2;
    view.data[3] = view3;

    const uint2 pixel = (uint2)(get_global_id(0),get_global_id(1));

    if(pixel.x >= inVertexSize.x || pixel.y >= inVertexSize.y ) {return;}

    float3 inNormalPixel = vload3(pixel.x + inNormalSize.x * pixel.y,inNormal);

    if(inNormalPixel.x == INVALID ) {
        output[pixel.x + outputSize.x * pixel.y].result = -1;
        return;
    }

    float3 inVertexPixel = vload3(pixel.x + inVertexSize.x * pixel.y,inVertex);
    const float3 projectedVertex = Mat4TimeFloat3 (Ttrack.data[0], Ttrack.data[1], Ttrack.data[2], Ttrack.data[3], inVertexPixel);
    const float3 projectedPos = Mat4TimeFloat3 (view.data[0], view.data[1], view.data[2], view.data[3], projectedVertex);
    const float2 projPixel = (float2) ( projectedPos.x / projectedPos.z + 0.5f, projectedPos.y / projectedPos.z + 0.5f);

    if(projPixel.x < 0 || projPixel.x > refVertexSize.x-1 || projPixel.y < 0 || projPixel.y > refVertexSize.y-1 ) {
        output[pixel.x + outputSize.x * pixel.y].result = -2;
        return;
    }

    const uint2 refPixel = (uint2) (projPixel.x, projPixel.y);
    const float3 referenceNormal = vload3(refPixel.x + refNormalSize.x * refPixel.y,refNormal);

    if(referenceNormal.x == INVALID) {
        output[pixel.x + outputSize.x * pixel.y].result = -3;
        return;
    }

    const float3 diff = vload3(refPixel.x + refVertexSize.x * refPixel.y,refVertex) - projectedVertex;
    const float3 projectedNormal = myrotate(Ttrack.data[0], Ttrack.data[1], Ttrack.data[2], Ttrack.data[3], inNormalPixel);

    if(length(diff) > dist_threshold ) {
        output[pixel.x + outputSize.x * pixel.y].result = -4;
        return;
    }
    if(dot(projectedNormal, referenceNormal) < normal_threshold) {
        output[pixel.x + outputSize.x * pixel.y] .result = -5;
        return;
    }

    output[pixel.x + outputSize.x * pixel.y].result = 1;
    output[pixel.x + outputSize.x * pixel.y].error = dot(referenceNormal, diff);

    vstore3(referenceNormal,0,(output[pixel.x + outputSize.x * pixel.y].J));
    vstore3(cross(projectedVertex, referenceNormal),1,(output[pixel.x + outputSize.x * pixel.y].J));

}