/*

 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

/************** TYPES ***************/

typedef struct sMatrix4 {
    float4 data[4];
} Matrix4;

/************** FUNCTIONS ***************/

inline float3 myrotate(const Matrix4 M, const float3 v) {
    return (float3)(dot((float3)(M.data[0].x, M.data[0].y, M.data[0].z), v),
            dot((float3)(M.data[1].x, M.data[1].y, M.data[1].z), v),
            dot((float3)(M.data[2].x, M.data[2].y, M.data[2].z), v));
}

__kernel void depth2vertexKernel( __global float * vertex, // float3
        const uint2 vertexSize ,
        const __global float * depth,
        const uint2 depthSize ,
        const float4 invK1,
        const float4 invK2,
        const float4 invK3,
        const float4 invK4 ) {

    Matrix4 invK;
    invK.data[0] = invK1;
    invK.data[1] = invK2;
    invK.data[2] = invK3;
    invK.data[3] = invK4;

    uint2 pixel = (uint2) (get_global_id(0),get_global_id(1));
    float3 vert = (float3)(get_global_id(0),get_global_id(1),1.0f);

    if(pixel.x >= depthSize.x || pixel.y >= depthSize.y ) {
        return;
    }

    float3 res = (float3) (0);

    if(depth[pixel.x + depthSize.x * pixel.y] > 0) {
        res = depth[pixel.x + depthSize.x * pixel.y] * (myrotate(invK, (float3)(pixel.x, pixel.y, 1.f)));
    }

    vstore3(res, pixel.x + vertexSize.x * pixel.y,vertex);  // vertex[pixel] =

}
