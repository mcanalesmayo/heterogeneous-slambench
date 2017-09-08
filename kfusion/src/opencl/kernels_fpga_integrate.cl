/*

 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

/************** TYPES ***************/

#define INVALID -2 

typedef struct sVolume {
    uint3 size;
    float3 dim;
    __global short2 * data;
} Volume;

typedef struct sTrackData {
    int result;
    float error;
    float J[6];
} TrackData;

typedef struct sMatrix4 {
    float4 data[4];
} Matrix4;

/************** FUNCTIONS ***************/

inline float sq(float r) {
    return r * r;
}

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

inline void setVolume(Volume v, uint3 pos, float2 d) {
    v.data[pos.x + pos.y * v.size.x + pos.z * v.size.x * v.size.y] = (short2)(
            d.x * 32766.0f, d.y);
}

inline float3 posVolume(const Volume v, const uint3 p) {
    return (float3)((p.x + 0.5f) * v.dim.x / v.size.x,
            (p.y + 0.5f) * v.dim.y / v.size.y,
            (p.z + 0.5f) * v.dim.z / v.size.z);
}

inline float2 getVolume(const Volume v, const uint3 pos) {
    const short2 d = v.data[pos.x + pos.y * v.size.x
            + pos.z * v.size.x * v.size.y];
    return (float2)(d.x * 0.00003051944088f, d.y); //  / 32766.0f
}

__kernel void integrateKernel (
        __global short2 * v_data,
        const uint3 v_size,
        const float3 v_dim,
        __global const float * depth,
        const uint2 depthSize,
        const float4 invTrack0, const float4 invTrack1, const float4 invTrack2, const float4 invTrack3,
        const float4 K0, const float4 K1, const float4 K2, const float4 K3,
        const float mu,
        const float maxweight ,
        const float3 delta ,
        const float3 cameraDelta
) {
    Matrix4 invTrack;
    invTrack.data[0] = invTrack0;
    invTrack.data[1] = invTrack1;
    invTrack.data[2] = invTrack2;
    invTrack.data[3] = invTrack3;

    Matrix4 K;
    K.data[0] = K0;
    K.data[1] = K1;
    K.data[2] = K2;
    K.data[3] = K3;

    Volume vol; vol.data = v_data; vol.size = v_size; vol.dim = v_dim;

    uint3 pix = (uint3) (get_global_id(0),get_global_id(1),0);
    const int sizex = get_global_size(0);

    float3 pos = Mat4TimeFloat3 (invTrack.data[0], invTrack.data[1], invTrack.data[2], invTrack.data[3], posVolume(vol,pix));
    float3 cameraX = Mat4TimeFloat3 ( K.data[0], K.data[1], K.data[2], K.data[3], pos);

    for(pix.z = 0; pix.z < vol.size.z; ++pix.z, pos += delta, cameraX += cameraDelta) {
        if(pos.z < 0.0001f) // some near plane constraint
        continue;
        const float2 pixel = (float2) (cameraX.x/cameraX.z + 0.5f, cameraX.y/cameraX.z + 0.5f);

        if(pixel.x < 0 || pixel.x > depthSize.x-1 || pixel.y < 0 || pixel.y > depthSize.y-1)
        continue;
        const uint2 px = (uint2)(pixel.x, pixel.y);
        float depthpx = depth[px.x + depthSize.x * px.y];

        if(depthpx == 0) continue;
        const float diff = ((depthpx) - cameraX.z) * sqrt(1+sq(pos.x/pos.z) + sq(pos.y/pos.z));

        if(diff > -mu) {
            const float sdf = fmin(1.f, diff/mu);
            float2 data = getVolume(vol,pix);
            data.x = clamp((data.y*data.x + sdf)/(data.y + 1), -1.f, 1.f);
            data.y = fmin(data.y+1, maxweight);
            setVolume(vol,pix, data);
        }
    }

}