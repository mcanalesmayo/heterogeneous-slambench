/*

 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

/************** TYPES ***************/

typedef struct sTrackData {
    int result;
    float error;
    float J[6];
} TrackData;

__kernel void reduceKernel (
        __global float * out,
        __global const TrackData * J,
        const uint2 JSize,
        const uint2 size
) {

    const uint size_of_group = 64;
    const uint number_of_groups = 8;
    uint gridDim = number_of_groups;
    uint blockDim = size_of_group;

    float S[number_of_groups * size_of_group * 32];
    float sums_g[number_of_groups * size_of_group * 32];
    for (uint gid = 0; gid < size_of_group * number_of_groups; gid++) {
        uint blockIdx = gid / size_of_group;
        uint threadIdx = gid % size_of_group;

        const uint sline = threadIdx;

        float * sums = sums_g + blockIdx * size_of_group * 32;
        float * jtj = sums + 7;
        float * info = sums + 28;

        for(uint i = 0; i < 32; ++i) {
            sums[i] = 0.0f;   
        }

        for(uint y = blockIdx; y < size.y; y += gridDim) {
            for(uint x = sline; x < size.x; x += blockDim ) {
                const TrackData row = J[x + y * JSize.x];
                if(row.result < 1) {
                    info[1] += row.result == -4 ? 1 : 0;
                    info[2] += row.result == -5 ? 1 : 0;
                    info[3] += row.result > -4 ? 1 : 0;
                    continue;
                }

                // Error part
                //sums[0] += row.error * row.error;
                sums[0] = mad(row.error, row.error, sums[0]);

                // JTe part
                for(int i = 0; i < 6; ++i) {
                    sums[i+1] = mad(row.error, row.J[i], sums[i+1]);
                }

                jtj[0] = mad(row.J[0], row.J[0], jtj[0]);
                jtj[1] = mad(row.J[0], row.J[1], jtj[1]);
                jtj[2] = mad(row.J[0], row.J[2], jtj[2]);
                jtj[3] = mad(row.J[0], row.J[3], jtj[3]);
                jtj[4] = mad(row.J[0], row.J[4], jtj[4]);
                jtj[5] = mad(row.J[0], row.J[5], jtj[5]);

                jtj[6] = mad(row.J[1], row.J[1], jtj[6]);
                jtj[7] = mad(row.J[1], row.J[2], jtj[7]);
                jtj[8] = mad(row.J[1], row.J[3], jtj[8]);
                jtj[9] = mad(row.J[1], row.J[4], jtj[9]);
                jtj[10] = mad(row.J[1], row.J[5], jtj[10]);

                jtj[11] = mad(row.J[2], row.J[2], jtj[11]);
                jtj[12] = mad(row.J[2], row.J[3], jtj[12]);
                jtj[13] = mad(row.J[2], row.J[4], jtj[13]);
                jtj[14] = mad(row.J[2], row.J[5], jtj[14]);

                jtj[15] = mad(row.J[3], row.J[3], jtj[15]);
                jtj[16] = mad(row.J[3], row.J[4], jtj[16]);
                jtj[17] = mad(row.J[3], row.J[5], jtj[17]);

                jtj[18] = mad(row.J[4], row.J[4], jtj[18]);
                jtj[19] = mad(row.J[4], row.J[5], jtj[19]);

                jtj[20] = mad(row.J[5], row.J[5], jtj[20]);
                
                // extra info here
                info[0] += 1;
            }
        }

        for(int i = 0; i < 32; ++i) {
            S[blockIdx * sline * 32 + i] = sums[i];
        }
    }

    for (uint gid = 0; gid < size_of_group * number_of_groups; gid++) {
        uint blockIdx = gid / size_of_group;
        uint threadIdx = gid % size_of_group;

        const uint sline = threadIdx;

        if(sline < 32) { // sum up columns and copy to global memory in the final 32 threads
            for(unsigned i = 1; i < blockDim; ++i) {
                S[blockIdx * sline] += S[blockIdx * sline + i * 32];
            }
            out[sline+blockIdx*32] = S[blockIdx * sline];
        }
    }
}
