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
		__global float * restrict out,
		__global const TrackData * restrict J,
		const uint2 JSize,
		const uint2 size
) {
	uint threadIdx = get_global_id(0);
	uint globalSize = get_global_size(0);
	uint batchSize = (JSize.y * JSize.x) / globalSize;

	float sums[32];

	uint i, k;

	#pragma unroll
	for(i = 0; i < 32; ++i) {
		sums[i] = 0.0f;
	}

	for(i = 0; i < batchSize; i++) {
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
