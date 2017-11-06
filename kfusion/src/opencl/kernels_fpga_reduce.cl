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
	uint yBatchSize = JSize.y / globalSize;

	float sums[32];
	float * restrict jtj = sums + 7;
	float * restrict info = sums + 28;

	uint y, y_aux, x, i;

	for(i = 0; i < 32; ++i) {
		sums[i] = 0.0f;
	}

	for(y = 0; y < yBatchSize; y++) {
		y_aux = y + yBatchSize;
		for(x = 0; x < size.x; x++) {
			const TrackData row = J[x + y_aux * JSize.x];
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
			for(i = 0; i < 6; ++i) {
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

	for(i = 0; i < 32; i++) {
		out[i+threadIdx*32] = sums[i];
	}
}
