/*

 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

/************** TYPES ***************/

#define FRACT_BITS 16
#define FRACT_BITS_D2 FRACT_BITS/2
#define MULT(x, y) ( ((x >> FRACT_BITS_D2) * (y >> FRACT_BITS_D2)) )

typedef struct sTrackDataFixedPoint {
	int result;
	int error;
	int J[6];
} TrackDataFixedPoint;

__kernel void reduceKernel (
		__global int * restrict out,
		__global const TrackDataFixedPoint * restrict J,
		const uint2 JSize,
		const uint2 size
) {
	uint threadIdx = get_global_id(0);
	uint globalSize = get_global_size(0);

	int sums[32];
	int * restrict jtj = sums + 7;
	int * restrict info = sums + 28;

	uint y, x, i;

	for(i = 0; i < 32; ++i) {
		sums[i] = 0;
	}

	for(y = 0; y < size.y; y++) {
		for(x = 0; x < size.x; x++) {
			const TrackDataFixedPoint row = J[x + y * JSize.x];
			if(row.result < 1) {
				info[1] += row.result == -4 ? 1 : 0;
				info[2] += row.result == -5 ? 1 : 0;
				info[3] += row.result > -4 ? 1 : 0;
				continue;
			}

			// Error part
			//sums[0] += row.error * row.error;
			sums[0] += MULT(row.error, row.error);

			// JTe part
			for(i = 0; i < 6; ++i) {
				sums[i+1] += MULT(row.error, row.J[i]);
			}

			jtj[0] += MULT(row.J[0], row.J[0]);
			jtj[1] += MULT(row.J[0], row.J[1]);
			jtj[2] += MULT(row.J[0], row.J[2]);
			jtj[3] += MULT(row.J[0], row.J[3]);
			jtj[4] += MULT(row.J[0], row.J[4]);
			jtj[5] += MULT(row.J[0], row.J[5]);

			jtj[6] += MULT(row.J[1], row.J[1]);
			jtj[7] += MULT(row.J[1], row.J[2]);
			jtj[8] += MULT(row.J[1], row.J[3]);
			jtj[9] += MULT(row.J[1], row.J[4]);
			jtj[10] += MULT(row.J[1], row.J[5]);

			jtj[11] += MULT(row.J[2], row.J[2]);
			jtj[12] += MULT(row.J[2], row.J[3]);
			jtj[13] += MULT(row.J[2], row.J[4]);
			jtj[14] += MULT(row.J[2], row.J[5]);

			jtj[15] += MULT(row.J[3], row.J[3]);
			jtj[16] += MULT(row.J[3], row.J[4]);
			jtj[17] += MULT(row.J[3], row.J[5]);

			jtj[18] += MULT(row.J[4], row.J[4]);
			jtj[19] += MULT(row.J[4], row.J[5]);

			jtj[20] += MULT(row.J[5], row.J[5]);

			// extra info here
			info[0] += 1;
		}
	}

	for(i = 0; i < 32; i++) {
		out[i+threadIdx*32] = sums[i];
	}
}
