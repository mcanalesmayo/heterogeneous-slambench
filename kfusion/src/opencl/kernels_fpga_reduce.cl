/*

 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

/************** TYPES ***************/

#include "half_float_device.h"

__kernel void reduceKernel (
		__global half * restrict out,
		__global const TrackDataHalfFloat * restrict J,
		const uint2 JSize,
		const uint2 size
) {
	uint threadIdx = get_global_id(0);

	half sums[32];
	half * restrict jtj = sums + 7;
	half * restrict info = sums + 28;

	uint y, x, i;
	half one_half;
	vstore_half(1.0f, 0, &one_half);

	for(i = 0; i < 32; ++i) {
		vstore_half(0.0f, 0, &sums[i]);
	}

	for(y = 0; y < size.y; y++) {
		for(x = 0; x < size.x; x++) {
			const TrackDataHalfFloat row = J[x + y * JSize.x];
			if(row.result < 1) {
				if (row.result == -4) {
					info[1] += one_half;
				} else if (row.result == -5) {
					info[2] += one_half;
				} else if (row.result > -4) {
					info[3] += one_half;
				}
				continue;
			}

			// Error part
			sums[0] += row.error * row.error;

			// JTe part
			for(int i = 0; i < 6; ++i) {
				sums[i+1] += row.error * row.J[i];
			}

			jtj[0] += row.J[0] * row.J[0];
			jtj[1] += row.J[0] * row.J[1];
			jtj[2] += row.J[0] * row.J[2];
			jtj[3] += row.J[0] * row.J[3];
			jtj[4] += row.J[0] * row.J[4];
			jtj[5] += row.J[0] * row.J[5];

			jtj[6] += row.J[1] * row.J[1];
			jtj[7] += row.J[1] * row.J[2];
			jtj[8] += row.J[1] * row.J[3];
			jtj[9] += row.J[1] * row.J[4];
			jtj[10] += row.J[1] * row.J[5];

			jtj[11] += row.J[2] * row.J[2];
			jtj[12] += row.J[2] * row.J[3];
			jtj[13] += row.J[2] * row.J[4];
			jtj[14] += row.J[2] * row.J[5];

			jtj[15] += row.J[3] * row.J[3];
			jtj[16] += row.J[3] * row.J[4];
			jtj[17] += row.J[3] * row.J[5];

			jtj[18] += row.J[4] * row.J[4];
			jtj[19] += row.J[4] * row.J[5];

			jtj[20] += row.J[5] * row.J[5];
			// extra info here
			info[0] += one_half;
		}
	}

	for(i = 0; i < 32; i++) {
		out[i+threadIdx*32] = sums[i];
	}
}
