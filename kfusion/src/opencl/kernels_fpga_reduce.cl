/*

 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

/************** TYPES ***************/

#include "fixed_point.h"

#define NUM_WI_1 200
#define NUM_WI_2 200
#define NUM_WI_3 200

#define BATCHSIZE_1 (320*240)/NUM_WI_1
#define BATCHSIZE_2 (160*120)/NUM_WI_2
#define BATCHSIZE_3 (80*60)/NUM_WI_3

__attribute__((reqd_work_group_size(NUM_WI_1,1,1)))
__kernel void reduceKernel1 (
		__global int * restrict out,
		__global float * restrict outJTe,
		__global const TrackDataFixedPoint * restrict J
) {
	uint threadIdx = get_global_id(0);
	uint globalSize = get_global_size(0);

	int sums[26];
	int * restrict jtj = sums + 1;
	int * restrict info = sums + 22;
	float sumsJTe[6];

	uint i, k;
	float pixelError;
	int pixelErrorPrep, rowJAux[6];

	#pragma unroll
	for(i = 0; i < 26; ++i) {
		sums[i] = 0;
	}
	#pragma unroll
	for(i = 0; i < 6; ++i) {
		sumsJTe[i] = 0.0f;
	}

	for(i = 0; i < BATCHSIZE_1; i++) {
		const TrackDataFixedPoint row = J[threadIdx + i*globalSize];
		if(row.result < 1) {
			if (row.result == -4) {
				info[1] += 1;
			} else if (row.result == -5) {
				info[2] += 1;
			} else if (row.result > -4) {
				info[3] += 1;
			}
		} else {
			// Prepare for fixed point computation
			pixelErrorPrep = PREPARE_FOR_MULT(row.error, FRACT_BITS_ERROR_D2);
			#pragma unroll
			for(k = 0; k < 6; k++) {
				rowJAux[k] = PREPARE_FOR_MULT(row.J[k], FRACT_BITS_J_D2);
			}

			// Compute output
			// Error part
			sums[0] += PREPARED_MULT(pixelErrorPrep, pixelErrorPrep);

			// JTe part
			pixelError = FIXED2FLOAT(row.error, FRACT_BITS_ERROR);
			#pragma unroll
			for(k = 0; k < 6; k++) {
				sumsJTe[k] += pixelError * FIXED2FLOAT(row.J[k], FRACT_BITS_J);
			}

			jtj[0] += PREPARED_MULT(rowJAux[0], rowJAux[0]);
			jtj[1] += PREPARED_MULT(rowJAux[0], rowJAux[1]);
			jtj[2] += PREPARED_MULT(rowJAux[0], rowJAux[2]);
			jtj[3] += PREPARED_MULT(rowJAux[0], rowJAux[3]);
			jtj[4] += PREPARED_MULT(rowJAux[0], rowJAux[4]);
			jtj[5] += PREPARED_MULT(rowJAux[0], rowJAux[5]);

			jtj[6] += PREPARED_MULT(rowJAux[1], rowJAux[1]);
			jtj[7] += PREPARED_MULT(rowJAux[1], rowJAux[2]);
			jtj[8] += PREPARED_MULT(rowJAux[1], rowJAux[3]);
			jtj[9] += PREPARED_MULT(rowJAux[1], rowJAux[4]);
			jtj[10] += PREPARED_MULT(rowJAux[1], rowJAux[5]);

			jtj[11] += PREPARED_MULT(rowJAux[2], rowJAux[2]);
			jtj[12] += PREPARED_MULT(rowJAux[2], rowJAux[3]);
			jtj[13] += PREPARED_MULT(rowJAux[2], rowJAux[4]);
			jtj[14] += PREPARED_MULT(rowJAux[2], rowJAux[5]);

			jtj[15] += PREPARED_MULT(rowJAux[3], rowJAux[3]);
			jtj[16] += PREPARED_MULT(rowJAux[3], rowJAux[4]);
			jtj[17] += PREPARED_MULT(rowJAux[3], rowJAux[5]);

			jtj[18] += PREPARED_MULT(rowJAux[4], rowJAux[4]);
			jtj[19] += PREPARED_MULT(rowJAux[4], rowJAux[5]);

			jtj[20] += PREPARED_MULT(rowJAux[5], rowJAux[5]);

			// extra info here
			info[0] += 1;
		}
	}

	#pragma unroll
	for(i = 0; i < 26; i++) {
		out[threadIdx*26 + i] = sums[i];
	}
	#pragma unroll
	for(i = 0; i < 6; i++) {
		outJTe[threadIdx*6 + i] = sumsJTe[i];
	}
}

__attribute__((reqd_work_group_size(NUM_WI_2,1,1)))
__kernel void reduceKernel2 (
		__global int * restrict out,
		__global float * restrict outJTe,
		__global const TrackDataFixedPoint * restrict J
) {
	uint threadIdx = get_global_id(0);
	uint globalSize = get_global_size(0);

	int sums[26];
	int * restrict jtj = sums + 1;
	int * restrict info = sums + 22;
	float sumsJTe[6];

	uint i, k;
	float pixelError;
	int pixelErrorPrep, rowJAux[6];

	#pragma unroll
	for(i = 0; i < 26; ++i) {
		sums[i] = 0;
	}
	#pragma unroll
	for(i = 0; i < 6; ++i) {
		sumsJTe[i] = 0.0f;
	}

	for(i = 0; i < BATCHSIZE_2; i++) {
		const TrackDataFixedPoint row = J[threadIdx + i*globalSize];
		if(row.result < 1) {
			if (row.result == -4) {
				info[1] += 1;
			} else if (row.result == -5) {
				info[2] += 1;
			} else if (row.result > -4) {
				info[3] += 1;
			}
		} else {
			// Prepare for fixed point computation
			pixelErrorPrep = PREPARE_FOR_MULT(row.error, FRACT_BITS_ERROR_D2);
			#pragma unroll
			for(k = 0; k < 6; k++) {
				rowJAux[k] = PREPARE_FOR_MULT(row.J[k], FRACT_BITS_J_D2);
			}

			// Compute output
			// Error part
			sums[0] += PREPARED_MULT(pixelErrorPrep, pixelErrorPrep);

			// JTe part
			pixelError = FIXED2FLOAT(row.error, FRACT_BITS_ERROR);
			#pragma unroll
			for(k = 0; k < 6; k++) {
				sumsJTe[k] += pixelError * FIXED2FLOAT(row.J[k], FRACT_BITS_J);
			}

			jtj[0] += PREPARED_MULT(rowJAux[0], rowJAux[0]);
			jtj[1] += PREPARED_MULT(rowJAux[0], rowJAux[1]);
			jtj[2] += PREPARED_MULT(rowJAux[0], rowJAux[2]);
			jtj[3] += PREPARED_MULT(rowJAux[0], rowJAux[3]);
			jtj[4] += PREPARED_MULT(rowJAux[0], rowJAux[4]);
			jtj[5] += PREPARED_MULT(rowJAux[0], rowJAux[5]);

			jtj[6] += PREPARED_MULT(rowJAux[1], rowJAux[1]);
			jtj[7] += PREPARED_MULT(rowJAux[1], rowJAux[2]);
			jtj[8] += PREPARED_MULT(rowJAux[1], rowJAux[3]);
			jtj[9] += PREPARED_MULT(rowJAux[1], rowJAux[4]);
			jtj[10] += PREPARED_MULT(rowJAux[1], rowJAux[5]);

			jtj[11] += PREPARED_MULT(rowJAux[2], rowJAux[2]);
			jtj[12] += PREPARED_MULT(rowJAux[2], rowJAux[3]);
			jtj[13] += PREPARED_MULT(rowJAux[2], rowJAux[4]);
			jtj[14] += PREPARED_MULT(rowJAux[2], rowJAux[5]);

			jtj[15] += PREPARED_MULT(rowJAux[3], rowJAux[3]);
			jtj[16] += PREPARED_MULT(rowJAux[3], rowJAux[4]);
			jtj[17] += PREPARED_MULT(rowJAux[3], rowJAux[5]);

			jtj[18] += PREPARED_MULT(rowJAux[4], rowJAux[4]);
			jtj[19] += PREPARED_MULT(rowJAux[4], rowJAux[5]);

			jtj[20] += PREPARED_MULT(rowJAux[5], rowJAux[5]);

			// extra info here
			info[0] += 1;
		}
	}

	#pragma unroll
	for(i = 0; i < 26; i++) {
		out[threadIdx*26 + i] = sums[i];
	}
	#pragma unroll
	for(i = 0; i < 6; i++) {
		outJTe[threadIdx*6 + i] = sumsJTe[i];
	}
}

__attribute__((reqd_work_group_size(NUM_WI_3,1,1)))
__kernel void reduceKernel3 (
		__global int * restrict out,
		__global float * restrict outJTe,
		__global const TrackDataFixedPoint * restrict J
) {
	uint threadIdx = get_global_id(0);
	uint globalSize = get_global_size(0);

	int sums[26];
	int * restrict jtj = sums + 1;
	int * restrict info = sums + 22;
	float sumsJTe[6];

	uint i, k;
	float pixelError;
	int pixelErrorPrep, rowJAux[6];

	#pragma unroll
	for(i = 0; i < 26; ++i) {
		sums[i] = 0;
	}
	#pragma unroll
	for(i = 0; i < 6; ++i) {
		sumsJTe[i] = 0.0f;
	}

	for(i = 0; i < BATCHSIZE_3; i++) {
		const TrackDataFixedPoint row = J[threadIdx + i*globalSize];
		if(row.result < 1) {
			if (row.result == -4) {
				info[1] += 1;
			} else if (row.result == -5) {
				info[2] += 1;
			} else if (row.result > -4) {
				info[3] += 1;
			}
		} else {
			// Prepare for fixed point computation
			pixelErrorPrep = PREPARE_FOR_MULT(row.error, FRACT_BITS_ERROR_D2);
			#pragma unroll
			for(k = 0; k < 6; k++) {
				rowJAux[k] = PREPARE_FOR_MULT(row.J[k], FRACT_BITS_J_D2);
			}

			// Compute output
			// Error part
			sums[0] += PREPARED_MULT(pixelErrorPrep, pixelErrorPrep);

			// JTe part
			pixelError = FIXED2FLOAT(row.error, FRACT_BITS_ERROR);
			#pragma unroll
			for(k = 0; k < 6; k++) {
				sumsJTe[k] += pixelError * FIXED2FLOAT(row.J[k], FRACT_BITS_J);
			}

			jtj[0] += PREPARED_MULT(rowJAux[0], rowJAux[0]);
			jtj[1] += PREPARED_MULT(rowJAux[0], rowJAux[1]);
			jtj[2] += PREPARED_MULT(rowJAux[0], rowJAux[2]);
			jtj[3] += PREPARED_MULT(rowJAux[0], rowJAux[3]);
			jtj[4] += PREPARED_MULT(rowJAux[0], rowJAux[4]);
			jtj[5] += PREPARED_MULT(rowJAux[0], rowJAux[5]);

			jtj[6] += PREPARED_MULT(rowJAux[1], rowJAux[1]);
			jtj[7] += PREPARED_MULT(rowJAux[1], rowJAux[2]);
			jtj[8] += PREPARED_MULT(rowJAux[1], rowJAux[3]);
			jtj[9] += PREPARED_MULT(rowJAux[1], rowJAux[4]);
			jtj[10] += PREPARED_MULT(rowJAux[1], rowJAux[5]);

			jtj[11] += PREPARED_MULT(rowJAux[2], rowJAux[2]);
			jtj[12] += PREPARED_MULT(rowJAux[2], rowJAux[3]);
			jtj[13] += PREPARED_MULT(rowJAux[2], rowJAux[4]);
			jtj[14] += PREPARED_MULT(rowJAux[2], rowJAux[5]);

			jtj[15] += PREPARED_MULT(rowJAux[3], rowJAux[3]);
			jtj[16] += PREPARED_MULT(rowJAux[3], rowJAux[4]);
			jtj[17] += PREPARED_MULT(rowJAux[3], rowJAux[5]);

			jtj[18] += PREPARED_MULT(rowJAux[4], rowJAux[4]);
			jtj[19] += PREPARED_MULT(rowJAux[4], rowJAux[5]);

			jtj[20] += PREPARED_MULT(rowJAux[5], rowJAux[5]);

			// extra info here
			info[0] += 1;
		}
	}

	#pragma unroll
	for(i = 0; i < 26; i++) {
		out[threadIdx*26 + i] = sums[i];
	}
	#pragma unroll
	for(i = 0; i < 6; i++) {
		outJTe[threadIdx*6 + i] = sumsJTe[i];
	}
}