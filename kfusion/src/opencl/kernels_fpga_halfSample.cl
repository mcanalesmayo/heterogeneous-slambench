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

__kernel void halfSampleRobustImageKernel(__global float * out,
		__global const float * in,
		const uint2 inSize,
		const float e_d,
		const int r) {

	uint2 pixel = (uint2) (get_global_id(0),get_global_id(1));
	uint2 outSize = inSize / 2;

	const uint2 centerPixel = 2 * pixel;

	float sum = 0.0f;
	float t = 0.0f;
	const float center = in[centerPixel.x + centerPixel.y * inSize.x];
	for(int i = -r + 1; i <= r; ++i) {
		for(int j = -r + 1; j <= r; ++j) {
			int2 from = (int2)(clamp((int2)(centerPixel.x + j, centerPixel.y + i), (int2)(0), (int2)(inSize.x - 1, inSize.y - 1)));
			float current = in[from.x + from.y * inSize.x];
			if(fabs(current - center) < e_d) {
				sum += 1.0f;
				t += current;
			}
		}
	}
	out[pixel.x + pixel.y * outSize.x] = t / sum;

}

