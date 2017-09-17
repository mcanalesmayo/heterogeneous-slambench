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

__kernel void renderTrackKernel( __global uchar3 * out,
		__global const TrackData * data ) {

	const int posx = get_global_id(0);
	const int posy = get_global_id(1);
	const int sizex = get_global_size(0);

	switch(data[posx + sizex * posy].result) {
		// The forth value in uchar4 is padding for memory alignement and so it is for following uchar4
		case  1: vstore4((uchar4)(128, 128, 128, 0), posx + sizex * posy, (__global uchar*)out); break; // ok	 GREY
		case -1: vstore4((uchar4)(000, 000, 000, 0), posx + sizex * posy, (__global uchar*)out); break; // no input BLACK
		case -2: vstore4((uchar4)(255, 000, 000, 0), posx + sizex * posy, (__global uchar*)out); break; // not in image RED
		case -3: vstore4((uchar4)(000, 255, 000, 0), posx + sizex * posy, (__global uchar*)out); break; // no correspondence GREEN
		case -4: vstore4((uchar4)(000, 000, 255, 0), posx + sizex * posy, (__global uchar*)out); break; // too far away BLUE
		case -5: vstore4((uchar4)(255, 255, 000, 0), posx + sizex * posy, (__global uchar*)out); break; // wrong normal YELLOW
		default: vstore4((uchar4)(255, 128, 128, 0), posx + sizex * posy, (__global uchar*)out); return;
	}
}