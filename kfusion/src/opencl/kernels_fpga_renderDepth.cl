/*

 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

__kernel void renderDepthKernel( __global uchar4 * out,
		__global float * depth,
		const float nearPlane,
		const float farPlane ) {

	const int posx = get_global_id(0);
	const int posy = get_global_id(1);
	const int sizex = get_global_size(0);
	float d= depth[posx + sizex * posy];
	if(d < nearPlane)
	vstore4((uchar4)(255, 255, 255, 0), posx + sizex * posy, (__global uchar*)out); // The forth value in uchar4 is padding for memory alignement and so it is for following uchar4 
	else {
		if(d > farPlane)
		vstore4((uchar4)(0, 0, 0, 0), posx + sizex * posy, (__global uchar*)out);
		else {
			float h =(d - nearPlane) / (farPlane - nearPlane);
			h *= 6.0f;
			const int sextant = (int)h;
			const float fract = h - sextant;
			const float mid1 = 0.25f + (0.5f*fract);
			const float mid2 = 0.75f - (0.5f*fract);
			switch (sextant)
			{
				case 0: vstore4((uchar4)(191, 255*mid1, 64, 0), posx + sizex * posy, (__global uchar*)out); break;
				case 1: vstore4((uchar4)(255*mid2, 191, 64, 0),posx + sizex * posy ,(__global uchar*)out); break;
				case 2: vstore4((uchar4)(64, 191, 255*mid1, 0),posx + sizex * posy ,(__global uchar*)out); break;
				case 3: vstore4((uchar4)(64, 255*mid2, 191, 0),posx + sizex * posy ,(__global uchar*)out); break;
				case 4: vstore4((uchar4)(255*mid1, 64, 191, 0),posx + sizex * posy ,(__global uchar*)out); break;
				case 5: vstore4((uchar4)(191, 64, 255*mid2, 0),posx + sizex * posy ,(__global uchar*)out); break;
			}
		}
	}
}