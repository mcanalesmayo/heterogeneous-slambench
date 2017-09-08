/*

 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

 inline float sq(float r) {
    return r * r;
}

__kernel void bilateralFilterKernel( __global float * out,
        const __global float * in,
        const __global float * gaussian,
        const float e_d,
        const int r ) {

    const uint2 pos = (uint2) (get_global_id(0),get_global_id(1));
    const uint2 size = (uint2) (get_global_size(0),get_global_size(1));

    const float center = in[pos.x + size.x * pos.y];

    if ( center == 0 ) {
        out[pos.x + size.x * pos.y] = 0;
        return;
    }

    float sum = 0.0f;
    float t = 0.0f;
    // FIXME : sum and t diverge too much from cpp version
    for(int i = -r; i <= r; ++i) {
        for(int j = -r; j <= r; ++j) {
            const uint2 curPos = (uint2)(clamp(pos.x + i, 0u, size.x-1), clamp(pos.y + j, 0u, size.y-1));
            const float curPix = in[curPos.x + curPos.y * size.x];
            if(curPix > 0) {
                const float mod = sq(curPix - center);
                const float factor = gaussian[i + r] * gaussian[j + r] * exp(-mod / (2 * e_d * e_d));
                t += factor * curPix;
                sum += factor;
            } else {
                //std::cerr << "ERROR BILATERAL " <<pos.x+i<< " "<<pos.y+j<< " " <<curPix<<" \n";
            }
        }
    }
    out[pos.x + size.x * pos.y] = t / sum;

}