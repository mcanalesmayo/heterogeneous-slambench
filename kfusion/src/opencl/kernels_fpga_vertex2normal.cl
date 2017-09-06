/*

 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

/************** TYPES ***************/

#define INVALID -2 

/************** FUNCTIONS ***************/

__kernel void vertex2normalKernel( __global float * normal,    // float3
        const uint2 normalSize,
        const __global float * vertex ,
        const uint2 vertexSize ) {  // float3

    uint2 pixel = (uint2) (get_global_id(0),get_global_id(1));

    if(pixel.x >= vertexSize.x || pixel.y >= vertexSize.y )
    return;

    //const float3 left = vertex[(uint2)(max(int(pixel.x)-1,0), pixel.y)];
    //const float3 right = vertex[(uint2)(min(pixel.x+1,vertex.size.x-1), pixel.y)];
    //const float3 up = vertex[(uint2)(pixel.x, max(int(pixel.y)-1,0))];
    //const float3 down = vertex[(uint2)(pixel.x, min(pixel.y+1,vertex.size.y-1))];

    uint2 vleft = (uint2)(max((int)(pixel.x)-1,0), pixel.y);
    uint2 vright = (uint2)(min(pixel.x+1,vertexSize.x-1), pixel.y);
    uint2 vup = (uint2)(pixel.x, max((int)(pixel.y)-1,0));
    uint2 vdown = (uint2)(pixel.x, min(pixel.y+1,vertexSize.y-1));

    const float3 left = vload3(vleft.x + vertexSize.x * vleft.y,vertex);
    const float3 right = vload3(vright.x + vertexSize.x * vright.y,vertex);
    const float3 up = vload3(vup.x + vertexSize.x * vup.y,vertex);
    const float3 down = vload3(vdown.x + vertexSize.x * vdown.y,vertex);
    /*
     unsigned long int val =  0 ;
     val = max(((int) pixel.x)-1,0) + vertexSize.x * pixel.y;
     const float3 left   = vload3(   val,vertex);

     val =  min(pixel.x+1,vertexSize.x-1)                  + vertexSize.x *     pixel.y;
     const float3 right  = vload3(    val     ,vertex);
     val =   pixel.x                        + vertexSize.x *     max(((int) pixel.y)-1,0)  ;
     const float3 up     = vload3(  val ,vertex);
     val =  pixel.x                       + vertexSize.x *   min(pixel.y+1,vertexSize.y-1)   ;
     const float3 down   = vload3(  val   ,vertex);
     */
    if(left.z == 0 || right.z == 0|| up.z ==0 || down.z == 0) {
        //float3 n = vload3(pixel.x + normalSize.x * pixel.y,normal);
        //n.x=INVALID;
        vstore3((float3)(INVALID,INVALID,INVALID),pixel.x + normalSize.x * pixel.y,normal);
        return;
    }
    const float3 dxv = right - left;
    const float3 dyv = down - up;
    vstore3((float3) normalize(cross(dyv, dxv)), pixel.x + pixel.y * normalSize.x, normal );

}
