#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef struct sTrackDataHalfFloat {
    int result;
    half error;
    half J[6];
} TrackDataHalfFloat;