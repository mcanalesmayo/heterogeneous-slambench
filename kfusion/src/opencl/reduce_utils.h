#define X_LEVEL_1 320
#define Y_LEVEL_1 240
#define SIZE_LEVEL_1 X_LEVEL_1*Y_LEVEL_1
#define X_LEVEL_2 X_LEVEL_1/2
#define Y_LEVEL_2 Y_LEVEL_1/2
#define SIZE_LEVEL_2 X_LEVEL_2*Y_LEVEL_2
#define X_LEVEL_3 X_LEVEL_2/2
#define Y_LEVEL_3 Y_LEVEL_2/2
#define SIZE_LEVEL_3 X_LEVEL_3*Y_LEVEL_3

#define NUM_WI_1 200
#define NUM_WI_2 200
#define NUM_WI_3 200

#define BATCHSIZE_1 SIZE_LEVEL_1/NUM_WI_1
#define BATCHSIZE_2 SIZE_LEVEL_2/NUM_WI_2
#define BATCHSIZE_3 SIZE_LEVEL_3/NUM_WI_3