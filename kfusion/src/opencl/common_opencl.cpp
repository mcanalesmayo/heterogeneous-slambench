/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#define EXTERNS
#include "common_opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include <sstream>
#include <cstring>

#define XSTR(x) #x
#define STR(x) XSTR(x)

#ifndef AOCX_PATH
#define AOCX_PATH "/home/mcanales/slambench/kfusion/src/opencl/kernels"
#endif

cl_int             clError;
cl_uint            num_platforms;
// first index corresponds to the platform
cl_platform_id    *platform_ids;
cl_context        *contexts;
cl_program        *programs;
// second index corresponds to the device in that platform
cl_device_id     **device_lists;
cl_command_queue **cmd_queues;

int opencl_clean(void) {

    // release resources
    clError &= clReleaseProgram(programs[0]);
    clError &= clReleaseProgram(programs[1]);
    clError &= clReleaseCommandQueue(cmd_queues[0][0]);
    clError &= clReleaseCommandQueue(cmd_queues[1][0]);
    clError &= clReleaseCommandQueue(cmd_queues[1][1]);
    clError &= clReleaseContext(contexts[0]);
    clError &= clReleaseContext(contexts[1]);

    free(cmd_queues[0]);
    free(cmd_queues[1]);
    free(cmd_queues);
    free(device_lists[0]);
    free(device_lists[1]);
    free(device_lists);
    free(programs);
    free(contexts);
    free(platform_ids);

    if (clError == CL_SUCCESS) return 0;
    else return -1;
}

int opencl_init(void) {
    size_t size;
    int ctxs_idx = 0;
    cl_uint num_devices;

    cl_platform_info platform_info;
    char platform_name[30];
    clError = clGetPlatformIDs(0, NULL, &num_platforms);
    if (clError != CL_SUCCESS) {
        printf("ERROR: clGetPlatformIDs(0,NULL,&num_platforms) failed\n");
        return -1; 
    }

    printf("Number of platforms: %d\n", num_platforms);
    platform_ids = (cl_platform_id *) malloc(num_platforms * sizeof(cl_platform_id));
    contexts = (cl_context *) malloc(num_platforms * sizeof(cl_context));
    programs = (cl_program *) malloc(num_platforms * sizeof(cl_program));
    device_lists = (cl_device_id **) malloc(num_platforms * sizeof(cl_device_id *));
    cmd_queues = (cl_command_queue **) malloc(num_platforms * sizeof(cl_command_queue *));

    if (clGetPlatformIDs(num_platforms, platform_ids, NULL) != CL_SUCCESS) {
        printf("ERROR: clGetPlatformIDs(num_platforms,platform_ids,NULL) failed\n");
        return -1;
    }

    for(int i=0; i<num_platforms; i++){
        clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, 30, platform_name, NULL);
        printf("Platform %d: %s\n", i, platform_name);
    }

    /* ---- */
    /* FPGA */
    /* ---- */

    // Intel Altera is idx=0
    // cl_context_properties:
    // Specifies a list of context property names and their corresponding values. Each property name is immediately followed by the corresponding desired value.
    // The list is terminated with 0. properties can be NULL in which case the platform that is selected is implementation-defined.
    // The list of supported properties is described in the table below.
    cl_context_properties ctxprop_fpga[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platform_ids[0], 0};

    contexts[0] = clCreateContextFromType(ctxprop_fpga, CL_DEVICE_TYPE_ACCELERATOR, NULL, NULL, NULL);
    if(!contexts[0]) {
        printf("ERROR: clCreateContextFromType(%s) failed\n", "FPGA");
        return -1;
    }

    // get the list of FPGAs
    clError = clGetContextInfo(contexts[0], CL_CONTEXT_DEVICES, 0, NULL, &size);
    num_devices = (cl_uint) (size / sizeof(cl_device_id));
    device_lists[0] = (cl_device_id *) malloc(num_devices * sizeof(cl_device_id));
    cmd_queues[0] = (cl_command_queue *) malloc(num_devices * sizeof(cl_command_queue));
    
    if( clError != CL_SUCCESS || num_devices < 1 ) {
        printf("ERROR: clGetContextInfo() failed\n");
        return -1;
    }

    clError = clGetContextInfo(contexts[0], CL_CONTEXT_DEVICES, size, device_lists[0], NULL);
    if( clError != CL_SUCCESS ) {
        printf("ERROR: clGetContextInfo() failed\n");
        return -1;
    }

    for(int j=0; j<num_devices; j++){
        cmd_queues[0][j] = clCreateCommandQueue(contexts[0], device_lists[0][j], 0, NULL);
        if( !cmd_queues[0][j] ) {
            printf("ERROR: clCreateCommandQueue() FPGA %d failed\n", j);
            return -1;
        }
    }

    clError = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_ALL, num_devices, device_lists[0], NULL);
    if (clError != CL_SUCCESS){
        printf("ERROR: Query for FPGA device ids\n");
        return -1;
    }

    // create and build the FPGA program
    std::string binary_file = aocl_utils::getBoardBinaryFile(AOCX_PATH, device_lists[0][0]);
    printf("Using AOCX: %s\n", binary_file.c_str());
    cl_program program_fpga = aocl_utils::createProgramFromBinary(contexts[0], binary_file.c_str(), device_lists[0], 1);
    clError = clBuildProgram(program_fpga, 0, NULL, NULL, NULL, NULL);
    if (clError != CL_SUCCESS) {
        printf("ERROR: FPGA clBuildProgram() => %d\n", clError);
        return -1;
    }

    /* ---- */
    /* GPUs */
    /* ---- */

    // NVIDIA CUDA is idx=1
    cl_context_properties ctxprop_gpu[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platform_ids[1], 0};
    contexts[1] = clCreateContextFromType(ctxprop_gpu, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);
    if( !contexts[1] ) {
        printf("ERROR: clCreateContextFromType(%s) failed\n", "GPU");
        return -1;
    }

    clError = clGetContextInfo(contexts[1], CL_CONTEXT_DEVICES, 0, NULL, &size);
    num_devices = (int) (size / sizeof(cl_device_id));
    device_lists[1] = (cl_device_id *) malloc(num_devices * sizeof(cl_device_id));
    cmd_queues[1] = (cl_command_queue *) malloc(num_devices * sizeof(cl_command_queue));
    
    if( clError != CL_SUCCESS || num_devices < 1 ) {
        printf("ERROR: clGetContextInfo() failed\n");
        return -1;
    }
    clError = clGetContextInfo(contexts[1], CL_CONTEXT_DEVICES, size, device_lists[1], NULL);
    if( clError != CL_SUCCESS ) {
        printf("ERROR: clGetContextInfo() failed\n");
        return -1;
    }

    for(int j=0; j<num_devices; j++){
        cmd_queues[1][j] = clCreateCommandQueue(contexts[1], device_lists[1][j], 0, NULL);
        if( !cmd_queues[1][j] ) {
            printf("ERROR: clCreateCommandQueue() GPU %d failed\n", j);
            return -1;
        }   
    }

    // try to read the kernel source
    int sourcesize = 1024*1024;
    char *source = (char *) calloc(sourcesize, sizeof(char)); 
    if(!source) {
        printf("ERROR: calloc(%d) failed\n", sourcesize);
        return -1;
    }

    char const * tempchar = "./kernels.cl";
    FILE *fp = fopen(tempchar, "rb"); 
    if(!fp) {
        printf("ERROR: unable to open '%s'\n", tempchar);
        return -1;
    }
    fread(source + strlen(source), sourcesize, 1, fp);
    fclose(fp);

    cl_int clError = 0;
    const char *slist[2] = { source, 0 };

    // create and build the GPU program
    cl_program program_gpus = clCreateProgramWithSource(contexts[1], 1, slist, NULL, &clError);
    if(clError != CL_SUCCESS) {
        printf("ERROR: GPUs clCreateProgramWithSource() => %d\n", clError);
        return -1;
    }
    clError = clBuildProgram(program_gpus, 0, NULL, NULL, NULL, NULL);
    if(clError != CL_SUCCESS) {
        printf("ERROR: GPUs clBuildProgram() => %d\n", clError);
        return -1;
    }

    return 0;

}
