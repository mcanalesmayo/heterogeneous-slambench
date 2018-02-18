/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC programs[1]me Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#include "common_opencl.h"
#include <kernels.h>

#include <TooN/TooN.h>
#include <TooN/se3.h>
#include <TooN/GR_SVD.h>

inline double benchmark_tock() {
	synchroniseDevices();
#ifdef __APPLE__
	clock_serv_t cclock;
	mach_timespec_t clockData;
	host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock);
	clock_get_time(cclock, &clockData);
	mach_port_deallocate(mach_task_self(), cclock);
#else
	struct timespec clockData;
	clock_gettime(CLOCK_MONOTONIC, &clockData);
#endif
	return (double) clockData.tv_sec + clockData.tv_nsec / 1000000000.0;
}

#define NUM_THREADS_REDUCE_KERNEL 200

double startOfTiming, endOfTiming;

////// USE BY KFUSION CLASS ///////////////

// input once
cl_mem ocl_gaussian = NULL;

// inter-frame
Matrix4 oldPose;
Matrix4 raycastPose;

TrackData * trackingResult;
float3 * vertex;
float3 * normal;
float3 ** inputVertex;
float3 ** inputNormal;

cl_mem ocl_vertex_GPU = NULL;
cl_mem ocl_normal_GPU = NULL;
cl_mem ocl_vertex_FPGA = NULL;
cl_mem ocl_normal_FPGA = NULL;
cl_mem ocl_volume_data = NULL;
cl_mem ocl_depth_buffer = NULL;
cl_mem ocl_output_render_buffer = NULL; // Common buffer for rendering track, depth and volume

// intra-frame
cl_mem ocl_reduce_output_buffer = NULL;
cl_mem ocl_trackingResult_FPGA = NULL;
cl_mem ocl_FloatDepth = NULL;
cl_mem * ocl_ScaledDepth = NULL;
cl_mem * ocl_inputVertex_GPU = NULL;
cl_mem * ocl_inputNormal_GPU = NULL;
cl_mem * ocl_inputVertex_FPGA = NULL;
cl_mem * ocl_inputNormal_FPGA = NULL;
float * reductionoutput = NULL;

// kernels
cl_kernel mm2meters_ocl_kernel;
cl_kernel bilateralFilter_ocl_kernel;
cl_kernel halfSampleRobustImage_ocl_kernel;
cl_kernel depth2vertex_ocl_kernel;
cl_kernel vertex2normal_ocl_kernel;
cl_kernel track_ocl_kernel;
cl_kernel reduce_ocl_kernel[3];
cl_kernel integrate_ocl_kernel;
cl_kernel raycast_ocl_kernel;
cl_kernel renderVolume_ocl_kernel;
cl_kernel renderLight_ocl_kernel;
cl_kernel renderDepth_ocl_kernel;
cl_kernel initVolume_ocl_kernel;

// reduction parameters
static const size_t size_of_group = 64;
static const size_t number_of_groups = 8;

uint2 computationSizeBkp = make_uint2(0, 0);
uint2 outputImageSizeBkp = make_uint2(0, 0);

void init() {
	if (opencl_init()) exit(1);
}

void clean() {
	if (opencl_clean()) exit(1);
}

void Kfusion::languageSpecificConstructor() {
	init();

	cl_ulong maxMemAlloc;
	clGetDeviceInfo(device_lists[1][0], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxMemAlloc), &maxMemAlloc, NULL);
	
	if (maxMemAlloc < sizeof(float4) * computationSize.x * computationSize.y) {
		std::cerr << "OpenCL maximum allocation does not support the computation size." << std::endl;
		exit(1);
	}
	if (maxMemAlloc < sizeof(TrackData) * computationSize.x * computationSize.y) {
		std::cerr << "OpenCL maximum allocation does not support the computation size." << std::endl;
		exit(1);
	}
	if (maxMemAlloc < sizeof(short2) * volumeResolution.x * volumeResolution.y * volumeResolution.z) {
		std::cerr << "OpenCL maximum allocation does not support the volume resolution." << std::endl;
		exit(1);
	}
	
	ocl_FloatDepth = clCreateBuffer(contexts[1], CL_MEM_READ_WRITE, sizeof(float) * computationSize.x * computationSize.y, NULL, &clError);
	checkErr(clError, "clCreateBuffer");
	ocl_ScaledDepth = (cl_mem*) malloc(sizeof(cl_mem) * iterations.size());
	ocl_inputVertex_GPU = (cl_mem*) malloc(sizeof(cl_mem) * iterations.size());
	ocl_inputNormal_GPU = (cl_mem*) malloc(sizeof(cl_mem) * iterations.size());
	ocl_inputVertex_FPGA = (cl_mem*) malloc(sizeof(cl_mem) * iterations.size());
	ocl_inputNormal_FPGA = (cl_mem*) malloc(sizeof(cl_mem) * iterations.size());

	for (unsigned int i = 0; i < iterations.size(); ++i) {
		ocl_ScaledDepth[i] = clCreateBuffer(contexts[1], CL_MEM_READ_WRITE, sizeof(float) * (computationSize.x * computationSize.y) / (int) pow(2, i), NULL, &clError);
		checkErr(clError, "clCreateBuffer");
		ocl_inputVertex_GPU[i] = clCreateBuffer(contexts[1], CL_MEM_READ_WRITE, sizeof(float3) * (computationSize.x * computationSize.y) / (int) pow(2, i), NULL, &clError);
		checkErr(clError, "clCreateBuffer");
		ocl_inputNormal_GPU[i] = clCreateBuffer(contexts[1], CL_MEM_READ_WRITE, sizeof(float3) * (computationSize.x * computationSize.y) / (int) pow(2, i), NULL, &clError);
		checkErr(clError, "clCreateBuffer");
		ocl_inputVertex_FPGA[i] = clCreateBuffer(contexts[0], CL_MEM_READ_ONLY, sizeof(float3) * (computationSize.x * computationSize.y) / (int) pow(2, i), NULL, &clError);
		checkErr(clError, "clCreateBuffer");
		ocl_inputNormal_FPGA[i] = clCreateBuffer(contexts[0], CL_MEM_READ_ONLY, sizeof(float3) * (computationSize.x * computationSize.y) / (int) pow(2, i), NULL, &clError);
		checkErr(clError, "clCreateBuffer");
	}

	ocl_vertex_GPU = clCreateBuffer(contexts[1], CL_MEM_READ_ONLY, sizeof(float3) * computationSize.x * computationSize.y, NULL, &clError);
	checkErr(clError, "clCreateBuffer");
	ocl_normal_GPU = clCreateBuffer(contexts[1], CL_MEM_READ_ONLY, sizeof(float3) * computationSize.x * computationSize.y, NULL, &clError);
	checkErr(clError, "clCreateBuffer");
	ocl_vertex_FPGA = clCreateBuffer(contexts[0], CL_MEM_READ_ONLY, sizeof(float3) * computationSize.x * computationSize.y, NULL, &clError);
	checkErr(clError, "clCreateBuffer");
	ocl_normal_FPGA = clCreateBuffer(contexts[0], CL_MEM_READ_ONLY, sizeof(float3) * computationSize.x * computationSize.y, NULL, &clError);
	checkErr(clError, "clCreateBuffer");
	ocl_trackingResult_FPGA = clCreateBuffer(contexts[0], CL_MEM_READ_WRITE, sizeof(TrackData) * computationSize.x * computationSize.y, NULL, &clError);
	checkErr(clError, "clCreateBuffer");

	ocl_reduce_output_buffer = clCreateBuffer(contexts[0], CL_MEM_WRITE_ONLY, NUM_THREADS_REDUCE_KERNEL * 32 * sizeof(float), NULL, &clError);
	checkErr(clError, "clCreateBuffer");

	posix_memalign((void **) &reductionoutput, 64, NUM_THREADS_REDUCE_KERNEL * 32 * sizeof(float));
	posix_memalign((void **) &inputVertex, 64, sizeof(float3*) * iterations.size());
	posix_memalign((void **) &inputNormal, 64, sizeof(float3*) * iterations.size());
	for (unsigned int i = 0; i < iterations.size(); ++i) {
		posix_memalign((void **) &inputVertex[i], 64, sizeof(float3) * (computationSize.x * computationSize.y) / (int) pow(2, i));
		for (unsigned int j=0; j<(computationSize.x * computationSize.y) / (int) pow(2, i); j++) {
			inputVertex[i][j].x = 0.0f;
			inputVertex[i][j].y = 0.0f;
			inputVertex[i][j].z = 0.0f;
		}

		posix_memalign((void **) &inputNormal[i], 64, sizeof(float3) * (computationSize.x * computationSize.y) / (int) pow(2, i));
		for (unsigned int j=0; j<(computationSize.x * computationSize.y) / (int) pow(2, i); j++) {
			inputNormal[i][j].x = 0.0f;
			inputNormal[i][j].y = 0.0f;
			inputNormal[i][j].z = 0.0f;
		}
	}

	posix_memalign((void **) &vertex, 64, sizeof(float3) * computationSize.x * computationSize.y);
	for (unsigned int i=0; i<computationSize.x * computationSize.y; i++) {
		vertex[i].x = 0.0f;
		vertex[i].y = 0.0f;
		vertex[i].z = 0.0f;
	}
	posix_memalign((void **) &normal, 64, sizeof(float3) * computationSize.x * computationSize.y);
	for (unsigned int i=0; i<computationSize.x * computationSize.y; i++) {
		normal[i].x = 0.0f;
		normal[i].y = 0.0f;
		normal[i].z = 0.0f;
	}
	posix_memalign((void **) &trackingResult, 64, sizeof(TrackData) * computationSize.x * computationSize.y);
	for (unsigned int i=0; i<computationSize.x * computationSize.y; i++) {
		trackingResult[i].result = 0;
		trackingResult[i].error = 0.0f;
		for (unsigned j=0; j<6; j++) {
			trackingResult[i].J[j] = 0.0f;
		}
	}

	// ********* BEGIN : Generate the gaussian *************
	size_t gaussianS = radius * 2 + 1;
	float *gaussian = (float*) malloc(gaussianS * sizeof(float));
	int x;
	for (unsigned int i = 0; i < gaussianS; i++) {
		x = i - 2;
		gaussian[i] = expf(-(x * x) / (2 * delta * delta));
	}
	ocl_gaussian = clCreateBuffer(contexts[1], CL_MEM_READ_ONLY, gaussianS * sizeof(float), NULL, &clError);
	checkErr(clError, "clCreateBuffer");
	clError = clEnqueueWriteBuffer(cmd_queues[1][0], ocl_gaussian, CL_TRUE, 0, gaussianS * sizeof(float), gaussian, 0, NULL, NULL);
	checkErr(clError, "clEnqueueWrite");
	free(gaussian);
	// ********* END : Generate the gaussian *************

	// Create kernel
	initVolume_ocl_kernel = clCreateKernel(programs[1], "initVolumeKernel", &clError);
	checkErr(clError, "clCreateKernel");

	ocl_volume_data = clCreateBuffer(contexts[1], CL_MEM_READ_WRITE, sizeof(short2) * volumeResolution.x * volumeResolution.y * volumeResolution.z, NULL, &clError);
	checkErr(clError, "clCreateBuffer");
	clError = clSetKernelArg(initVolume_ocl_kernel, 0, sizeof(cl_mem), &ocl_volume_data);
	checkErr(clError, "clSetKernelArg");

	size_t globalWorksize[3] = { volumeResolution.x, volumeResolution.y, volumeResolution.z };
	clError = clEnqueueNDRangeKernel(cmd_queues[1][0], initVolume_ocl_kernel, 3, NULL, globalWorksize, NULL, 0, NULL, NULL);
	checkErr(clError, "clEnqueueNDRangeKernel");

	//Kernels
	mm2meters_ocl_kernel = clCreateKernel(programs[1], "mm2metersKernel", &clError);
	checkErr(clError, "clCreateKernel");
	bilateralFilter_ocl_kernel = clCreateKernel(programs[1], "bilateralFilterKernel", &clError);
	checkErr(clError, "clCreateKernel");
	halfSampleRobustImage_ocl_kernel = clCreateKernel(programs[1], "halfSampleRobustImageKernel", &clError);
	checkErr(clError, "clCreateKernel");
	depth2vertex_ocl_kernel = clCreateKernel(programs[1], "depth2vertexKernel", &clError);
	checkErr(clError, "clCreateKernel");
	vertex2normal_ocl_kernel = clCreateKernel(programs[1], "vertex2normalKernel", &clError);
	checkErr(clError, "clCreateKernel");
	track_ocl_kernel = clCreateKernel(programs[0], "trackKernel", &clError);
	checkErr(clError, "clCreateKernel");
	char kernelName[20];
	for(int i=0; i<3; i++) {
		sprintf(kernelName, "reduceKernel%d", i+1);
		reduce_ocl_kernel[i] = clCreateKernel(programs[0], kernelName, &clError);
		checkErr(clError, "clCreateKernel");
	}
	integrate_ocl_kernel = clCreateKernel(programs[1], "integrateKernel", &clError);
	checkErr(clError, "clCreateKernel");
	raycast_ocl_kernel = clCreateKernel(programs[1], "raycastKernel", &clError);
	checkErr(clError, "clCreateKernel");
	renderVolume_ocl_kernel = clCreateKernel(programs[1], "renderVolumeKernel", &clError);
	checkErr(clError, "clCreateKernel");
	renderDepth_ocl_kernel = clCreateKernel(programs[1], "renderDepthKernel", &clError);
	checkErr(clError, "clCreateKernel");

}
Kfusion::~Kfusion() {
	if (reductionoutput) free(reductionoutput);
	reductionoutput = NULL;

	free(vertex);
	free(normal);
	for (unsigned int i = 0; i < iterations.size(); ++i) {
		free(inputVertex[i]);
		free(inputNormal[i]);
	}
	free(inputVertex);
	free(inputNormal);
	free(trackingResult);

	for (unsigned int i = 0; i < iterations.size(); ++i) {
		if (ocl_ScaledDepth[i]) {
			clError = clReleaseMemObject(ocl_ScaledDepth[i]);
			checkErr(clError, "clReleaseMem");
			ocl_ScaledDepth[i] = NULL;
		}
		if (ocl_inputVertex_GPU[i]) {
			clError = clReleaseMemObject(ocl_inputVertex_GPU[i]);
			checkErr(clError, "clReleaseMem");
			ocl_inputVertex_GPU[i] = NULL;
		}
		if (ocl_inputNormal_GPU[i]) {
			clError = clReleaseMemObject(ocl_inputNormal_GPU[i]);
			checkErr(clError, "clReleaseMem");
			ocl_inputNormal_GPU[i] = NULL;
		}
		if (ocl_inputVertex_FPGA[i]) {
			clError = clReleaseMemObject(ocl_inputVertex_FPGA[i]);
			checkErr(clError, "clReleaseMem");
			ocl_inputVertex_FPGA[i] = NULL;
		}
		if (ocl_inputNormal_FPGA[i]) {
			clError = clReleaseMemObject(ocl_inputNormal_FPGA[i]);
			checkErr(clError, "clReleaseMem");
			ocl_inputNormal_FPGA[i] = NULL;
		}
	}
	if (ocl_ScaledDepth) {
		free(ocl_ScaledDepth);
		ocl_ScaledDepth = NULL;
	}
	if (ocl_inputVertex_GPU) {
		free(ocl_inputVertex_GPU);
		ocl_inputVertex_GPU = NULL;
	}
	if (ocl_inputNormal_GPU) {
		free(ocl_inputNormal_GPU);
		ocl_inputNormal_GPU = NULL;
	}
	if (ocl_inputVertex_FPGA) {
		free(ocl_inputVertex_FPGA);
		ocl_inputVertex_FPGA = NULL;
	}
	if (ocl_inputNormal_FPGA) {
		free(ocl_inputNormal_FPGA);
		ocl_inputNormal_FPGA = NULL;
	}

	if (ocl_FloatDepth) {
		clError = clReleaseMemObject(ocl_FloatDepth);
		checkErr(clError, "clReleaseMem");
		ocl_FloatDepth = NULL;
	}
	if (ocl_vertex_GPU) {
		clError = clReleaseMemObject(ocl_vertex_GPU);
		checkErr(clError, "clReleaseMem");
		ocl_vertex_GPU = NULL;
	}
	if (ocl_normal_GPU) {
	  	clError = clReleaseMemObject(ocl_normal_GPU);
	  	checkErr(clError, "clReleaseMem");
		ocl_normal_GPU = NULL;
	}
	if (ocl_vertex_FPGA) {
		clError = clReleaseMemObject(ocl_vertex_FPGA);
		checkErr(clError, "clReleaseMem");
		ocl_vertex_FPGA = NULL;
	}
	if (ocl_normal_FPGA) {
	  	clError = clReleaseMemObject(ocl_normal_FPGA);
	  	checkErr(clError, "clReleaseMem");
		ocl_normal_FPGA = NULL;
	}
	if (ocl_trackingResult_FPGA) {
	 	clError = clReleaseMemObject(ocl_trackingResult_FPGA);
		checkErr(clError, "clReleaseMem");
		ocl_trackingResult_FPGA = NULL;
	}
	if (ocl_gaussian) {
	 	clError = clReleaseMemObject(ocl_gaussian);
		checkErr(clError, "clReleaseMem");
		ocl_gaussian = NULL;
	}
	if (ocl_volume_data) {
	 	clError = clReleaseMemObject(ocl_volume_data);
		checkErr(clError, "clReleaseMem");
		ocl_volume_data = NULL;
	}
	if (ocl_depth_buffer) {
	 	clError = clReleaseMemObject(ocl_depth_buffer);
		checkErr(clError, "clReleaseMem");
		ocl_depth_buffer = NULL;
	}
	if(ocl_output_render_buffer) {
	    clError = clReleaseMemObject(ocl_output_render_buffer);
	    checkErr(clError, "clReleaseMem");
		ocl_output_render_buffer = NULL;
	}
	if (ocl_reduce_output_buffer) {
		clError = clReleaseMemObject(ocl_reduce_output_buffer);
		checkErr(clError, "clReleaseMem");
		ocl_reduce_output_buffer = NULL;
	}
	RELEASE_KERNEL(mm2meters_ocl_kernel);
	RELEASE_KERNEL(bilateralFilter_ocl_kernel);
	RELEASE_KERNEL(halfSampleRobustImage_ocl_kernel);
	RELEASE_KERNEL(depth2vertex_ocl_kernel);
	RELEASE_KERNEL(vertex2normal_ocl_kernel);
	RELEASE_KERNEL(track_ocl_kernel);
	for(int i=0; i<3; i++) {
		RELEASE_KERNEL(reduce_ocl_kernel[i]);
		reduce_ocl_kernel[i] = NULL;
	}
	RELEASE_KERNEL(integrate_ocl_kernel);
	RELEASE_KERNEL(raycast_ocl_kernel);
	RELEASE_KERNEL(renderVolume_ocl_kernel);
	RELEASE_KERNEL(renderDepth_ocl_kernel);
	RELEASE_KERNEL(initVolume_ocl_kernel);

	mm2meters_ocl_kernel = NULL ;
	bilateralFilter_ocl_kernel = NULL;
	halfSampleRobustImage_ocl_kernel = NULL;
	depth2vertex_ocl_kernel = NULL;
	vertex2normal_ocl_kernel = NULL;
	track_ocl_kernel = NULL;
	integrate_ocl_kernel = NULL;
	raycast_ocl_kernel = NULL;
	renderVolume_ocl_kernel = NULL;
	renderLight_ocl_kernel = NULL;
	renderDepth_ocl_kernel = NULL;

	computationSizeBkp = make_uint2(0, 0);
	outputImageSizeBkp = make_uint2(0, 0);

	clean();
}

void Kfusion::reset() {
	std::cerr << "Reset function to clear volume model needs to be implemented\n";
	exit(1);
}

bool Kfusion::preprocessing(const uint16_t * inputDepth, const uint2 inSize) {
	startOfTiming = benchmark_tock();

	uint2 outSize = computationSize;

	// Check for unsupported conditions
	if ((inSize.x < outSize.x) || (inSize.y < outSize.y)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}
	if ((inSize.x % outSize.x != 0) || (inSize.y % outSize.y != 0)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}
	if ((inSize.x / outSize.x != inSize.y / outSize.y)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}

	int ratio = inSize.x / outSize.x;

	if (computationSizeBkp.x < inSize.x|| computationSizeBkp.y < inSize.y || ocl_depth_buffer == NULL) {
		computationSizeBkp = make_uint2(inSize.x, inSize.y);
		if (ocl_depth_buffer != NULL) {
			clError = clReleaseMemObject(ocl_depth_buffer);
			checkErr(clError, "clReleaseMemObject");
		}
		ocl_depth_buffer = clCreateBuffer(contexts[1], CL_MEM_READ_WRITE, inSize.x * inSize.y * sizeof(uint16_t), NULL, &clError);
		checkErr(clError, "clCreateBuffer input");
	}

	clError = clEnqueueWriteBuffer(cmd_queues[1][0], ocl_depth_buffer, CL_FALSE, 0, inSize.x * inSize.y * sizeof(uint16_t), inputDepth, 0, NULL, NULL);
	checkErr(clError, "clEnqueueWriteBuffer");

	int arg = 0;
	char errStr[20];

	clError = clSetKernelArg(mm2meters_ocl_kernel, arg++, sizeof(cl_mem), &ocl_FloatDepth);
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);
	clError = clSetKernelArg(mm2meters_ocl_kernel, arg++, sizeof(cl_uint2), &outSize);
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);
	clError = clSetKernelArg(mm2meters_ocl_kernel, arg++, sizeof(cl_mem), &ocl_depth_buffer);
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);
	clError = clSetKernelArg(mm2meters_ocl_kernel, arg++, sizeof(cl_uint2), &inSize);
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);
	clError = clSetKernelArg(mm2meters_ocl_kernel, arg++, sizeof(cl_int), &ratio);
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);

	size_t globalWorksize[2] = { outSize.x, outSize.y };

	clError = clEnqueueNDRangeKernel(cmd_queues[1][0], mm2meters_ocl_kernel, 2, NULL, globalWorksize, NULL, 0, NULL, NULL);
	checkErr(clError, "clEnqueueNDRangeKernel");

	endOfTiming = benchmark_tock();
	timingsCPU[1] = endOfTiming - startOfTiming;
	*logstreamCustom << "mm2meters:" << (timingsCPU[1]) << "\t";
	startOfTiming = endOfTiming;

	arg = 0;

	clError = clSetKernelArg(bilateralFilter_ocl_kernel, arg++, sizeof(cl_mem), &ocl_ScaledDepth[0]);
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);
	clError = clSetKernelArg(bilateralFilter_ocl_kernel, arg++, sizeof(cl_mem), &ocl_FloatDepth);
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);
	clError = clSetKernelArg(bilateralFilter_ocl_kernel, arg++, sizeof(cl_mem), &ocl_gaussian);
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);
	clError = clSetKernelArg(bilateralFilter_ocl_kernel, arg++, sizeof(cl_float), &e_delta);
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);
	clError = clSetKernelArg(bilateralFilter_ocl_kernel, arg++, sizeof(cl_int), &radius);
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);

	clError = clEnqueueNDRangeKernel(cmd_queues[1][0], bilateralFilter_ocl_kernel, 2, NULL, globalWorksize, NULL, 0, NULL, NULL);
	checkErr(clError, "clEnqueueNDRangeKernel");

	endOfTiming = benchmark_tock();
	timingsCPU[2] = endOfTiming - startOfTiming;
	*logstreamCustom << "bilateralFilter:" << (timingsCPU[2]) << "\t";

	return true;

}
bool Kfusion::tracking(float4 k, float icp_threshold, uint tracking_rate, uint frame) {
	startOfTiming = benchmark_tock();

	if ((frame % tracking_rate) != 0)
		return false;

	// half sample the input depth maps into the pyramid levels
	for (unsigned int i = 1; i < iterations.size(); ++i) {
		// outSize = computationSize / 2
		// outSize = computationSize / 4
		// outSize = computationSize / 8
		uint2 outSize = make_uint2(computationSize.x / (int) pow(2, i), computationSize.y / (int) pow(2, i));

		float e_d = e_delta * 3;
		int r = 1;
		uint2 inSize = outSize * 2;

		int arg = 0;
		char errStr[20];

		clError = clSetKernelArg(halfSampleRobustImage_ocl_kernel, arg++, sizeof(cl_mem), &ocl_ScaledDepth[i]);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(halfSampleRobustImage_ocl_kernel, arg++, sizeof(cl_mem), &ocl_ScaledDepth[i - 1]);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(halfSampleRobustImage_ocl_kernel, arg++, sizeof(cl_uint2), &inSize);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(halfSampleRobustImage_ocl_kernel, arg++, sizeof(cl_float), &e_d);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(halfSampleRobustImage_ocl_kernel, arg++, sizeof(cl_int), &r);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);

		size_t globalWorksize[2] = { outSize.x, outSize.y };

		clError = clEnqueueNDRangeKernel(cmd_queues[1][0],
				halfSampleRobustImage_ocl_kernel, 2, NULL, globalWorksize, NULL,
				0,
				NULL, NULL);
		checkErr(clError, "clEnqueueNDRangeKernel");
	}

	endOfTiming = benchmark_tock();
	timingsCPU[3] = endOfTiming - startOfTiming;
	*logstreamCustom << "halfSample:" << (timingsCPU[3]) << "\t";

	// prepare the 3D information from the input depth maps
	uint2 localimagesize = computationSize;

	timingsCPU[4] = 0.0f;
	timingsCPU[5] = 0.0f;
	for (unsigned int i = 0; i < iterations.size(); ++i) {
		startOfTiming = endOfTiming;

		Matrix4 invK = getInverseCameraMatrix(k / float(1 << i));

		uint2 imageSize = localimagesize;
		// Create kernel

		int arg = 0;
		char errStr[20];
		clError = clSetKernelArg(depth2vertex_ocl_kernel, arg++, sizeof(cl_mem), &ocl_inputVertex_GPU[i]);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(depth2vertex_ocl_kernel, arg++, sizeof(cl_uint2), &imageSize);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(depth2vertex_ocl_kernel, arg++, sizeof(cl_mem), &ocl_ScaledDepth[i]);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(depth2vertex_ocl_kernel, arg++, sizeof(cl_uint2), &imageSize);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(depth2vertex_ocl_kernel, arg++, sizeof(Matrix4), &invK);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		size_t globalWorksize[2] = { imageSize.x, imageSize.y };

		clError = clEnqueueNDRangeKernel(cmd_queues[1][0], depth2vertex_ocl_kernel, 2, NULL, globalWorksize, NULL, 0, NULL, NULL);
		checkErr(clError, "clEnqueueNDRangeKernel");

		endOfTiming = benchmark_tock();
		timingsCPU[4] += endOfTiming - startOfTiming;

		startOfTiming = endOfTiming;

		arg = 0;
		clError = clSetKernelArg(vertex2normal_ocl_kernel, arg++, sizeof(cl_mem), &ocl_inputNormal_GPU[i]);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(vertex2normal_ocl_kernel, arg++, sizeof(cl_uint2), &imageSize);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(vertex2normal_ocl_kernel, arg++, sizeof(cl_mem), &ocl_inputVertex_GPU[i]);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(vertex2normal_ocl_kernel, arg++, sizeof(cl_uint2), &imageSize);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);

		size_t globalWorksize2[2] = { imageSize.x, imageSize.y };

		clError = clEnqueueNDRangeKernel(cmd_queues[1][0], vertex2normal_ocl_kernel, 2, NULL, globalWorksize2, NULL, 0, NULL, NULL);
		checkErr(clError, "clEnqueueNDRangeKernel");

		localimagesize = make_uint2(localimagesize.x / 2, localimagesize.y / 2);

		endOfTiming = benchmark_tock();
		timingsCPU[5] += endOfTiming - startOfTiming;
	}

	*logstreamCustom << "depth2vertex:" << (timingsCPU[4]) << "\t";
	*logstreamCustom << "vertex2normal:" << (timingsCPU[5]) << "\t";

	
	timingsIO[6] = 0.0f;
	timingsCPU[6] = 0.0f;
	timingsIO[7] = 0.0f;
	timingsCPU[7] = 0.0f;

	startOfTiming = benchmark_tock();
	oldPose = pose;
	const Matrix4 projectReference = getCameraMatrix(k) * inverse(raycastPose);

	bool checkPoseKernelRes, updatePoseKernelRes;

	clError = clEnqueueReadBuffer(cmd_queues[1][0], ocl_vertex_GPU, CL_TRUE, 0, sizeof(float3) * (computationSize.x * computationSize.y), vertex, 0, NULL, NULL);
    clError = clEnqueueReadBuffer(cmd_queues[1][0], ocl_normal_GPU, CL_TRUE, 0, sizeof(float3) * (computationSize.x * computationSize.y), normal, 0, NULL, NULL);

	clError = clEnqueueWriteBuffer(cmd_queues[0][0], ocl_vertex_FPGA, CL_TRUE, 0, sizeof(float3) * (computationSize.x * computationSize.y), vertex, 0, NULL, NULL);
    clError = clEnqueueWriteBuffer(cmd_queues[0][0], ocl_normal_FPGA, CL_TRUE, 0, sizeof(float3) * (computationSize.x * computationSize.y), normal, 0, NULL, NULL);

    endOfTiming = benchmark_tock();
    timingsIO[6] += endOfTiming - startOfTiming;
    
    startOfTiming = endOfTiming;
	for (int level = iterations.size() - 1; level >= 0; --level) {
		uint2 localimagesize = make_uint2(
				computationSize.x / (int) pow(2, level),
				computationSize.y / (int) pow(2, level));

		clError = clEnqueueReadBuffer(cmd_queues[1][0], ocl_inputVertex_GPU[level], CL_TRUE, 0, sizeof(float3) * (computationSize.x * computationSize.y) / (int) pow(2, level), inputVertex[level], 0, NULL, NULL);
		clError = clEnqueueReadBuffer(cmd_queues[1][0], ocl_inputNormal_GPU[level], CL_TRUE, 0, sizeof(float3) * (computationSize.x * computationSize.y) / (int) pow(2, level), inputNormal[level], 0, NULL, NULL);

		clError = clEnqueueWriteBuffer(cmd_queues[0][0], ocl_inputVertex_FPGA[level], CL_TRUE, 0, sizeof(float3) * (computationSize.x * computationSize.y) / (int) pow(2, level), inputVertex[level], 0, NULL, NULL);
        clError = clEnqueueWriteBuffer(cmd_queues[0][0], ocl_inputNormal_FPGA[level], CL_TRUE, 0, sizeof(float3) * (computationSize.x * computationSize.y) / (int) pow(2, level), inputNormal[level], 0, NULL, NULL);
        endOfTiming = benchmark_tock();
        timingsIO[6] += endOfTiming - startOfTiming;

        startOfTiming = endOfTiming;
		for (int i = 0; i < iterations[level]; ++i) {
            int arg = 0;
            char errStr[20];
            clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_mem), &ocl_trackingResult_FPGA);
            sprintf(errStr, "clSetKernelArg%d", arg);
            checkErr(clError, errStr);
            clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_uint2), &localimagesize);
            sprintf(errStr, "clSetKernelArg%d", arg);
            checkErr(clError, errStr);
            clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_mem), &ocl_inputVertex_FPGA[level]);
            sprintf(errStr, "clSetKernelArg%d", arg);
            checkErr(clError, errStr);
            clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_uint2), &localimagesize);
            sprintf(errStr, "clSetKernelArg%d", arg);
            checkErr(clError, errStr);
            clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_mem), &ocl_inputNormal_FPGA[level]);
            sprintf(errStr, "clSetKernelArg%d", arg);
            checkErr(clError, errStr);
            clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_uint2), &localimagesize);
            sprintf(errStr, "clSetKernelArg%d", arg);
            checkErr(clError, errStr);
            clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_mem), &ocl_vertex_FPGA);
            sprintf(errStr, "clSetKernelArg%d", arg);
            checkErr(clError, errStr);
            clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_uint2), &computationSize);
            sprintf(errStr, "clSetKernelArg%d", arg);
            checkErr(clError, errStr);
            clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_mem), &ocl_normal_FPGA);
            sprintf(errStr, "clSetKernelArg%d", arg);
            checkErr(clError, errStr);
            clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_uint2), &computationSize);
            sprintf(errStr, "clSetKernelArg%d", arg);
            checkErr(clError, errStr);
            clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_float4), &(pose.data[0]));
            sprintf(errStr, "clSetKernelArg%d", arg);
            checkErr(clError, errStr);
            clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_float4), &(pose.data[1]));
            sprintf(errStr, "clSetKernelArg%d", arg);
            checkErr(clError, errStr);
            clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_float4), &(pose.data[2]));
            sprintf(errStr, "clSetKernelArg%d", arg);
            checkErr(clError, errStr);
            clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_float4), &(pose.data[3]));
            sprintf(errStr, "clSetKernelArg%d", arg);
            checkErr(clError, errStr);
            clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_float4), &(projectReference.data[0]));
            sprintf(errStr, "clSetKernelArg%d", arg);
            checkErr(clError, errStr);
            clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_float4), &(projectReference.data[1]));
            sprintf(errStr, "clSetKernelArg%d", arg);
            checkErr(clError, errStr);
            clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_float4), &(projectReference.data[2]));
            sprintf(errStr, "clSetKernelArg%d", arg);
            checkErr(clError, errStr);
            clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_float4), &(projectReference.data[3]));
            sprintf(errStr, "clSetKernelArg%d", arg);
            checkErr(clError, errStr);
            clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_float), &dist_threshold);
            sprintf(errStr, "clSetKernelArg%d", arg);
            checkErr(clError, errStr);
            clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_float), &normal_threshold);
            sprintf(errStr, "clSetKernelArg%d", arg);
            checkErr(clError, errStr);
			
            size_t globalWorksize[2] = { localimagesize.x, localimagesize.y };
            clError = clEnqueueNDRangeKernel(cmd_queues[0][0], track_ocl_kernel, 2, NULL, globalWorksize, NULL, 0, NULL, NULL);
            checkErr(clError, "clEnqueueNDRangeKernel");

            endOfTiming = benchmark_tock();
        	timingsCPU[6] += endOfTiming - startOfTiming;
            startOfTiming = endOfTiming;

			arg = 0;
			clError = clSetKernelArg(reduce_ocl_kernel[level], arg++, sizeof(cl_mem), &ocl_reduce_output_buffer);
			sprintf(errStr, "clSetKernelArg%d", arg);
			checkErr(clError, errStr);
			clError = clSetKernelArg(reduce_ocl_kernel[level], arg++, sizeof(cl_mem), &ocl_trackingResult_FPGA);
			sprintf(errStr, "clSetKernelArg%d", arg);
			checkErr(clError, errStr);

			size_t RglobalWorksize[1] = { NUM_THREADS_REDUCE_KERNEL };

			clError = clEnqueueNDRangeKernel(cmd_queues[0][0], reduce_ocl_kernel[level], 1, NULL, RglobalWorksize, NULL, 0, NULL, NULL);
            checkErr(clError, "clEnqueueNDRangeKernel");

            endOfTiming = benchmark_tock();
			timingsCPU[7] += endOfTiming - startOfTiming;
			startOfTiming = endOfTiming;

            clError = clEnqueueReadBuffer(cmd_queues[0][0], ocl_reduce_output_buffer, CL_TRUE, 0, NUM_THREADS_REDUCE_KERNEL * 32 * sizeof(float), reductionoutput, 0, NULL, NULL);
			checkErr(clError, "clEnqueueReadBuffer");

			endOfTiming = benchmark_tock();
			timingsIO[7] += endOfTiming - startOfTiming;
			startOfTiming = endOfTiming;

			TooN::Matrix<TooN::Dynamic, TooN::Dynamic, float, TooN::Reference::RowMajor> values(reductionoutput, NUM_THREADS_REDUCE_KERNEL, 32);
			for (int j = 1; j < NUM_THREADS_REDUCE_KERNEL; ++j) {
				values[0] += values[j];
			}

			endOfTiming = benchmark_tock();
			timingsCPU[7] += endOfTiming - startOfTiming;
			startOfTiming = endOfTiming;

			updatePoseKernelRes = updatePoseKernel(pose, reductionoutput, icp_threshold);

			endOfTiming = benchmark_tock();
			timingsCPU[6] += endOfTiming - startOfTiming;
			startOfTiming = endOfTiming;

			if (updatePoseKernelRes) break;

		}
	}
	checkPoseKernelRes = checkPoseKernel(pose, oldPose, reductionoutput, computationSize, track_threshold);
	endOfTiming = benchmark_tock();
	timingsCPU[6] += endOfTiming - startOfTiming;

	startOfTiming = endOfTiming;
	clError = clEnqueueReadBuffer(cmd_queues[0][0], ocl_trackingResult_FPGA, CL_TRUE, 0, sizeof(TrackData) * (computationSize.x * computationSize.y), trackingResult, 0, NULL, NULL);
	checkErr(clError, "clEnqueueReadBuffer");
	timingsIO[6] += endOfTiming - startOfTiming;

	*logstreamCustom << "track_IO:" << (timingsIO[6]) << std::endl;
	*logstreamCustom << "track_Kernel:" << (timingsCPU[6]) << std::endl;
	*logstreamCustom << "reduce_IO:" << (timingsIO[7]) << std::endl;
	*logstreamCustom << "reduce_Kernel:" << (timingsCPU[7]) << std::endl;

	return checkPoseKernelRes;
}

bool updatePoseKernel(Matrix4 & pose, const float * output, float icp_threshold) {
	bool res = false;
	// Update the pose regarding the tracking result
	TooN::Matrix<NUM_THREADS_REDUCE_KERNEL, 32, const float, TooN::Reference::RowMajor> values(output);
	TooN::Vector<6> x = solve(values[0].slice<1, 27>());
	TooN::SE3<> delta(x);
	pose = toMatrix4(delta) * pose;

	// Return validity test result of the tracking
	if (norm(x) < icp_threshold)
		res = true;

	return res;
}

bool checkPoseKernel(Matrix4 & pose, Matrix4 oldPose, const float * output, uint2 imageSize, float track_threshold) {

	// Check the tracking result, and go back to the previous camera position if necessary

	TooN::Matrix<NUM_THREADS_REDUCE_KERNEL, 32, const float, TooN::Reference::RowMajor> values(output);

	if ((std::sqrt(values(0, 0) / values(0, 28)) > 2e-2)
			|| (values(0, 28) / (imageSize.x * imageSize.y) < track_threshold)) {
		pose = oldPose;
		return false;
	} else {
		return true;
	}

}
bool Kfusion::integration(float4 k, uint integration_rate, float mu, uint frame) {
	startOfTiming = benchmark_tock();

	bool doIntegrate = checkPoseKernel(pose, oldPose, reductionoutput, computationSize, track_threshold);

	if ((doIntegrate && ((frame % integration_rate) == 0)) || (frame <= 3)) {
		doIntegrate = true;
		// integrate(integration, ScaledDepth[0],inputSize, inverse(pose), getCameraMatrix(k), mu, maxweight );

		uint2 depthSize = computationSize;
		const Matrix4 invTrack = inverse(pose);
		const Matrix4 K = getCameraMatrix(k);

		//uint3 pix = make_uint3(thr2pos2());
		const float3 delta = rotate(invTrack, make_float3(0, 0, volumeDimensions.z / volumeResolution.z));
		const float3 cameraDelta = rotate(K, delta);

		int arg = 0;
		char errStr[20];

		clError = clSetKernelArg(integrate_ocl_kernel, arg++, sizeof(cl_mem), (void*) &ocl_volume_data);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(integrate_ocl_kernel, arg++, sizeof(cl_uint3), (void*) &volumeResolution);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(integrate_ocl_kernel, arg++, sizeof(cl_float3), (void*) &volumeDimensions);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(integrate_ocl_kernel, arg++, sizeof(cl_mem), (void*) &ocl_FloatDepth);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(integrate_ocl_kernel, arg++, sizeof(cl_uint2), (void*) &depthSize);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(integrate_ocl_kernel, arg++, sizeof(Matrix4), (void*) &invTrack);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(integrate_ocl_kernel, arg++, sizeof(Matrix4), (void*) &K);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(integrate_ocl_kernel, arg++, sizeof(cl_float), (void*) &mu);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(integrate_ocl_kernel, arg++, sizeof(cl_float), (void*) &maxweight);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);

		clError = clSetKernelArg(integrate_ocl_kernel, arg++, sizeof(cl_float3), (void*) &delta);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(integrate_ocl_kernel, arg++, sizeof(cl_float3), (void*) &cameraDelta);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);

		size_t globalWorksize[2] = { volumeResolution.x, volumeResolution.y };

		clError = clEnqueueNDRangeKernel(cmd_queues[1][0], integrate_ocl_kernel, 2, NULL, globalWorksize, NULL, 0, NULL, NULL);
	} else {
		doIntegrate = false;
	}

	endOfTiming = benchmark_tock();
	timingsCPU[8] = endOfTiming - startOfTiming;
	*logstreamCustom << "integrate:" << (timingsCPU[8]) << "\t";

	return doIntegrate;
}

bool Kfusion::raycasting(float4 k, float mu, uint frame) {
	startOfTiming = benchmark_tock();

	bool doRaycast = false;
	float largestep = mu * 0.75f;

	if (frame > 2) {
		checkErr(clError, "clEnqueueNDRangeKernel");
		raycastPose = pose;
		const Matrix4 view = raycastPose * getInverseCameraMatrix(k);

		int arg = 0;
		char errStr[20];

		clError = clSetKernelArg(raycast_ocl_kernel, arg++, sizeof(cl_mem), (void*) &ocl_vertex_GPU);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(raycast_ocl_kernel, arg++, sizeof(cl_mem), (void*) &ocl_normal_GPU);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(raycast_ocl_kernel, arg++, sizeof(cl_mem), (void*) &ocl_volume_data);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(raycast_ocl_kernel, arg++, sizeof(cl_uint3), (void*) &volumeResolution);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(raycast_ocl_kernel, arg++, sizeof(cl_float3), (void*) &volumeDimensions);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(raycast_ocl_kernel, arg++, sizeof(Matrix4), (void*) &view);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(raycast_ocl_kernel, arg++, sizeof(cl_float), (void*) &nearPlane);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(raycast_ocl_kernel, arg++, sizeof(cl_float), (void*) &farPlane);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(raycast_ocl_kernel, arg++, sizeof(cl_float), (void*) &step);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(raycast_ocl_kernel, arg++, sizeof(cl_float), (void*) &largestep);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);

		size_t RaycastglobalWorksize[2] = { computationSize.x, computationSize.y };

		clError = clEnqueueNDRangeKernel(cmd_queues[1][0], raycast_ocl_kernel, 2, NULL, RaycastglobalWorksize, NULL, 0, NULL, NULL);
		checkErr(clError, "clEnqueueNDRangeKernel");

	}

	endOfTiming = benchmark_tock();
	timingsCPU[9] = endOfTiming - startOfTiming;
	*logstreamCustom << "raycast:" << (timingsCPU[9]) << "\t";

	return doRaycast;
}

void Kfusion::renderDepth(uchar4 * out, uint2 outputSize) {
	startOfTiming = benchmark_tock();

    // Create render opencl buffer if needed
    if(outputImageSizeBkp.x < outputSize.x || outputImageSizeBkp.y < outputSize.y || ocl_output_render_buffer == NULL) 
    {
		outputImageSizeBkp = make_uint2(outputSize.x, outputSize.y);
		if(ocl_output_render_buffer != NULL){
		    std::cout << "Release" << std::endl;
		    clError = clReleaseMemObject(ocl_output_render_buffer);
		    checkErr( clError, "clReleaseMemObject");
		}
		ocl_output_render_buffer = clCreateBuffer(contexts[1], CL_MEM_WRITE_ONLY, outputSize.x * outputSize.y * sizeof(uchar4), NULL , &clError);
		checkErr(clError, "clCreateBuffer output" );
    }

	clError = clSetKernelArg(renderDepth_ocl_kernel, 0, sizeof(cl_mem), &ocl_output_render_buffer);
	clError &= clSetKernelArg(renderDepth_ocl_kernel, 1, sizeof(cl_mem), &ocl_FloatDepth);
	clError &= clSetKernelArg(renderDepth_ocl_kernel, 2, sizeof(cl_float), &nearPlane);
	clError &= clSetKernelArg(renderDepth_ocl_kernel, 3, sizeof(cl_float), &farPlane);
	checkErr(clError, "clSetKernelArg");

	size_t globalWorksize[2] = { computationSize.x, computationSize.y };

	clError = clEnqueueNDRangeKernel(cmd_queues[1][0], renderDepth_ocl_kernel, 2,
			NULL, globalWorksize, NULL, 0, NULL, NULL);
	checkErr(clError, "clEnqueueNDRangeKernel");


    clError = clEnqueueReadBuffer(cmd_queues[1][0], ocl_output_render_buffer, CL_FALSE, 0, outputSize.x * outputSize.y * sizeof(uchar4), out, 0, NULL, NULL );  
    checkErr( clError, "clEnqueueReadBuffer");

    endOfTiming = benchmark_tock();
	timingsCPU[10] = endOfTiming - startOfTiming;
	*logstreamCustom << "renderDepth:" << (timingsCPU[10]) << "\t";
}

void Kfusion::renderTrack(uchar4 * out, uint2 outputSize) {
	startOfTiming = benchmark_tock();

    unsigned int y;
	#pragma omp parallel for shared(out), private(y)
		for (y = 0; y < computationSize.y; y++)
			for (unsigned int x = 0; x < computationSize.x; x++) {
				uint pos = x + y * computationSize.x;
				switch (trackingResult[pos].result) {
				case 1:
					out[pos] = make_uchar4(128, 128, 128, 0);  // ok	 GREY
					break;
				case -1:
					out[pos] = make_uchar4(0, 0, 0, 0);      // no input BLACK
					break;
				case -2:
					out[pos] = make_uchar4(255, 0, 0, 0);        // not in image RED
					break;
				case -3:
					out[pos] = make_uchar4(0, 255, 0, 0);    // no correspondence GREEN
					break;
				case -4:
					out[pos] = make_uchar4(0, 0, 255, 0);        // to far away BLUE
					break;
				case -5:
					out[pos] = make_uchar4(255, 255, 0, 0);     // wrong normal YELLOW
					break;
				default:
					out[pos] = make_uchar4(255, 128, 128, 0);
					break;
				}
	}

    endOfTiming = benchmark_tock();
	timingsCPU[11] = endOfTiming - startOfTiming;
	*logstreamCustom << "renderTrack:" << (timingsCPU[11]) << "\t";
}

void Kfusion::renderVolume(uchar4 * out, uint2 outputSize, int frame, int rate, float4 k, float largestep) {
	startOfTiming = benchmark_tock();

    if (frame % rate != 0) return;
    // Create render opencl buffer if needed
    if(outputImageSizeBkp.x < outputSize.x || outputImageSizeBkp.y < outputSize.y || ocl_output_render_buffer == NULL) {
		outputImageSizeBkp = make_uint2(outputSize.x, outputSize.y);
		if(ocl_output_render_buffer != NULL) {
		    std::cout << "Release" << std::endl;
		    clError = clReleaseMemObject(ocl_output_render_buffer);
		    checkErr(clError, "clReleaseMemObject");
		}
		ocl_output_render_buffer = clCreateBuffer(contexts[1],  CL_MEM_WRITE_ONLY, outputSize.x * outputSize.y * sizeof(uchar4), NULL , &clError);
		checkErr(clError, "clCreateBuffer output" );
    }

	Matrix4 view = *viewPose * getInverseCameraMatrix(k);

	int arg = 0;
	char errStr[20];

    clError = clSetKernelArg(renderVolume_ocl_kernel, arg++, sizeof(cl_mem), (void*) &ocl_output_render_buffer);
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);
	clError = clSetKernelArg(renderVolume_ocl_kernel, arg++, sizeof(cl_mem), (void*) &ocl_volume_data);
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);
	clError = clSetKernelArg(renderVolume_ocl_kernel, arg++, sizeof(cl_uint3), (void*) &volumeResolution);
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);
	clError = clSetKernelArg(renderVolume_ocl_kernel, arg++, sizeof(cl_float3), (void*) &volumeDimensions);
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);
	clError = clSetKernelArg(renderVolume_ocl_kernel, arg++, sizeof(Matrix4), (void*) &view);
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);
	clError = clSetKernelArg(renderVolume_ocl_kernel, arg++, sizeof(cl_float), (void*) &nearPlane);
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);
	clError = clSetKernelArg(renderVolume_ocl_kernel, arg++, sizeof(cl_float), (void*) &farPlane);
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);
	clError = clSetKernelArg(renderVolume_ocl_kernel, arg++, sizeof(cl_float), (void*) &step);
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);
	clError = clSetKernelArg(renderVolume_ocl_kernel, arg++, sizeof(cl_float), (void*) &largestep);
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);
	clError = clSetKernelArg(renderVolume_ocl_kernel, arg++, sizeof(cl_float3), (void*) &light);
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);
	clError = clSetKernelArg(renderVolume_ocl_kernel, arg++, sizeof(cl_float3), (void*) &ambient);
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);

	size_t globalWorksize[2] = { computationSize.x, computationSize.y };

	clError = clEnqueueNDRangeKernel(cmd_queues[1][0], renderVolume_ocl_kernel, 2, NULL, globalWorksize, NULL, 0, NULL, NULL);
	checkErr(clError, "clEnqueueNDRangeKernel");

    clError = clEnqueueReadBuffer(cmd_queues[1][0], ocl_output_render_buffer, CL_FALSE, 0, outputSize.x * outputSize.y * sizeof(uchar4), out, 0, NULL, NULL );  
    checkErr(clError, "clEnqueueReadBuffer");

    endOfTiming = benchmark_tock();
	timingsCPU[12] = endOfTiming - startOfTiming;
	*logstreamCustom << "renderVolume:" << (timingsCPU[12]) << "\n";
}

void Kfusion::dumpVolume(const char* filename) {

	std::ofstream fDumpFile;

	if (filename == NULL) {
		return;
	}

	fDumpFile.open(filename, std::ios::out | std::ios::binary);
	if (fDumpFile.fail()) {
		std::cout << "Error opening file: " << filename << std::endl;
		exit(1);
	}
	short2 * volume_data = (short2*) malloc(
			volumeResolution.x * volumeResolution.y * volumeResolution.z
					* sizeof(short2));
	clEnqueueReadBuffer(cmd_queues[1][0], ocl_volume_data, CL_TRUE, 0,
			volumeResolution.x * volumeResolution.y * volumeResolution.z
					* sizeof(short2), volume_data, 0, NULL, NULL);

	std::cout << "Dumping the volumetric representation on file: " << filename
			<< std::endl;

	// Dump on file without the y component of the short2 variable
	for (unsigned int i = 0;
			i < volumeResolution.x * volumeResolution.y * volumeResolution.z;
			i++) {
		fDumpFile.write((char *) (volume_data + i), sizeof(short));
	}

	fDumpFile.close();
	free(volume_data);

}

void Kfusion::computeFrame(const ushort * inputDepth, const uint2 inputSize, float4 k, uint integration_rate, uint tracking_rate, float icp_threshold, float mu, const uint frame) {
	preprocessing(inputDepth, inputSize);
	_tracked = tracking(k, icp_threshold, tracking_rate, frame);
	_integrated = integration(k, integration_rate, mu, frame);
	raycasting(k, mu, frame);
}

void synchroniseDevices() {
	clFinish(cmd_queues[1][0]);
}
