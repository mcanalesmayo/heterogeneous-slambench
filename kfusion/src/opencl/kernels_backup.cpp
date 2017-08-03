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

////// USE BY KFUSION CLASS ///////////////

// input once
cl_mem ocl_gaussian = NULL;

// inter-frame
Matrix4 oldPose;
Matrix4 raycastPose;
cl_mem ocl_vertex = NULL;
cl_mem ocl_normal = NULL;
cl_mem ocl_volume_data = NULL;
cl_mem ocl_depth_buffer = NULL;
cl_mem ocl_output_render_buffer = NULL; // Common buffer for rendering track, depth and volume

// intra-frame
cl_mem ocl_reduce_output_buffer = NULL;
cl_mem ocl_trackingResult = NULL;
cl_mem ocl_FloatDepth = NULL;
cl_mem * ocl_ScaledDepth = NULL;
cl_mem * ocl_inputVertex = NULL;
cl_mem * ocl_inputNormal = NULL;
float * reduceOutputBuffer = NULL;

// kernels
cl_kernel mm2meters_ocl_kernel;
cl_kernel bilateralFilter_ocl_kernel;
cl_kernel halfSampleRobustImage_ocl_kernel;
cl_kernel depth2vertex_ocl_kernel;
cl_kernel vertex2normal_ocl_kernel;
cl_kernel track_ocl_kernel;
cl_kernel reduce_ocl_kernel;
cl_kernel integrate_ocl_kernel;
cl_kernel raycast_ocl_kernel;
cl_kernel renderVolume_ocl_kernel;
cl_kernel renderLight_ocl_kernel;
cl_kernel renderTrack_ocl_kernel;
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
	ocl_inputVertex = (cl_mem*) malloc(sizeof(cl_mem) * iterations.size());
	ocl_inputNormal = (cl_mem*) malloc(sizeof(cl_mem) * iterations.size());

	for (unsigned int i = 0; i < iterations.size(); ++i) {
		ocl_ScaledDepth[i] = clCreateBuffer(contexts[1], CL_MEM_READ_WRITE, sizeof(float) * (computationSize.x * computationSize.y) / (int) pow(2, i), NULL, &clError);
		checkErr(clError, "clCreateBuffer");
		ocl_inputVertex[i] = clCreateBuffer(contexts[1], CL_MEM_READ_WRITE, sizeof(float3) * (computationSize.x * computationSize.y) / (int) pow(2, i), NULL, &clError);
		checkErr(clError, "clCreateBuffer");
		ocl_inputNormal[i] = clCreateBuffer(contexts[1], CL_MEM_READ_WRITE, sizeof(float3) * (computationSize.x * computationSize.y) / (int) pow(2, i), NULL, &clError);
		checkErr(clError, "clCreateBuffer");
	}

	ocl_vertex = clCreateBuffer(contexts[1], CL_MEM_READ_WRITE, sizeof(float3) * computationSize.x * computationSize.y, NULL, &clError);
	checkErr(clError, "clCreateBuffer");
	ocl_normal = clCreateBuffer(contexts[1], CL_MEM_READ_WRITE, sizeof(float3) * computationSize.x * computationSize.y, NULL, &clError);
	checkErr(clError, "clCreateBuffer");
	ocl_trackingResult = clCreateBuffer(contexts[1], CL_MEM_READ_WRITE, sizeof(TrackData) * computationSize.x * computationSize.y, NULL, &clError);
	checkErr(clError, "clCreateBuffer");

	ocl_reduce_output_buffer = clCreateBuffer(contexts[1], CL_MEM_WRITE_ONLY, 32 * number_of_groups * sizeof(float), NULL, &clError);
	checkErr(clError, "clCreateBuffer");
	reduceOutputBuffer = (float*) malloc(number_of_groups * 32 * sizeof(float));
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
	track_ocl_kernel = clCreateKernel(programs[1], "trackKernel", &clError);
	checkErr(clError, "clCreateKernel");
	reduce_ocl_kernel = clCreateKernel(programs[1], "reduceKernel", &clError);
	checkErr(clError, "clCreateKernel");
	integrate_ocl_kernel = clCreateKernel(programs[1], "integrateKernel", &clError);
	checkErr(clError, "clCreateKernel");
	raycast_ocl_kernel = clCreateKernel(programs[1], "raycastKernel", &clError);
	checkErr(clError, "clCreateKernel");
	renderVolume_ocl_kernel = clCreateKernel(programs[1], "renderVolumeKernel", &clError);
	checkErr(clError, "clCreateKernel");
	renderDepth_ocl_kernel = clCreateKernel(programs[1], "renderDepthKernel", &clError);
	checkErr(clError, "clCreateKernel");
	renderTrack_ocl_kernel = clCreateKernel(programs[1], "renderTrackKernel", &clError);
	checkErr(clError, "clCreateKernel");

}
Kfusion::~Kfusion() {
	if (reduceOutputBuffer) free(reduceOutputBuffer);
	reduceOutputBuffer = NULL;

	for (unsigned int i = 0; i < iterations.size(); ++i) {
		if (ocl_ScaledDepth[i]) {
			clError = clReleaseMemObject(ocl_ScaledDepth[i]);
			checkErr(clError, "clReleaseMem");
			ocl_ScaledDepth[i] = NULL;
		}
		if (ocl_inputVertex[i]) {
			clError = clReleaseMemObject(ocl_inputVertex[i]);
			checkErr(clError, "clReleaseMem");
			ocl_inputVertex[i] = NULL;
		}
		if (ocl_inputNormal[i]) {
			clError = clReleaseMemObject(ocl_inputNormal[i]);
			checkErr(clError, "clReleaseMem");
			ocl_inputNormal[i] = NULL;
		}
	}
	if (ocl_ScaledDepth) {
		free(ocl_ScaledDepth);
		ocl_ScaledDepth = NULL;
	}
	if (ocl_inputVertex) {
		free(ocl_inputVertex);
		ocl_inputVertex = NULL;
	}
	if (ocl_inputNormal) {
		free(ocl_inputNormal);
		ocl_inputNormal = NULL;
	}

	if (ocl_FloatDepth) {
		clError = clReleaseMemObject(ocl_FloatDepth);
		checkErr(clError, "clReleaseMem");
		ocl_FloatDepth = NULL;
	}
	if (ocl_vertex) {
		clError = clReleaseMemObject(ocl_vertex);
		checkErr(clError, "clReleaseMem");
		ocl_vertex = NULL;
	}
	if (ocl_normal) {
	  	clError = clReleaseMemObject(ocl_normal);
	  	checkErr(clError, "clReleaseMem");
		ocl_normal = NULL;
	}
	if (ocl_trackingResult) {
	 	clError = clReleaseMemObject(ocl_trackingResult);
		checkErr(clError, "clReleaseMem");
		ocl_trackingResult = NULL;
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
	RELEASE_KERNEL(reduce_ocl_kernel);
	RELEASE_KERNEL(integrate_ocl_kernel);
	RELEASE_KERNEL(raycast_ocl_kernel);
	RELEASE_KERNEL(renderVolume_ocl_kernel);
	RELEASE_KERNEL(renderDepth_ocl_kernel);
	RELEASE_KERNEL(renderTrack_ocl_kernel);
	RELEASE_KERNEL(initVolume_ocl_kernel);

	mm2meters_ocl_kernel = NULL ;
	bilateralFilter_ocl_kernel = NULL;
	halfSampleRobustImage_ocl_kernel = NULL;
	depth2vertex_ocl_kernel = NULL;
	vertex2normal_ocl_kernel = NULL;
	track_ocl_kernel = NULL;
	reduce_ocl_kernel = NULL;
	integrate_ocl_kernel = NULL;
	raycast_ocl_kernel = NULL;
	renderVolume_ocl_kernel = NULL;
	renderLight_ocl_kernel = NULL;
	renderTrack_ocl_kernel = NULL;
	renderDepth_ocl_kernel = NULL;

	computationSizeBkp = make_uint2(0, 0);
	outputImageSizeBkp = make_uint2(0, 0);

	clean();
}

bool updatePoseKernel(Matrix4 & pose, const float * output, float icp_threshold) {
	// Update the pose regarding the tracking result
	TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);
	TooN::Vector<6> x = solve(values[0].slice<1, 27>());
	TooN::SE3<> delta(x);
	pose = toMatrix4(delta) * pose;

	// Return validity test result of the tracking
	if (norm(x) < icp_threshold) return true;
	return false;
}

bool checkPoseKernel(Matrix4 & pose, Matrix4 oldPose, const float * output, uint2 imageSize, float track_threshold) {
	// Check the tracking result, and go back to the previous camera position if necessary
	if ((std::sqrt(output[0] / output[28]) > 2e-2) || (output[28] / (imageSize.x * imageSize.y) < track_threshold)) {
		pose = oldPose;
		return false;
	} else {
		return true;
	}
}

void Kfusion::reset() {
	std::cerr << "Reset function to clear volume model needs to be implemented\n";
	exit(1);
}

void Kfusion::renderVolume(uchar4 * out, uint2 outputSize, int frame, int rate, float4 k, float largestep) {
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
	clError = clSetKernelArg(renderVolume_ocl_kernel, arg++, sizeof(cl_float4), (void*) &(view.data[0]));
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);
	clError = clSetKernelArg(renderVolume_ocl_kernel, arg++, sizeof(cl_float4), (void*) &(view.data[1]));
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);
	clError = clSetKernelArg(renderVolume_ocl_kernel, arg++, sizeof(cl_float4), (void*) &(view.data[2]));
	sprintf(errStr, "clSetKernelArg%d", arg);
	checkErr(clError, errStr);
	clError = clSetKernelArg(renderVolume_ocl_kernel, arg++, sizeof(cl_float4), (void*) &(view.data[3]));
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
}

void Kfusion::renderTrack(uchar4 * out, uint2 outputSize) {
    // Create render opencl buffer if needed
    if(outputImageSizeBkp.x < outputSize.x || outputImageSizeBkp.y < outputSize.y || ocl_output_render_buffer == NULL) 
    {
		outputImageSizeBkp = make_uint2(outputSize.x, outputSize.y);
		if(ocl_output_render_buffer != NULL){
		    std::cout << "Release" << std::endl;
		    clError = clReleaseMemObject(ocl_output_render_buffer);
		    checkErr(clError, "clReleaseMemObject");
		}
		ocl_output_render_buffer = clCreateBuffer(contexts[1], CL_MEM_WRITE_ONLY, outputSize.x * outputSize.y * sizeof(uchar4), NULL , &clError);
		checkErr(clError, "clCreateBuffer output" );
    }

	// set param and run kernel
	clError = clSetKernelArg(renderTrack_ocl_kernel, 0, sizeof(cl_mem), &ocl_output_render_buffer);
	clError &= clSetKernelArg(renderTrack_ocl_kernel, 1, sizeof(cl_mem), &ocl_trackingResult);
	checkErr(clError, "clSetKernelArg");

	size_t globalWorksize[2] = { computationSize.x, computationSize.y };

	clError = clEnqueueNDRangeKernel(cmd_queues[1][0], renderTrack_ocl_kernel, 2, NULL, globalWorksize, NULL, 0, NULL, NULL);
	checkErr(clError, "clEnqueueNDRangeKernel");

    clError = clEnqueueReadBuffer(cmd_queues[1][0], ocl_output_render_buffer, CL_FALSE, 0, outputSize.x * outputSize.y * sizeof(uchar4), out, 0, NULL, NULL );  
    checkErr(clError, "clEnqueueReadBuffer");

}

void Kfusion::renderDepth(uchar4 * out, uint2 outputSize) {
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

bool Kfusion::preprocessing(const uint16_t * inputDepth, const uint2 inSize) {

	// bilateral_filter(ScaledDepth[0], inputDepth, inputSize , gaussian, e_delta, radius);
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

	return true;

}
bool Kfusion::tracking(float4 k, float icp_threshold, uint tracking_rate, uint frame) {
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

	// prepare the 3D information from the input depth maps
	uint2 localimagesize = computationSize;
	for (unsigned int i = 0; i < iterations.size(); ++i) {
		Matrix4 invK = getInverseCameraMatrix(k / float(1 << i));

		uint2 imageSize = localimagesize;
		// Create kernel

		int arg = 0;
		char errStr[20];

		clError = clSetKernelArg(depth2vertex_ocl_kernel, arg++, sizeof(cl_mem), &ocl_inputVertex[i]);
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
		clError = clSetKernelArg(depth2vertex_ocl_kernel, arg++, sizeof(cl_float4), &(invK.data[0]));
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(depth2vertex_ocl_kernel, arg++, sizeof(cl_float4), &(invK.data[1]));
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(depth2vertex_ocl_kernel, arg++, sizeof(cl_float4), &(invK.data[2]));
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(depth2vertex_ocl_kernel, arg++, sizeof(cl_float4), &(invK.data[3]));
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);

		size_t globalWorksize[2] = { imageSize.x, imageSize.y };

		clError = clEnqueueNDRangeKernel(cmd_queues[1][0], depth2vertex_ocl_kernel, 2, NULL, globalWorksize, NULL, 0, NULL, NULL);
		checkErr(clError, "clEnqueueNDRangeKernel");

		arg = 0;
		clError = clSetKernelArg(vertex2normal_ocl_kernel, arg++, sizeof(cl_mem), &ocl_inputNormal[i]);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(vertex2normal_ocl_kernel, arg++, sizeof(cl_uint2), &imageSize);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(vertex2normal_ocl_kernel, arg++, sizeof(cl_mem), &ocl_inputVertex[i]);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(vertex2normal_ocl_kernel, arg++, sizeof(cl_uint2), &imageSize);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);

		size_t globalWorksize2[2] = { imageSize.x, imageSize.y };

		clError = clEnqueueNDRangeKernel(cmd_queues[1][0], vertex2normal_ocl_kernel, 2, NULL, globalWorksize2, NULL, 0, NULL, NULL);
		checkErr(clError, "clEnqueueNDRangeKernel");

		localimagesize = make_uint2(localimagesize.x / 2, localimagesize.y / 2);
	}
	oldPose = pose;
	const Matrix4 projectReference = getCameraMatrix(k) * inverse(raycastPose);

	for (int level = iterations.size() - 1; level >= 0; --level) {
		uint2 localimagesize = make_uint2(computationSize.x / (int) pow(2, level), computationSize.y / (int) pow(2, level));
		for (int i = 0; i < iterations[level]; ++i) {
			int arg = 0;
			char errStr[20];

			clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_mem), &ocl_trackingResult);
			sprintf(errStr, "clSetKernelArg%d", arg);
			checkErr(clError, errStr);
			clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_uint2), &computationSize);
			sprintf(errStr, "clSetKernelArg%d", arg);
			checkErr(clError, errStr);
			clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_mem), &ocl_inputVertex[level]);
			sprintf(errStr, "clSetKernelArg%d", arg);
			checkErr(clError, errStr);
			clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_uint2), &localimagesize);
			sprintf(errStr, "clSetKernelArg%d", arg);
			checkErr(clError, errStr);
			clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_mem), &ocl_inputNormal[level]);
			sprintf(errStr, "clSetKernelArg%d", arg);
			checkErr(clError, errStr);
			clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_uint2), &localimagesize);
			sprintf(errStr, "clSetKernelArg%d", arg);
			checkErr(clError, errStr);
			clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_mem), &ocl_vertex);
			sprintf(errStr, "clSetKernelArg%d", arg);
			checkErr(clError, errStr);
			clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_uint2), &computationSize);
			sprintf(errStr, "clSetKernelArg%d", arg);
			checkErr(clError, errStr);
			clError = clSetKernelArg(track_ocl_kernel, arg++, sizeof(cl_mem), &ocl_normal);
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

			clError = clEnqueueNDRangeKernel(cmd_queues[1][0], track_ocl_kernel, 2, NULL, globalWorksize, NULL, 0, NULL, NULL);
			checkErr(clError, "clEnqueueNDRangeKernel");

			arg = 0;
			clError = clSetKernelArg(reduce_ocl_kernel, arg++, sizeof(cl_mem), &ocl_reduce_output_buffer);
			sprintf(errStr, "clSetKernelArg%d", arg);
			checkErr(clError, errStr);
			clError = clSetKernelArg(reduce_ocl_kernel, arg++, sizeof(cl_mem), &ocl_trackingResult);
			sprintf(errStr, "clSetKernelArg%d", arg);
			checkErr(clError, errStr);
			clError = clSetKernelArg(reduce_ocl_kernel, arg++, sizeof(cl_uint2), &computationSize);
			sprintf(errStr, "clSetKernelArg%d", arg);
			checkErr(clError, errStr);
			clError = clSetKernelArg(reduce_ocl_kernel, arg++, sizeof(cl_uint2), &localimagesize);
			sprintf(errStr, "clSetKernelArg%d", arg);
			checkErr(clError, errStr);
			clError = clSetKernelArg(reduce_ocl_kernel, arg++, size_of_group * 32 * sizeof(float), NULL);
			sprintf(errStr, "clSetKernelArg%d", arg);
			checkErr(clError, errStr);

			size_t RglobalWorksize[1] = { size_of_group * number_of_groups };
			size_t RlocalWorksize[1] = { size_of_group }; // Dont change it !

			clError = clEnqueueNDRangeKernel(cmd_queues[1][0], reduce_ocl_kernel, 1, NULL, RglobalWorksize, RlocalWorksize, 0, NULL, NULL);
			checkErr(clError, "clEnqueueNDRangeKernel");

			clError = clEnqueueReadBuffer(cmd_queues[1][0], ocl_reduce_output_buffer, CL_TRUE, 0, 32 * number_of_groups * sizeof(float), reduceOutputBuffer, 0, NULL, NULL);
			checkErr(clError, "clEnqueueReadBuffer");

			TooN::Matrix<TooN::Dynamic, TooN::Dynamic, float, TooN::Reference::RowMajor> values(reduceOutputBuffer, number_of_groups, 32);

			for (int j = 1; j < number_of_groups; ++j) {
				values[0] += values[j];
			}

			if (updatePoseKernel(pose, reduceOutputBuffer, icp_threshold)) break;
		}
	}

	return checkPoseKernel(pose, oldPose, reduceOutputBuffer, computationSize, track_threshold);
}

bool Kfusion::raycasting(float4 k, float mu, uint frame) {
	bool doRaycast = false;
	float largestep = mu * 0.75f;

	if (frame > 2) {
		checkErr(clError, "clEnqueueNDRangeKernel");
		raycastPose = pose;
		const Matrix4 view = raycastPose * getInverseCameraMatrix(k);

		int arg = 0;
		char errStr[20];

		clError = clSetKernelArg(raycast_ocl_kernel, arg++, sizeof(cl_mem), (void*) &ocl_vertex);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(raycast_ocl_kernel, arg++, sizeof(cl_mem), (void*) &ocl_normal);
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
		clError = clSetKernelArg(raycast_ocl_kernel, arg++, sizeof(cl_float4), (void*) &(view.data[0]));
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(raycast_ocl_kernel, arg++, sizeof(cl_float4), (void*) &(view.data[1]));
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(raycast_ocl_kernel, arg++, sizeof(cl_float4), (void*) &(view.data[2]));
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(raycast_ocl_kernel, arg++, sizeof(cl_float4), (void*) &(view.data[3]));
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

	return doRaycast;
}

bool Kfusion::integration(float4 k, uint integration_rate, float mu, uint frame) {
	bool doIntegrate = checkPoseKernel(pose, oldPose, reduceOutputBuffer, computationSize, track_threshold);

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
		clError = clSetKernelArg(integrate_ocl_kernel, arg++, sizeof(cl_float4), (void*) &(invTrack.data[0]));
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(integrate_ocl_kernel, arg++, sizeof(cl_float4), (void*) &(invTrack.data[1]));
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(integrate_ocl_kernel, arg++, sizeof(cl_float4), (void*) &(invTrack.data[2]));
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(integrate_ocl_kernel, arg++, sizeof(cl_float4), (void*) &(invTrack.data[3]));
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(integrate_ocl_kernel, arg++, sizeof(cl_float4), (void*) &K.data[0]);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(integrate_ocl_kernel, arg++, sizeof(cl_float4), (void*) &K.data[1]);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(integrate_ocl_kernel, arg++, sizeof(cl_float4), (void*) &K.data[2]);
		sprintf(errStr, "clSetKernelArg%d", arg);
		checkErr(clError, errStr);
		clError = clSetKernelArg(integrate_ocl_kernel, arg++, sizeof(cl_float4), (void*) &K.data[3]);
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

	return doIntegrate;
}

void Kfusion::computeFrame(const ushort * inputDepth, const uint2 inputSize, float4 k, uint integration_rate, uint tracking_rate, float icp_threshold, float mu, const uint frame) {
	preprocessing(inputDepth, inputSize);
	_tracked = tracking(k, icp_threshold, tracking_rate, frame);
	_integrated = integration(k, integration_rate, mu, frame);
	raycasting(k, mu, frame);
}

void synchroniseDevices() {
	clFinish(cmd_queues[0][0]);
	clFinish(cmd_queues[1][0]);
	clFinish(cmd_queues[1][1]);
}
