/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#include <kernels.h>
#include <interface.h>
#include <stdint.h>
#include <vector>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <csignal>

#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>
#include <iomanip>
#include <getopt.h>

// OMP
// #include <thread>
// #include <omp.h>

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



/***
 * This program loop over a scene recording
 */

int main(int argc, char ** argv) {

	Configuration config(argc, argv);

	// ========= CHECK ARGS =====================

	std::ostream* logstreamIO = &std::cout;
	std::ostream* logstreamCPU = &std::cout;
	std::ofstream logfilestreamIO;
	std::ofstream logfilestreamCPU;
	assert(config.compute_size_ratio > 0);
	assert(config.integration_rate > 0);
	assert(config.volume_size.x > 0);
	assert(config.volume_resolution.x > 0);

	// if (getenv("OMP")){
	// 	// set OMP threads if running OMP version
	// 	unsigned int nThreads = std::thread::hardware_concurrency();
	// 	omp_set_num_threads(nThreads);	
	// }

	if (config.log_file != "") {
		logfilestreamIO.open(config.log_file.c_str());
		logstreamIO = &logfilestreamIO;
	}
	if (config.log_file_cpu != "") {
		logfilestreamCPU.open(config.log_file_cpu.c_str());
		logstreamCPU = &logfilestreamCPU;
	}
	if (config.input_file == "") {
		std::cerr << "No input found." << std::endl;
		config.print_arguments();
		exit(1);
	}

	// ========= READER INITIALIZATION  =========

	DepthReader * reader;

	if (is_file(config.input_file)) {
		reader = new RawDepthReader(config.input_file, config.fps,
				config.blocking_read);

	} else {
		reader = new SceneDepthReader(config.input_file, config.fps,
				config.blocking_read);
	}

	std::cout.precision(10);
	std::cerr.precision(10);

	float3 init_pose = config.initial_pos_factor * config.volume_size;
	const uint2 inputSize = reader->getinputSize();
	std::cerr << "input Size is = " << inputSize.x << "," << inputSize.y
			<< std::endl;

	//  =========  BASIC PARAMETERS  (input size / computation size )  =========

	printf("input size.x: %d; input size.y: %d; config.compute_size_ratio: %d\n", inputSize.x, inputSize.y, config.compute_size_ratio);
	const uint2 computationSize = make_uint2(
			inputSize.x / config.compute_size_ratio,
			inputSize.y / config.compute_size_ratio);
	float4 camera = reader->getK() / config.compute_size_ratio;

	if (config.camera_overrided)
		camera = config.camera / config.compute_size_ratio;
	//  =========  BASIC BUFFERS  (input / output )  =========

	// Construction Scene reader and input buffer
	uint16_t* inputDepth = (uint16_t*) malloc(
			sizeof(uint16_t) * inputSize.x * inputSize.y);
	uchar4* depthRender = (uchar4*) malloc(
			sizeof(uchar4) * computationSize.x * computationSize.y);
	uchar4* trackRender = (uchar4*) malloc(
			sizeof(uchar4) * computationSize.x * computationSize.y);
	uchar4* volumeRender = (uchar4*) malloc(
			sizeof(uchar4) * computationSize.x * computationSize.y);

	uint frame = 0;

	// 13 kernels
	double* timingsIO = (double *) malloc(13 * sizeof(double));
	double* timingsCPU = (double *) malloc(13 * sizeof(double));
	double startOfKernel, endOfKernel, computationTotalIO, computationTotalCPU, overallTotalIO, overallTotalCPU;
	Kfusion kfusion(computationSize, config.volume_resolution,
			config.volume_size, init_pose, config.pyramid, timingsIO, timingsCPU);

	*logstreamIO
			<< "frame\tacquisition\tpreprocess_mm2meters\tpreprocess_bilateralFilter\ttrack_halfSample\ttrack_depth2vertex\ttrack_vertex2normal"
			<< "\ttrack_track\ttrack_reduce\tintegrate\traycast\trenderDepth\trenderTrack\trenderVolume"
			<< "\tcomputation\ttotal    \tX          \tY          \tZ         \ttracked   \tintegrated"
			<< std::endl;
	logstreamIO->setf(std::ios::fixed, std::ios::floatfield);

	startOfKernel = benchmark_tock();
	while (reader->readNextDepthFrame(inputDepth)) {

		Matrix4 pose = kfusion.getPose();

		float xt = pose.data[0].w - init_pose.x;
		float yt = pose.data[1].w - init_pose.y;
		float zt = pose.data[2].w - init_pose.z;

		endOfKernel = benchmark_tock();
		timingsIO[0] = endOfKernel - startOfKernel;

		kfusion.preprocessing(inputDepth, inputSize);

		bool tracked = kfusion.tracking(camera, config.icp_threshold,
				config.tracking_rate, frame);

		bool integrated = kfusion.integration(camera, config.integration_rate,
				config.mu, frame);

		bool raycast = kfusion.raycasting(camera, config.mu, frame);

		kfusion.renderDepth(depthRender, computationSize);
		kfusion.renderTrack(trackRender, computationSize);
		kfusion.renderVolume(volumeRender, computationSize, frame,
				config.rendering_rate, camera, 0.75 * config.mu);

		// skip acquisition stage for computation measure
		computationTotalIO = 0.0f;
		for(uint i=1; i<13; i++) {
			computationTotalIO += timingsIO[i];
		}

		overallTotalIO = computationTotalIO + timingsIO[0];

		*logstreamIO << frame << "\t" << timingsIO[0] << "\t" //  acquisition
				<< timingsIO[1] << "\t"     //  preprocessing --> mm2meters
				<< timingsIO[2] << "\t"     //  preprocessing --> bilateralFilter
				<< timingsIO[3] << "\t"     //  tracking --> halfSample
				<< timingsIO[4] << "\t"     //  tracking --> depth2vertex
				<< timingsIO[5] << "\t"     //  tracking --> vertex2normal
				<< timingsIO[6] << "\t"     //  tracking --> track
				<< timingsIO[7] << "\t"     //  tracking --> reduce
				<< timingsIO[8] << "\t"     //  integration --> integrate
				<< timingsIO[9] << "\t"     //  raycasting --> raycast
				<< timingsIO[10] << "\t"     //  rendering --> renderDepth
				<< timingsIO[11] << "\t"     //  rendering --> renderTrack
				<< timingsIO[12] << "\t"     //  rendering --> renderVolume
				<< computationTotalIO << "\t"     //  computation
				<< overallTotalIO << "\t"     //  total
				<< xt << "\t" << yt << "\t" << zt << "\t"     //  X,Y,Z
				<< tracked << "        \t" << integrated // tracked and integrated flags
				<< std::endl;

		// skip acquisition stage for computation measure
		computationTotalCPU = 0.0f;
		for(uint i=1; i<13; i++) {
			computationTotalCPU += timingsCPU[i];
		}

		overallTotalCPU = computationTotalCPU + timingsCPU[0];

		*logstreamCPU << frame << "\t" << timingsCPU[0] << "\t" //  acquisition
				<< timingsCPU[1] << "\t"     //  preprocessing --> mm2meters
				<< timingsCPU[2] << "\t"     //  preprocessing --> bilateralFilter
				<< timingsCPU[3] << "\t"     //  tracking --> halfSample
				<< timingsCPU[4] << "\t"     //  tracking --> depth2vertex
				<< timingsCPU[5] << "\t"     //  tracking --> vertex2normal
				<< timingsCPU[6] << "\t"     //  tracking --> track
				<< timingsCPU[7] << "\t"     //  tracking --> reduce
				<< timingsCPU[8] << "\t"     //  integration --> integrate
				<< timingsCPU[9] << "\t"     //  raycasting --> raycast
				<< timingsCPU[10] << "\t"     //  rendering --> renderDepth
				<< timingsCPU[11] << "\t"     //  rendering --> renderTrack
				<< timingsCPU[12] << "\t"     //  rendering --> renderVolume
				<< computationTotalCPU << "\t"     //  computation
				<< overallTotalCPU << "\t"     //  total
				<< xt << "\t" << yt << "\t" << zt << "\t"     //  X,Y,Z
				<< tracked << "        \t" << integrated // tracked and integrated flags
				<< std::endl;

		frame++;

		startOfKernel = benchmark_tock();
	}
	// ==========     DUMP VOLUME      =========

	if (config.dump_volume_file != "") {
	  kfusion.dumpVolume(config.dump_volume_file.c_str());
	}

	//  =========  FREE BASIC BUFFERS  =========

	free(timingsIO);
	free(timingsCPU);
	free(inputDepth);
	free(depthRender);
	free(trackRender);
	free(volumeRender);
	return 0;

}

