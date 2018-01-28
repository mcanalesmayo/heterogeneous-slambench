
#### ICL-NUIM TRAJECTORIES PARAMETERS  ####
0 = -s 5.0 -p 0.34,0.5,0.24 -z 4 -c 2 -r 2 -k 481.2,480,320,240
1 = -s 5.0 -p 0.485,0.5,0.55 -z 4 -c 2 -r 2 -k 481.2,480,320,240
2 = -s 4.8 -p 0.34,0.5,0.24 -z 4 -c 2 -r 2 -k 481.2,480,320,240
3 = -s 5.0 -p 0.2685,0.5,0.4 -z 4 -c 2 -r 2 -k 481.2,480,320,240

TIMESTAMP=$(shell date "+%Y_%m_%d_%H_%M_%S_%N")
COMMIT_HASH=$(shell git rev-parse --verify HEAD)
ROOT_DIR=$(shell pwd)
TOON_DIR=${ROOT_DIR}/TooN/install_dir
TOON_INCLUDE_DIR=${TOON_DIR}/include/
ifdef emulate
EMULATE=$(emulate)
else
EMULATE=false
endif

all : build

build : TooN
	mkdir -p build/
	cd build/ && cmake .. -DTOON_INCLUDE_PATH=${TOON_INCLUDE_DIR} $(CMAKE_ARGUMENTS)
	$(MAKE) -C build  $(MFLAGS) $(SPECIFIC_TARGET)

#### Dependencies ####

TooN:
	git clone https://github.com/edrosten/TooN.git
	cd TooN &&  git checkout 92241416d2a4874fd2334e08a5d417dfea6a1a3f
	mkdir -p ${TOON_DIR}
	cd TooN && ./configure --prefix=${TOON_DIR} --disable-lapack && make install


#### DATA SET GENERATION ####
./build/kfusion/thirdparty/scene2raw : TooN
	mkdir -p build/
	cd build/ && cmake .. -DTOON_INCLUDE_PATH=${TOON_INCLUDE_DIR} $(CMAKE_ARGUMENTS)
	$(MAKE) -C build  $(MFLAGS) scene2raw

living_room_traj%_loop.raw : ./build/kfusion/thirdparty/scene2raw 
	if test -x ./build/kfusion/thirdparty/scene2raw ; then echo "..." ; else echo "do make before"; false ; fi
	if test -x living_room_traj$(*F)_loop.raw; then echo "raw input file already present, no need for conversion. " ; else ./build/kfusion/thirdparty/scene2raw living_room_traj$(*F)_loop living_room_traj$(*F)_loop.raw; fi

living_room_traj%_loop : 
	mkdir $@
	cd $@ ; wget http://www.doc.ic.ac.uk/~ahanda/$@.tgz; tar xzf $@.tgz 

livingRoom%.gt.freiburg : 
	echo  "Download ground truth trajectory..."
	if test -x $@ ; then echo "Done" ; else wget http://www.doc.ic.ac.uk/~ahanda/VaFRIC/$@ ; fi


#### LOG GENERATION ####

%.opencl.log  : living_room_traj%_loop.raw livingRoom%.gt.freiburg
	$(MAKE) -C build  $(MFLAGS) kfusion-benchmark-opencl oclwrapper
	if ${EMULATE} == true; then CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 KERNEL_TIMINGS=1 LD_PRELOAD=./build/kfusion/thirdparty/liboclwrapper.so ./build/kfusion/kfusion-benchmark-opencl $($(*F)) -i  living_room_traj$(*F)_loop.raw -o benchmark.$@ -d volume.$@ 2> oclwrapper.$@; else KERNEL_TIMINGS=1 LD_PRELOAD=./build/kfusion/thirdparty/liboclwrapper.so ./build/kfusion/kfusion-benchmark-opencl $($(*F)) -i  living_room_traj$(*F)_loop.raw -o benchmark.$@ -d volume.$@ 2> oclwrapper.$@; fi
	cat  oclwrapper.$@ |grep -E ".+ [0-9]+ [0-9]+ [0-9]+" |cut -d" " -f1,4 >   kernels.$@
	./kfusion/thirdparty/checkPos.py benchmark.$@  livingRoom$(*F).gt.freiburg ${TIMESTAMP} ${COMMIT_HASH} ${ROOT_DIR}/$@.pos.csv > resume.$@
	./kfusion/thirdparty/checkKernels.py kernels.$@ ${TIMESTAMP} ${COMMIT_HASH} ${ROOT_DIR}/$@.kernels.csv >> resume.$@

%.cpp.log  :  living_room_traj%_loop.raw livingRoom%.gt.freiburg
	$(MAKE) -C build  $(MFLAGS) kfusion-benchmark-cpp
	KERNEL_TIMINGS=1 ./build/kfusion/kfusion-benchmark-cpp $($(*F)) -i  living_room_traj$(*F)_loop.raw -o  benchmark.$@ -d volume.$@ 2> kernels.$@
	./kfusion/thirdparty/checkPos.py benchmark.$@  livingRoom$(*F).gt.freiburg ${TIMESTAMP} ${COMMIT_HASH} ${ROOT_DIR}/$@.pos.csv > resume.$@
	./kfusion/thirdparty/checkKernels.py kernels.$@ ${TIMESTAMP} ${COMMIT_HASH} ${ROOT_DIR}/$@.kernels.csv >> resume.$@

%.openmp.log  :  living_room_traj%_loop.raw livingRoom%.gt.freiburg
	$(MAKE) -C build $(MFLAGS) kfusion-benchmark-openmp
	KERNEL_TIMINGS=1 OMP=1 ./build/kfusion/kfusion-benchmark-openmp $($(*F)) -i  living_room_traj$(*F)_loop.raw -o  benchmark.$@ -d volume.$@ 2> kernels.$@
	./kfusion/thirdparty/checkPos.py benchmark.$@  livingRoom$(*F).gt.freiburg ${TIMESTAMP} ${COMMIT_HASH} ${ROOT_DIR}/$@.pos.csv > resume.$@
	./kfusion/thirdparty/checkKernels.py kernels.$@ ${TIMESTAMP} ${COMMIT_HASH} ${ROOT_DIR}/$@.kernels.csv >> resume.$@

%.cuda.log  : living_room_traj%_loop.raw livingRoom%.gt.freiburg
	$(MAKE) -C build  $(MFLAGS) kfusion-benchmark-cuda
	nvprof --print-gpu-trace ./build/kfusion/kfusion-benchmark-cuda $($(*F)) -i  living_room_traj$(*F)_loop.raw -o  benchmark.$@ -d volume.$@ 2> nvprof.$@ || true
	cat  nvprof.$@ | kfusion/thirdparty/nvprof2log.py >   kernels.$@
	./kfusion/thirdparty/checkPos.py benchmark.$@  livingRoom$(*F).gt.freiburg ${TIMESTAMP} ${COMMIT_HASH} ${ROOT_DIR}/$@.pos.csv > resume.$@
	./kfusion/thirdparty/checkKernels.py kernels.$@ ${TIMESTAMP} ${COMMIT_HASH} ${ROOT_DIR}/$@.kernels.csv >> resume.$@


#### GENERAL GENERATION ####

clean :
	rm -rf build TooN
cleanall : 
	rm -rf build TooN
	rm -rf living_room_traj*_loop livingRoom*.gt.freiburg living_room_traj*_loop.raw
	rm -f *.log 


.PHONY : clean bench test all validate build

.PRECIOUS: living_room_traj%_loop livingRoom%.gt.freiburg living_room_traj%_loop.raw

