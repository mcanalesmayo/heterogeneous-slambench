# Module for locating OpenCL.
#
# Customizable variables:
#   OPENCL_ROOT_DIR
#     Specifies OpenCL's root directory. The find module uses this variable to
#     locate OpenCL. The variable will be filled automatically unless explicitly
#     set using CMake's -D command-line option. Instead of setting a CMake
#     variable, an environment variable called OCLROOT can be used.
#     While locating the root directory, the module will try to detect OpenCL
#     implementations provided by AMD's Accelerated Parallel Processing SDK,
#     NVIDIA's GPU Computing Toolkit and Intel's OpenCL SDK by examining the
#     AMDAPPSDKROOT, CUDA_PATH and INTELOCLSDKROOT environment variables,
#     respectively.
#
# Read-only variables:
#   OPENCL_FOUND
#     Indicates whether OpenCL has been found.
#
#   OPENCL_INCLUDE_DIRS
#     Specifies the OpenCL include directories.
#
#   OPENCL_LIBRARIES
#     Specifies the OpenCL libraries that should be passed to
#     target_link_libararies.
#
#
# Copyright (c) 2012 Sergiu Dotenco
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTOPENCLLAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

INCLUDE (FindPackageHandleStandardArgs)

IF (CMAKE_SIZEOF_VOID_P EQUAL 8)
  SET (_OPENCL_POSSIBLE_LIB_SUFFIXES lib/Win64 lib/x86_64 lib/x64 lib lib64)
ELSE (CMAKE_SIZEOF_VOID_P EQUAL 8)
  SET (_OPENCL_POSSIBLE_LIB_SUFFIXES lib/Win32 lib/x86 lib)
ENDIF (CMAKE_SIZEOF_VOID_P EQUAL 8)

LIST (APPEND _OPENCL_POSSIBLE_LIB_SUFFIXES lib/nvidia-current linux64/lib)

FIND_PATH (OPENCL_ROOT_DIR
NAMES OpenCL/cl.h
      CL/cl.h
      include/CL/cl.h
      include/nvidia-current/CL/cl.h
HINTS ${CUDA_TOOLKIT_ROOT_DIR}
PATHS ENV OCLROOT
      ENV AMDAPPSDKROOT
      ENV CUDA_PATH
      ENV INTELOCLSDKROOT
      ENV ALTERAOCLSDKROOT
PATH_SUFFIXES host/include
DOC "OpenCL root directory")

MESSAGE ("-- OpenCL root directory found in ${OPENCL_ROOT_DIR}")

FIND_PATH (OPENCL_INCLUDE_DIR
NAMES OpenCL/cl.h CL/cl.h
HINTS ${OPENCL_ROOT_DIR}
PATH_SUFFIXES include include/nvidia-current
DOC "OpenCL include directory")

MESSAGE ("-- OpenCL include directory found in ${OPENCL_INCLUDE_DIR}")

FIND_LIBRARY (OPENCL_LIBRARY
NAMES OpenCL
HINTS ${OPENCL_ROOT_DIR}  /usr/local/lib/mali/fbdev/
PATHS ENV ALTERAOCLSDKROOT
PATH_SUFFIXES ${_OPENCL_POSSIBLE_LIB_SUFFIXES}
DOC "OpenCL library")

MESSAGE ("-- OpenCL library found in ${OPENCL_LIBRARY}")

SET (OPENCL_INCLUDE_DIRS ${OPENCL_INCLUDE_DIR})
SET (OPENCL_LIBRARIES ${OPENCL_LIBRARY})

# for Altera environments more steps need to be taken: include additional dirs and link additional libraries
IF(DEFINED ENV{ALTERAOCLSDKROOT})
  # find OpenCL compiler
  # aoc kernel.cl -o kernel.aocx --board de5net_a7
  #FIND_PROGRAM(AOC_BIN aoc)
  #IF(NOT AOC_BIN)
  #  MESSAGE (FATAL_ERROR "Altera OpenCL compiler not found")
  #ENDIF(NOT AOC_BIN)

  # include OpenCL headers
  EXECUTE_PROCESS(COMMAND aocl compile-config OUTPUT_VARIABLE AOCL_INCLUDE_DIRS)
  STRING(REPLACE "-I" "" AOCL_INCLUDE_DIRS ${AOCL_INCLUDE_DIRS})
  STRING(REPLACE " " ";" AOCL_INCLUDE_DIRS ${AOCL_INCLUDE_DIRS})
  FOREACH(INC_DIR ${AOCL_INCLUDE_DIRS})
    # check if directory was already included
    LIST (FIND OPENCL_INCLUDE_DIRS ${INC_DIR} _IDX)
    IF (${_IDX} GREATER -1)
      SET (OPENCL_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS} ${INC_DIR})
    ENDIF(${_IDX} GREATER -1)
  ENDFOREACH(INC_DIR)
  
  # required OpenCL libraries
  # get directories where libraries are located
  EXECUTE_PROCESS(COMMAND aocl ldflags OUTPUT_VARIABLE AOCL_LINK_LIBRARIES_DIRS)
  STRING(REPLACE "-L" "" AOCL_LINK_LIBRARIES_DIRS ${AOCL_LINK_LIBRARIES_DIRS})
  STRING(REPLACE "\n" "" AOCL_LINK_LIBRARIES_DIRS ${AOCL_LINK_LIBRARIES_DIRS})
  STRING(REPLACE " " ";" AOCL_LINK_LIBRARIES_DIRS ${AOCL_LINK_LIBRARIES_DIRS})

  # get libraries to find
  EXECUTE_PROCESS(COMMAND aocl ldlibs OUTPUT_VARIABLE AOCL_LINK_LIBRARIES)
  STRING(REPLACE " " ";" AOCL_LINK_LIBRARIES ${AOCL_LINK_LIBRARIES})
  FOREACH(BKP_FLAG ${AOCL_LINK_LIBRARIES})
    STRING(REPLACE "-l" "" LINK_FLAG ${BKP_FLAG})
    # link library flag
    IF (NOT ${LINK_FLAG} STREQUAL ${BKP_FLAG})
      # trim flag
      STRING(REPLACE "\n" "" LINK_FLAG ${LINK_FLAG})
      # skip libstdc++
      IF (NOT ${LINK_FLAG} STREQUAL "stdc++")
        # find library
        FIND_LIBRARY (OPENCL_LIBRARY_${LINK_FLAG}
        NAMES ${LINK_FLAG}
        PATHS ${AOCL_LINK_LIBRARIES_DIRS}
        DOC "OpenCL library")

        # if library was not found then raise fatal error
        IF (${OPENCL_LIBRARY_${LINK_FLAG}} STREQUAL "OPENCL_LIBRARY_${LINK_FLAG}-NOTFOUND")
          MESSAGE (FATAL_ERROR "-- Library ${LINK_FLAG} not found")
        ENDIF (${OPENCL_LIBRARY_${LINK_FLAG}} STREQUAL "OPENCL_LIBRARY_${LINK_FLAG}-NOTFOUND")

        # append to list of libraries
        SET (OPENCL_LIBRARIES "${OPENCL_LIBRARIES} ${OPENCL_LIBRARY_${LINK_FLAG}}")
      ENDIF (NOT ${LINK_FLAG} STREQUAL "stdc++")
    # other flags
    ELSE (NOT ${LINK_FLAG} STREQUAL ${BKP_FLAG})
      # append to list of linker flags
      SET (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BKP_FLAG}")
    ENDIF (NOT ${LINK_FLAG} STREQUAL ${BKP_FLAG})
  ENDFOREACH(BKP_FLAG)

  SET(OPENCL_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS} src/opencl/AOCLUtils)
  SET(AOCL_UTILS_SRCS src/opencl/AOCLUtils/opencl.cpp src/opencl/AOCLUtils/options.cpp)

  MESSAGE ("-- Found Altera OpenCL $ENV{ALTERAOCLSDKROOT}")
ENDIF(DEFINED ENV{ALTERAOCLSDKROOT})

MESSAGE ("-- Included directories: ${OPENCL_INCLUDE_DIRS}")
MESSAGE ("-- Linked libraries: ${OPENCL_LIBRARIES}")
MESSAGE ("-- Compiler version: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

IF (OPENCL_INCLUDE_DIR AND OPENCL_LIBRARY)
  SET (_OPENCL_VERSION_TEST_SOURCE
"
#if __APPLE__
#include <OpenCL/cl.h>
#else /* !__APPLE__ */
#include <CL/cl.h>
#endif /* __APPLE__ */

#include <stdio.h>
#include <stdlib.h>

int main()
{
    char *version;
    cl_int result;
    cl_platform_id id;
    size_t n;

    result = clGetPlatformIDs(1, &id, NULL);

    if (result == CL_SUCCESS) {
        result = clGetPlatformInfo(id, CL_PLATFORM_VERSION, 0, NULL, &n);

        if (result == CL_SUCCESS) {
            version = (char*)malloc(n * sizeof(char));

            result = clGetPlatformInfo(id, CL_PLATFORM_VERSION, n, version,
                NULL);

            if (result == CL_SUCCESS) {
                printf(\"%s\", version);
                fflush(stdout);
            }

            free(version);
        }
    }

    return result == CL_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
  ")

  SET (_OPENCL_VERSION_SOURCE
    "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/openclversion.c")

  FILE (WRITE ${_OPENCL_VERSION_SOURCE} "${_OPENCL_VERSION_TEST_SOURCE}\n")

  STRING(REPLACE " " ";" OPENCL_LIBRARIES ${OPENCL_LIBRARIES})

  TRY_RUN (_OPENCL_VERSION_RUN_RESULT _OPENCL_VERSION_COMPILE_RESULT
    ${CMAKE_BINARY_DIR} ${_OPENCL_VERSION_SOURCE}
    RUN_OUTPUT_VARIABLE _OPENCL_VERSION_STRING
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${OPENCL_INCLUDE_DIRS}"
                "-DLINK_LIBRARIES:STRING=${OPENCL_LIBRARIES}"
    COMPILE_OUTPUT_VARIABLE _OPENCL_VERSION_COMPILE_OUTPUT)

  MESSAGE("Tried compiling and running openclversion.c, result is: ${_OPENCL_VERSION_RUN_RESULT}, succesfully compiled: ${_OPENCL_VERSION_COMPILE_RESULT}, compile output:\n\n${_OPENCL_VERSION_COMPILE_OUTPUT}, \n\nend of compile output")

  IF (_OPENCL_VERSION_RUN_RESULT EQUAL 0)
    STRING (REGEX REPLACE "OpenCL[ \t]+([0-9]+)\\.[0-9]+.*" "\\1"
      OPENCL_VERSION_MAJOR "${_OPENCL_VERSION_STRING}")
    STRING (REGEX REPLACE "OpenCL[ \t]+[0-9]+\\.([0-9]+).*" "\\1"
      OPENCL_VERSION_MINOR "${_OPENCL_VERSION_STRING}")

    SET (OPENCL_VERSION_COMPONENTS 2)
    SET (OPENCL_VERSION "${OPENCL_VERSION_MAJOR}.${OPENCL_VERSION_MINOR}")
  ENDIF (_OPENCL_VERSION_RUN_RESULT EQUAL 0)

  IF ("${OPENCL_VERSION}" STREQUAL "")
    MESSAGE (WARNING "Cannot determine OpenCL's version")
  ENDIF ("${OPENCL_VERSION}" STREQUAL "")
ENDIF (OPENCL_INCLUDE_DIR AND OPENCL_LIBRARY)

MARK_AS_ADVANCED (OPENCL_INCLUDE_DIR OPENCL_LIBRARY)

FIND_PACKAGE_HANDLE_STANDARD_ARGS (OpenCL REQUIRED_VARS OPENCL_ROOT_DIR
  OPENCL_INCLUDE_DIR OPENCL_LIBRARY VERSION_VAR OPENCL_VERSION)


SET(OPENCL_INCLUDE_DIRS /home/root/opencl_arm32_rte/host/include /home/root/heterogeneous-slambench/kfusion/src/opencl/AOCLUtils/src/opencl/AOCLUtils)
SET(AOCL_UTILS_SRCS /home/root/heterogeneous-slambench/kfusion/src/opencl/AOCLUtils/opencl.cpp /home/root/heterogeneous-slambench/kfusion/src/opencl/AOCLUtils/options.cpp)
SET(OPENCL_LIBRARIES -L/home/root/opencl_arm32_rte/board/c5soc/arm32/lib -L/home/root/opencl_arm32_rte/host/arm32/lib alteracl alterammdpcie elf)
