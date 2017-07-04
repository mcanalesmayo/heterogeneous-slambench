#!/usr/bin/python
# Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
# Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1
#
# This code is licensed under the MIT License.




import sys
import re
import math
import numpy

import csv
import os.path

KERNEL_CONSISTENCY = [
    ["mm2meters", "mm2metersKernel"],  
    ["bilateral_filter","bilateralFilterKernel"],
    ["halfSampleRobust","halfSampleRobustImageKernel"],
    ["depth2vertex","depth2vertexKernel"],
    ["vertex2normal","vertex2normalKernel"],
    ["track","trackKernel"],
    ["reduce","reduceKernel"],
    ["integrate","integrateKernel"],
    ["raycast","raycastKernel"],
    ["renderDepth","renderDepthKernel"],
    ["renderLight","renderLightKernel"],
    ["renderTrack","renderTrackKernel"],
    ["renderVolume", "renderVolumeKernel"],
    ["ResetVolume","initVolumeKernel"],
    ["updatePose","updatePoseKernel"]
]

def translateName(n) :
    for variations in KERNEL_CONSISTENCY:
        if n in variations:
            return variations[0]
    return n

KERNEL_LOG_REGEX = "([^ ]+)\s([0-9.]+)"

if len(sys.argv) != 5:
    print "1st param: log file\n"
    print "2nd param: timestamp (as execution identifier)\n"
    print "3rd param: commit hash (as version identifier)\n"
    print "4th param: CSV filename\n"
    exit (1)

# open log file first
print "Kernel-level statistics. Times are in nanoseconds." 
fileref = open(sys.argv[1], 'r')
data = fileref.read()
fileref.close()
lines = data.split("\n") # remove head + first line

data = {}

for line in lines:
    matching = re.match(KERNEL_LOG_REGEX, line)
    if matching:
        name = translateName(matching.group(1))
        if not name in data:
            data[name] = []
        data[name].append(float(matching.group(2)))
#    else :
#        print  "Skip SlamBench line : " + line


timestamp = sys.argv[2].strip()
commitHash = sys.argv[3].strip()
filename = sys.argv[4].strip()
file_exists = os.path.isfile(filename)
csvHeader = ['Timestamp', 'CommitHash', 'Name', 'Count', 'Min', 'Max', 'Mean', 'Total']
with open(filename, 'a') as f:
    writer = csv.writer(f, delimiter=',')

    # if file didn't exist then header must be appended
    if not file_exists:
        writer.writerow(csvHeader)

    for variable in sorted(data.keys()):
        dataName = str(variable)
        dataCount = len(data[variable])
        dataMin = min(data[variable])
        dataMax = max(data[variable])
        dataMean = numpy.mean(data[variable])
        dataTotal = sum(data[variable])
        csvRow = [timestamp, commitHash, dataName, dataCount, dataMin, dataMax, dataMean, dataTotal]
        writer.writerow(csvRow)

        print "%20.20s" % dataName,
        print "\tCount : %d" % dataCount,
        print "\tMin   : %d" % dataMin,
        print "\tMax   : %d" % dataMax,
        print "\tMean  : %f" % dataMean,
        print "\tTotal : %d" % dataTotal