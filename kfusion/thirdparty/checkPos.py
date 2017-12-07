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

# open files
if len(sys.argv) != 6:
    print "1st param: benchmark log file\n"
    print "2nd param: original scene camera position file\n"
    print "3rd param: timestamp (as execution identifier)\n"
    print "4th param: commit hash (as version identifier)\n"
    print "5th param: CSV filename\n"
    exit (1)

KFUSION_LOG_REGEX =      "([0-9]+[\s]*)\\t" 
KFUSION_LOG_REGEX += 15 * "([0-9.]+)\\t" 
KFUSION_LOG_REGEX += 3 * "([-0-9.]+)\\t" 
KFUSION_LOG_REGEX +=     "([01])\s+([01])" 

NUIM_LOG_REGEX =      "([0-9]+)" 
NUIM_LOG_REGEX += 7 * "\\s+([-0-9e.]+)\\s*" 

timestamp = sys.argv[3].strip()
commitHash = sys.argv[4].strip()

# open benchmark log file first
print "Get KFusion output data for version with commit {0}, execution id {1}".format(str(commitHash), str(timestamp))
framesDropped = 0
validFrames = 0
lastFrame = -1
untracked = -4
kfusion_traj = []
fileref = open(sys.argv[1],'r')
data = fileref.read()
fileref.close()
lines = data.split("\n") # remove head + first line
headers = lines[0].split("\t")
lenHeaders = len(headers)
fulldata = {}

for variable in headers:
    fulldata[variable] = []

for line in lines[1:]:
    matching = re.match(KFUSION_LOG_REGEX, line)
    if matching:
        print line
        dropped = int(matching.group(1)) - lastFrame - 1
        if dropped > 0:
            framesDropped = framesDropped + dropped
            for pad in range(0,dropped):
                 kfusion_traj.append(lastValid)

        kfusion_traj.append((matching.group(lenHeaders-4), matching.group(lenHeaders-3), matching.group(lenHeaders-2), matching.group(lenHeaders-1), 1))
        lastValid = (matching.group(lenHeaders-4), matching.group(lenHeaders-3), matching.group(lenHeaders-2), matching.group(lenHeaders-1), 0)
        if int(matching.group(lenHeaders-1)) == 0:
            untracked = untracked + 1

        validFrames = validFrames + 1
        for elem_idx in range(len(headers)):
            fulldata[headers[elem_idx]].append(float(matching.group(elem_idx + 1)))
        
        lastFrame = int(matching.group(1))
    else:
        #print "Skip KFusion line : " + line
        break

# open benchmark log file first
nuim_traj = []
fileref = open(sys.argv[2], 'r')
data = fileref.read()
fileref.close()
lines = data.split("\n") # remove head + first line
for line in lines:
    matching = re.match(NUIM_LOG_REGEX, line)
    if matching:
        nuim_traj.append((matching.group(2), matching.group(3), matching.group(4)))
    else:
        #print "Skip nuim line : " + line
        break

working_position = min(len(kfusion_traj), len(nuim_traj))
print "KFusion valid frames " + str(validFrames) + ", dropped frames: " + str(framesDropped)
print "KFusion result        : " + str(len(kfusion_traj)) + " positions."
print "NUIM result         : " + str(len(nuim_traj)) + " positions."
print "Working position is : " + str(working_position)
print "Untracked frames: " +str(untracked)
nuim_traj = nuim_traj[0:working_position]
kfusion_traj = kfusion_traj[0:working_position]

print "Shift KFusion trajectory..."

first = nuim_traj[0]
fulldata["ATE"] = []
#ATE_wrt_kfusion does not consider the ATE for frames which were dropped if we are running in non process-every-frame mode
fulldata["ATE_wrt_kfusion"] = []
distance_since_valid = 0
#print "Frame  speed(m/s)   dlv(m) ATE(m)   valid   tracked"
for p in range(working_position):
    kfusion_traj[p] = (float(kfusion_traj[p][0]) + float(first[0]), - (float(kfusion_traj[p][1]) + float(first[1])), float(kfusion_traj[p][2]) + float(first[2]), int(kfusion_traj[p][3]), int(kfusion_traj[p][4]))
    diff = (abs(kfusion_traj[p][0] - float(nuim_traj[p][0])), abs(kfusion_traj[p][1] - float(nuim_traj[p][1])), abs(kfusion_traj[p][2] - float(nuim_traj[p][2])))
    ate = math.sqrt(sum((diff[0] * diff[0], diff[1] * diff[1], diff[2] * diff[2])))

    if p == 1:
        lastValid = nuim_traj[p]

    dx = float(nuim_traj[p][0]) - float(lastValid[0])
    dy = float(nuim_traj[p][1]) - float(lastValid[1]) 
    dz = float(nuim_traj[p][2]) - float(lastValid[2])
    distA = math.sqrt((dx*dx) + (dz*dz))
    dist = math.sqrt((dy*dy) + (distA *distA))
    speed = dist/0.0333
    if kfusion_traj[p][3] == 0:
        tracked = "untracked"
    else:
        tracked = ""

    if kfusion_traj[p][4] == 0:
        valid = "dropped"
    else:
        valid = "-"

    distance_since_valid = distance_since_valid + dist
        #print "%4d %6.6f %6.6f %6.6f %10s %10s"% (p, speed, distance_since_valid, ate, valid, tracked )
    lastValid = nuim_traj[p]
    if kfusion_traj[p][4] == 1:
        distance_since_valid = 0
        fulldata["ATE_wrt_kfusion"].append(ate)

    fulldata["ATE"].append(ate)
        
#print "The following are designed to enable easy macchine readability of key data" 
#print "MRkey:,logfile,ATE,computaion,dropped,untracked"
#print ("MRdata:,%s,%6.6f,%6.6f,%d,%d") % ( sys.argv[1], numpy.mean(fulldata["ATE"]), numpy.mean(fulldata["computation"]), framesDropped, untracked)

print "\nA detailed statistical analysis is provided."
print "Runtimes are in seconds and the absolute trajectory error (ATE) is in meters." 
print "The ATE measure accuracy, check this number to see how precise your computation is."
print "Acceptable values are in the range of few centimeters."

filename = sys.argv[5].strip()
file_exists = os.path.isfile(filename)
csvHeader = ['Timestamp', 'CommitHash', 'Name', 'Min', 'Max', 'Mean', 'Total']
with open(filename, 'a') as f:
    writer = csv.writer(f, delimiter=',')

    # if file didn't exist then header must be appended
    if not file_exists:
        writer.writerow(csvHeader)

    for variable in sorted(fulldata.keys()) :
        if "X" in variable or "Z" in variable or "Y" in variable or "frame" in variable or "tracked" in variable or "integrated" in variable:  
            continue

        if (framesDropped == 0) and (str(variable) == "ATE_wrt_kfusion"):
            continue
        
        dataName = str(variable).strip()
        dataMin = min(fulldata[variable])
        dataMax = max(fulldata[variable])
        dataMean = numpy.mean(fulldata[variable])
        dataTotal = sum(fulldata[variable])
        csvRow = [timestamp, commitHash, dataName, dataMin, dataMax, dataMean, dataTotal]
        writer.writerow(csvRow)

        print "\t%s" % dataName,
        print "\tMin : %6.6f" % dataMin,
        print "\tMax : %0.6f"  % dataMax,
        print "\tMean : %0.6f" % dataMean,
        print "\tTotal : %0.8f" % dataTotal

#first2 = []S
#derive = []

#for row_idx in range(len(rows1)) :
#    col1 = rows1[row_idx].split("\t")
#    col2 = rows2[row_idx].split(" ")
#    v1 = col1[8:11]
#    v2 = col2[1:4]
#    if first2 == [] :
#        first2 = v2
#    v1 = [float(v1[0]) + float(first2[0]) , - (float(v1[1]) + float(first2[1]) ) , float(v1[2]) + float(first2[2]) ]
#    derive.append([abs(float(v1[0]) - float(v2[0])) , abs (float(v1[1]) - float(v2[1]) ) , abs(float(v1[2]) - float(v2[2])) ])

#maxderive = reduce(lambda a,d:  [max(a[0] , d[0]),max(a[1] , d[1]),max(a[2] , d[2])], derive, [0.0,0.0,0.0])
#minderive = reduce(lambda a,d:  [min(a[0] , d[0]),min(a[1] , d[1]),min(a[2] , d[2])], derive, [1000000,10000000.0,10000000.0])
#total = map(lambda x: x/len(rows1), reduce(lambda a,d: [a[0] + d[0],a[1] + d[1],a[2] + d[2]], derive, [0.0,0.0,0.0])) 

#print "Min derivation : " + str(min(minderive))
#print "Max derivation : " + str(max(maxderive))
#print "Average derivation : " + str(total[0])
