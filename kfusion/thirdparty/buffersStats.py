#!/usr/bin/python

import sys
import re

import os.path

if len(sys.argv) != 3:
    print "1st param: benchmark buffers log file\n"
    print "2nd param: output file\n"
    exit (1)

# {kernel} {read/write} {level} {iteration} {size of buffer in bytes} {time of transfer in seconds}
ENTRY_REGEXP = "([a-zA-Z0-9]+)\\t([a-zA-Z0-9]+)\\t([0-9]+)\\t([0-9]+)\\t([0-9]+)\\t([0-9.]+)"

fileref = open(sys.argv[1],'r')
data = fileref.read()
fileref.close()
lines = data.split("\n") # remove head + first line
entries = []
bufSizes = []
transferTimes = []
bandwidths = []

for line in lines:
    matching = re.match(ENTRY_REGEXP, line)

    if matching:
        kernelName = matching.group(1)
        operation = matching.group(2)
        level = int(matching.group(3))
        iteration = int(matching.group(4))
        bufSize = int(matching.group(5))
        transferTime = float(matching.group(6))
        bandwidth = float(bufSize)/transferTime

        entries.append({'kernelName': kernelName, 'operation': operation, 'level': level, 'iteration': iteration, 'bufSize': bufSize, 'transferTime': transferTime, 'bandwidth': bandwidth})
        bufSizes.append(bufSize)
        transferTimes.append(transferTime)
        bandwidths.append(bandwidth)

with open(sys.argv[2], 'w') as f:
    totalBufSize = sum(bufSizes)
    totalTransferTime = sum(transferTimes)
    totalbandwidth = float(totalBufSize)/totalTransferTime

    avgBufSize = totalBufSize/len(bufSizes)
    avgTransferTime = totalTransferTime/len(transferTimes)
    avgbandwidth = totalbandwidth/len(bandwidths)

    maxBufSize = max(bufSizes)
    minBufSize = min(bufSizes)

    maxTransferTime = max(transferTimes)
    maxTransferTimeEntry = (item for item in entries if item["transferTime"] == maxTransferTime).next()
    minTransferTime = min(transferTimes)
    minTransferTimeEntry = (item for item in entries if item["transferTime"] == minTransferTime).next()

    maxbandwidthTime = max(bandwidths)
    maxbandwidthTimeEntry = (item for item in entries if item["bandwidth"] == maxbandwidthTime).next()
    minbandwidthTime = min(bandwidths)
    minbandwidthTimeEntry = (item for item in entries if item["bandwidth"] == minbandwidthTime).next()

    f.write("Amount of entries: " + str(len(entries)) + "\n")
    f.write("Total size transferred: " + str(float(totalBufSize)/(1024*1024)) + " MB\n")
    f.write("Total transfer time: " + str(totalTransferTime) + " s\n")
    f.write("Overall bandwidth: " + str(totalbandwidth/(1024*1024)) + " MB/s\n\n")

    f.write("Average buffer size: " + str(float(avgBufSize)/(1024*1024)) + " MB\n")
    f.write("Average transfer time: " + str(avgTransferTime) + " s\n")
    f.write("Average bandwidth: " + str(avgbandwidth/(1024*1024)) + " MB/s\n\n")

    f.write("Max buffer size: " + str(float(maxBufSize)/(1024*1024)) + " MB\n")
    f.write("Min buffer size: " + str(float(minBufSize)/(1024*1024)) + " MB\n\n")
    
    f.write("Max transfer time entry... Buffer Size: " + str(float(maxTransferTimeEntry['bufSize'])/(1024*1024)) + " MB\tTransfer time: " + str(maxTransferTime) + " s\tbandwidth: " + str(maxTransferTimeEntry['bandwidth']/(1024*1024)) + " MB/s\n")
    f.write("Min transfer time entry... Buffer Size: " + str(float(minTransferTimeEntry['bufSize'])/(1024*1024)) + " MB\tTransfer time: " + str(minTransferTime) + " s\tbandwidth: " + str(minTransferTimeEntry['bandwidth']/(1024*1024)) + " MB/s\n")

    f.write("Max bandwidth entry... Buffer Size: " + str(float(maxbandwidthTimeEntry['bufSize'])/(1024*1024)) + " MB\tTransfer time: " + str(maxbandwidthTimeEntry['transferTime']) + " s\tbandwidth: " + str(maxbandwidthTime/(1024*1024)) + " MB/s\n")
    f.write("Min bandwidth entry... Buffer Size: " + str(float(minbandwidthTimeEntry['bufSize'])/(1024*1024)) + " MB\tTransfer time: " + str(minbandwidthTimeEntry['transferTime']) + " s\tbandwidth: " + str(minbandwidthTime/(1024*1024)) + " MB/s\n")
