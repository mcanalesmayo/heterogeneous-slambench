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
entries = {}
bufSizes = {}
transferTimes = {}
bandwiths = {}

for line in lines:
    matching = re.match(ENTRY_REGEXP, line)

    if matching:
        kernelName = matching.group(1)
        operation = matching.group(2)
        level = int(matching.group(3))
        iteration = int(matching.group(4))
        bufSize = int(matching.group(5))
        transferTime = float(matching.group(6))
        bandwith = float(bufSize)/transferTime

        entries[(kernelName, operation, level, iteration)] = {'bufSize': bufSize, 'transferTime': transferTime, 'bandwith': bandwith}
        bufSizes[(kernelName, operation, level, iteration)] = bufSize
        transferTimes[(kernelName, operation, level, iteration)] = transferTime
        bandwiths[(kernelName, operation, level, iteration)] = bandwith

def averageInDict(d):
    total = 0.0
    length = 0
    for key,value in d.items():
        length += 1
        total += float(value)
    return total/length

with open(sys.argv[2], 'w') as f:
    avgBufSize = averageInDict(bufSizes)
    avgTransferTime = averageInDict(transferTimes)
    avgBandwith = averageInDict(bandwiths)

    totalBufSize = sum(bufSizes.values())
    totalTransferTime = sum(transferTimes.values())
    totalBandwith = float(totalBufSize)/totalTransferTime

    maxBufSize = max(bufSizes.values())
    minBufSize = min(bufSizes.values())

    maxTransferTime = max(transferTimes.values())
    maxTransferTimeIdx = transferTimes.keys()[transferTimes.values().index(maxTransferTime)]
    minTransferTime = min(transferTimes.values())
    minTransferTimeIdx = transferTimes.keys()[transferTimes.values().index(minTransferTime)]

    maxBandwithTime = max(bandwiths.values())
    maxBandwithTimeIdx = bandwiths.keys()[bandwiths.values().index(maxBandwithTime)]
    minBandwithTime = min(bandwiths.values())
    minBandwithTimeIdx = bandwiths.keys()[bandwiths.values().index(minBandwithTime)]

    f.write("Amount of entries: " + str(len(entries)) + "\n")
    f.write("Total size transferred: " + str(float(totalBufSize)/(1024*1024)) + " MB\n")
    f.write("Total transfer time: " + str(totalTransferTime) + " s\n")
    f.write("Overall bandwith: " + str(totalBandwith/(1024*1024)) + " MB/s\n\n")
    f.write("Average buffer size: " + str(float(avgBufSize)/(1024*1024)) + " MB\n")
    f.write("Average transfer time: " + str(avgTransferTime) + " s\n")
    f.write("Average bandwith: " + str(avgBandwith/(1024*1024)) + " MB/s\n\n")

    f.write("Max buffer size: " + str(float(maxBufSize)/(1024*1024)) + " MB\n")
    f.write("Min buffer size: " + str(float(minBufSize)/(1024*1024)) + " MB\n\n")
    
    f.write("Max transfer time entry... Buffer Size: " + str(float(bufSizes[maxTransferTimeIdx])/(1024*1024)) + " MB\tTransfer time: " + str(maxTransferTime) + " s\tBandwith: " + str(bandwiths[maxTransferTimeIdx]/(1024*1024)) + " MB/s\n")
    f.write("Min transfer time entry... Buffer Size: " + str(float(bufSizes[minTransferTimeIdx])/(1024*1024)) + " MB\tTransfer time: " + str(minTransferTime) + " s\tBandwith: " + str(bandwiths[minTransferTimeIdx]/(1024*1024)) + " MB/s\n")

    f.write("Max bandwith entry... Buffer Size: " + str(float(bufSizes[maxBandwithTimeIdx])/(1024*1024)) + " MB\tTransfer time: " + str(transferTimes[maxBandwithTimeIdx]) + " s\tBandwith: " + str(maxBandwithTime/(1024*1024)) + " MB/s\n")
    f.write("Min bandwith entry... Buffer Size: " + str(float(bufSizes[minBandwithTimeIdx])/(1024*1024)) + " MB\tTransfer time: " + str(transferTimes[minBandwithTimeIdx]) + " s\tBandwith: " + str(minBandwithTime/(1024*1024)) + " MB/s\n")
