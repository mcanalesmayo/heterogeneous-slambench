#/usr/bin/python

import os
import sys
import csv
from subprocess import call

if (len(sys.argv) != 4):
    print "1st param: number of executions\n"
    print "2nd param: ICL-nuim dataset ID\n"
    print "3rd param: platform: cpp | openmp | cuda | opencl\n"
    exit (1)

# number of executions
N_EXECS = int(sys.argv[1])
# ICL-nuim dataset ID
DATASET = sys.argv[2]
# cpp | openmp | cuda | opencl
PLATFORM = sys.argv[3]

ANALYZE_LOGS = ['pos', 'kernels']

# clean benchmark files
KERNELS_FILE = DATASET + '.' + PLATFORM + '.log.kernels.csv'
POS_FILE = DATASET + '.' + PLATFORM + '.log.pos.csv'

file_exists = os.path.isfile(KERNELS_FILE)
if file_exists:
    print 'Removing ' + KERNELS_FILE
    os.remove(KERNELS_FILE)
file_exists = os.path.isfile(POS_FILE)
if file_exists:
    print 'Removing ' + POS_FILE
    os.remove(POS_FILE)

proc_params = ['make', '.'.join([DATASET, PLATFORM, 'log'])]

for i in range(0, N_EXECS):
    print 'Running #{0}'.format(str(i+1))
    call(proc_params)

for fileSuffix in ANALYZE_LOGS:
    filename = '.'.join([DATASET, PLATFORM, 'log', fileSuffix, 'csv'])

    inputFile = csv.DictReader(open(filename))

    result = dict()
    avoidStages = ['ATE', 'total']
    checkedExecutions = []
    for row in inputFile:
        stageName = row['Name']
        if stageName in avoidStages:
            continue

        execVersion = row['CommitHash']
        if execVersion not in result:
            result[execVersion] = dict()
            result[execVersion]['values'] = dict()
            result[execVersion]['num_values'] = 0
            result[execVersion]['total_time'] = float(0)

        execId = row['Timestamp']
        if execId not in checkedExecutions:
            checkedExecutions.append(execId)
            result[execVersion]['num_values'] += 1

        stageTime = float(row['Total'])
        if stageName not in result[execVersion]['values']:
            result[execVersion]['values'][stageName] = stageTime
        else:
            result[execVersion]['values'][stageName] += stageTime

        result[execVersion]['total_time'] += stageTime

    for version in result:
        print "Analyzing %s" % filename
        print "\tVersion: %s" % version
        print "\tNumber of execs: %d" % result[execVersion]['num_values']
        for stageName in result[execVersion]['values']:
            stageTotal = result[execVersion]['values'][stageName]/result[execVersion]['num_values']
            stagePercentage = (100*result[execVersion]['values'][stageName])/result[execVersion]['total_time']
            print "\t\tStage: %s\tTotal: %0.6f\tPercentage: %0.2f%%" % (stageName, stageTotal, stagePercentage)