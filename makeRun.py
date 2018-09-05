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

ANALYZE_LOGS = ['pos_io', 'pos_cpu', 'kernels']

# clean
for fileSuffix in ANALYZE_LOGS:
    filename = '.'.join([DATASET, PLATFORM, 'log', fileSuffix, 'csv'])
    file_exists = os.path.isfile(filename)
    if file_exists:
        print 'Removing ' + filename
        os.remove(filename)

# exec N_EXECS times
proc_params = ['make', '.'.join([DATASET, PLATFORM, 'log'])]
for i in range(0, N_EXECS):
    print 'Running #{0}'.format(str(i+1))
    call(proc_params)

# analyze
for fileSuffix in ANALYZE_LOGS:
    filename = '.'.join([DATASET, PLATFORM, 'log', fileSuffix, 'csv'])

    inputFile = csv.DictReader(open(filename))

    result = dict()
    checkedExecutions = []
    # collect measures
    for row in inputFile:
        execVersion = row['CommitHash']
        stageName = row['Name']
        stageValue = float(row['Total'])

        if stageName != 'total':
            if execVersion not in result:
                result[execVersion] = dict()
                result[execVersion]['ATE'] = float(0)
                result[execVersion]['values'] = dict()
                result[execVersion]['num_values'] = 0
                result[execVersion]['total_time'] = float(0)

            execId = row['Timestamp']
            if execId not in checkedExecutions:
                checkedExecutions.append(execId)
                result[execVersion]['num_values'] += 1

            # ATE handler
            if stageName == 'ATE':
                result[execVersion]['ATE'] += stageValue
            # ordinary stage handler
            else:
                if stageName not in result[execVersion]['values']:
                    result[execVersion]['values'][stageName] = stageValue
                else:
                    result[execVersion]['values'][stageName] += stageValue
                result[execVersion]['total_time'] += stageValue

    # print results
    for execVersion in result:
        totalTime = float(0)
        print "Analyzing %s" % filename
        print "\tVersion: %s" % execVersion
        print "\tNumber of execs: %d" % result[execVersion]['num_values']
        for stageName in result[execVersion]['values']:
            stageTotal = result[execVersion]['values'][stageName]/result[execVersion]['num_values']
            stagePercentage = (100*result[execVersion]['values'][stageName])/result[execVersion]['total_time']
            print "\t\tStage: %s\tTotal: %0.6f\tPercentage: %0.2f%%" % (stageName, stageTotal, stagePercentage)
        medTotalTime = result[execVersion]['total_time']/result[execVersion]['num_values']
        print "Sum of all values (total): %0.6f" % medTotalTime
        if result[execVersion]['ATE'] != float(0):
            medATE = result[execVersion]['ATE']/result[execVersion]['num_values']
            print "ATE (accuracy): %0.6f" % medATE
