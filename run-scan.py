#!/usr/bin/env python3

import os
import re
import json
import time
import argparse
import subprocess
import multiprocessing

# Number of events for each application
n_events_unit = 1000
n_blocks_per_stream = {
    "fwtest": 1,
    "cuda": {"": 100, "transfer": 100},
    "cudauvm": {"": 100, "transfer": 100},
    "cudacompat": {"": 8},
}

# 30 ev/s * 8 hours should the sufficent and fit into signed int for ~2k threads
background_events_per_thread = 30*3600*8

result_re = re.compile("Processed (?P<events>\d+) events in (?P<time>\S+) seconds, throughput (?P<throughput>\S+) events/s")


def printMessage(*args):
    print(time.strftime("%y-%m-%d %H:%M:%S"), *args)

def throughput(output):
    for line in output:
        m = result_re.search(line)
        if m:
            printMessage(line.rstrip())
            return (float(m.group("throughput")), float(m.group("time")))

    raise Exception("Did not find throughput from the log")

def partition_cores(cores, nth):
    if nth >= len(cores):
        return (cores, [])

    return (cores[0:nth], cores[nth:])

def run(nev, nstr, cores_main, opts, logfilename):
    nth = len(cores_main)
    with open(logfilename, "w") as logfile:
        taskset = []
        nvprof = []
        command = [opts.program, "--maxEvents", str(nev), "--numberOfStreams", str(nstr), "--numberOfThreads", str(nth)] + opts.args
        if opts.taskset:
            taskset = ["taskset", "-c", ",".join(cores_main)]

        logfile.write(" ".join(taskset+command))
        logfile.write("\n----\n")
        logfile.flush()
        if opts.dryRun:
            print(" ".join(taskset+command))
            return (0, 0)
        p = subprocess.Popen(taskset+command, stdout=logfile, stderr=subprocess.STDOUT, universal_newlines=True)
        try:
            p.wait()
        except KeyboardInterrupt:
            try:
                p.terminate()
            except OSError:
                pass
            p.wait()
        if p.returncode != 0:
            raise Exception("Got return code %d, see output in the log file %s" % (p.returncode, logfilename))
    with open(logfilename) as logfile:
        return throughput(logfile)

def launchBackground(opts, cores_bkg, logfile):
    if opts.fill <= 0:
        return None
    nth = len(cores_bkg)
    if nth == 0:
        return None
    nev = background_events_per_thread * nth
    taskset = []
    exe = os.path.join(os.path.dirname(opts.program), "cudacompat")
    command = [exe, "--maxEvents", str(nev), "--numberOfThreads", str(nth)]
    if opts.taskset:
        taskset = ["taskset", "-c", ",".join(cores_bkg)]
    if opts.bkgNice is not None:
        taskset.extend(["nice", "-n", str(opts.bkgNice)])

    logfile.write(" ".join(taskset+command))
    logfile.write("\n----\n")
    logfile.flush()
    if opts.dryRun:
        print(" ".join(taskset+command))
        return None
    cudacompat = subprocess.Popen(taskset+command, stdout=logfile, stderr=subprocess.STDOUT, universal_newlines=True)
    return cudacompat

def main(opts):
    ncores = multiprocessing.cpu_count()
    if opts.fill > 0:
        ncores = opts.fill

    if len(opts.tasksetCores) > 0:
        cores = opts.tasksetCores[:]
    else:
        cores = [str(x) for x in range(0, ncores)]
    maxThreads = len(cores)
    if opts.maxThreads > 0:
        maxThreads = min(maxThreads, opts.maxThreads)

    nthreads = range(opts.minThreads,maxThreads+1)
    if len(opts.numThreads) > 0:
        nthreads = [x for x in opts.numThreads if x >= opts.minThreads and x <= maxThreads]
    n_streams_threads = [(i, i) for i in nthreads]
    if len(opts.numStreams) > 0:
        n_streams_threads = [(s, t) for t in nthreads for s in opts.numStreams]

    nev_per_stream = opts.eventsPerStream
    if nev_per_stream is None:
        tmp = n_blocks_per_stream.get(os.path.basename(opts.program), None)
        if tmp is None:
            raise Exception("No default number of event blocks for program %s, and --eventsPerStream was not given" % opts.program)
        if isinstance(tmp, dict):
            if "--transfer" in opts.args:
                eventBlocksPerStream = tmp["transfer"]
            else:
                eventBlocksPerStream = tmp[""]
        else:
            eventBlocksPerStream = tmp
        nev_per_stream = eventBlocksPerStream * n_events_unit

    data = dict(
        program=opts.program,
        args=" ".join(opts.args),
        results=[]
    )
    outputJson = opts.output+".json"
    alreadyExists = set()
    if not opts.overwrite and os.path.exists(outputJson):
        with open(outputJson) as inp:
            data = json.load(inp)
    if not opts.append:
        for res in data["results"]:
            alreadyExists.add( (res["streams"], res["threads"]) )

    stop = False

    for nstr, nth in n_streams_threads:
        if nstr == 0:
            nstr = nth
        if (nstr, nth) in alreadyExists:
            continue

        if opts.maxStreamsToAddEvents > 0 and nstr > opts.maxStreamsToAddEvents:
            nev = nev_per_stream * opts.maxStreamsToAddEvents
        else:
            nev = nev_per_stream*nstr
        (cores_main, cores_bkg) = partition_cores(cores, nth)

        if opts.warmup:
          printMessage("Warming up")
          run(nev, nstr, cores_main, opts, opts.output+"_warmup.txt")
          print()
          opts.warmup = False

        with open(opts.output+"_log_nstr{}_nth{}_bkg.txt".format(nstr, nth), "w") as bkglogfile:
            backgroundJob = launchBackground(opts, cores_bkg, bkglogfile)
            if backgroundJob is not None:
                msg = "Background cudacompat pid {}".format(backgroundJob.pid)
                if opts.taskset:
                    msg +=", running on cores " + ",".join(cores_bkg)
                printMessage(msg)

            try:
                msg = "Number of streams {} threads {} events {}".format(nstr, nth, nev)
                if opts.taskset:
                    msg += ", running on cores " + ",".join(cores_main)
                printMessage(msg)
                throughputs = []
                for i in range(opts.repeat):
                    tryAgain = opts.tryAgain
                    while tryAgain > 0:
                        try:
                            (th, wtime) = run(nev, nstr, cores_main, opts, opts.output+"_log_nstr{}_nth{}_n{}.txt".format(nstr, nth, i))
                            break
                        except Exception as e:
                            tryAgain -= 1
                            if tryAgain == 0:
                                raise
                            print("Got exception (see below), trying again ({} times left)".format(tryAgain))
                            print("--------------------")
                            print(str(e))
                            print("--------------------")
            finally:
                if backgroundJob is not None:
                    printMessage("Run complete, terminating background cudacompat")
                    try:
                        backgroundJob.terminate()
                    except OSError:
                        pass
                    backgroundJob.wait()

            if opts.dryRun:
                continue
            throughputs.append(th)
            data["results"].append(dict(
                threads=nth,
                streams=nstr,
                events=nev,
                throughput=th
            ))
            # Save results after each test
            with open(outputJson, "w") as out:
                json.dump(data, out, indent=2)
            if opts.stopAfterWallTime > 0 and wtime > opts.stopAfterWallTime:
                stop = True
                break

        thr = 0
        if len(throughputs) > 0:
            thr = sum(throughputs)/len(throughputs)
        printMessage("Number of streams %d threads %d, average throughput %f" % (nstr, nth, thr))
        print()
        if stop:
            print("Reached max wall time of %d s, stopping scan" % opts.stopAfterWallTime)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a scan of a given test program")
    parser.add_argument("program", type=str,
                        help="Path to the test program to run")
    parser.add_argument("-o", "--output", type=str, default="result",
                        help="Prefix of output JSON and log files. If the output JSON file exists, it will be updated (see also --overwrite) (default: 'result')")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite the output JSON instead of updating it")
    parser.add_argument("--append", action="store_true",
                        help="Append new (stream, threads) results insteads of ignoring already existing point")
    parser.add_argument("--taskset", action="store_true",
                        help="Use taskset to explicitly set the cores where to run on")
    parser.add_argument("--tasksetCores", type=str, default="",
                        help="Comma-separated list of cores to be used for taskset in that order. Default (empty) is to use range(0, N(cores))")
    parser.add_argument("--fill", type=int, default=-1,
                        help="Launch cudacompat program in the background so that this many threads are always running. If given, this will also become the upper limit for the number of threads instead of the number of cores of the machine. (default: -1 to disable")
    parser.add_argument("--bkgNice", type=int, default=None,
                        help="If given, use this 'nice' level for the background program")
    parser.add_argument("--minThreads", type=int, default=1,
                        help="Minimum number of threads to use in the scan (default: 1)")
    parser.add_argument("--maxThreads", type=int, default=-1,
                        help="Maximum number of threads to use in the scan (default: -1 for the number of cores)")
    parser.add_argument("--numThreads", type=str, default="",
                        help="Comma separated list of numbers of threads to use in the scan (default: empty for all)")
    parser.add_argument("--numStreams", type=str, default="",
                        help="Comma separated list of numbers of streams to use in the scan (default: empty for always the same as the number of threads). If both number of threads and number of streams have more than 1 element, a 2D scan is done with all the combinations")
    parser.add_argument("--eventsPerStream", type=int, default=None,
                        help="Number of events to be used per EDM stream (default: 400*4kev for cuda, others also hardcoded in the top of the script file)")
    parser.add_argument("--maxStreamsToAddEvents", type=int, default=-1,
                        help="Maximum number of streams to add events (default: -1 for no limit")
    parser.add_argument("--stopAfterWallTime", type=int, default=-1,
                        help="Stop running after the wall time of the job reaches this many in seconds (default: -1 for no limit)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat each point this many times (default: 1)")
    parser.add_argument("--tryAgain", type=int, default=1,
                        help="In case of failure on a point, try again at most this many times (default: 1)")
    parser.add_argument("--warmup", action="store_true",
                        help="Run the command once before starting the profiling")
    parser.add_argument("--dryRun", action="store_true",
                        help="Print out commands, don't actually run anything")

    parser.add_argument("args", nargs=argparse.REMAINDER)

    opts = parser.parse_args()
    if opts.minThreads <= 0:
        parser.error("minThreads must be > 0, got %d" % opts.minThreads)
    if opts.maxThreads <= 0 and opts.maxThreads != -1:
        parser.error("maxThreads must be > 0 or -1, got %d" % opts.maxThreads)
    if opts.numThreads != "":
        opts.numThreads = [int(x) for x in opts.numThreads.split(",")]
    if opts.numStreams != "":
        opts.numStreams = [int(x) for x in opts.numStreams.split(",")]
    if opts.tasksetCores != "":
        opts.tasksetCores = opts.tasksetCores.split(",")
    if len(opts.tasksetCores) > 0 and opts.fill != -1 and len(opts.tasksetCores) != opts.fill:
        parser.error("When both --tasksetCores and --fill are given, --fill must match to the number of elements in --tasksetCores. No got --fill {} and {} elements in --tasksetCores {}".format(opts.fill, len(opts.tasksetCores)))

    main(opts)
