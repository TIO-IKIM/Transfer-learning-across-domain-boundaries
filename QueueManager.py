import os, sys, time, signal, traceback
from socket import timeout
import subprocess

class Job:

    def __init__(self, jstr):

        self.jstr = jstr
        try:
            if self.jstr.startswith("#"):
                self.cmd = None
                self.status = None
            else:
                self.cmd = jstr.split(",")[0].split(" ")
                self.status = jstr.split(",")[1]
        except:
            self.cmd = None
            self.status = None

def queue_shutdown(jobfile, jobs):
    os.remove(jobfile)
    with open(jobfile, 'w+') as o:
        for job in jobs:
            if job.cmd is None or job.status is None:
                o.write(job.jstr)
            else:
                o.write(" ".join(job.cmd)+","+job.status+",\n")

jobfile = "/Projects/DATL/Jobs.csv"

with open(jobfile, 'r') as o:
    lines = o.readlines()
jobs = [Job(jstr=jstr) for jstr in lines]

for n, job in enumerate(jobs):
    if job.status == "completed":
        print(f"Job #{n} was previously completed.")
        continue
    if job.cmd is None or job.status is None:
        continue
    
    try:
        print(f"Running job #{n}: {job.jstr.split(',')[0]}")
        time.sleep(30) # Wait for resources to be freed?
        p = subprocess.Popen(job.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while p.poll() is None:
            for line in p.stdout:
                print(line.decode("utf-8"))
            time.sleep(0.2)
        for line in p.stdout.readlines():
            print(line.decode("utf-8"))
        if p.returncode == 0:
            job.status = "completed"
            print(f"Job #{n} completed successfully.")
        else:
            raise RuntimeError(f"Job #{n} failed.")
    except KeyboardInterrupt:
        print("Shutting down queue ...")
        queue_shutdown(jobfile, jobs)
        os.kill(p.pid, signal.SIGINT)
        print("Queue shut down successfully.")
        raise
    except Exception as e:
        print(str(traceback.format_exc()))
        continue

print("Shutting down queue ...")
queue_shutdown(jobfile, jobs)
print("Queue shut down successfully.")