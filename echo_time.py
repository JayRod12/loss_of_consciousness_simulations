
import sys
import time


def echo_start(msg):
    t = time.time()
    sys.stdout.write(msg)
    sys.stdout.flush()
    return t

def echo_end(t):
    elapsed = time.time() - t
    sys.stdout.write("Done [{} seconds]\n".format(elapsed))
    sys.stdout.flush()
