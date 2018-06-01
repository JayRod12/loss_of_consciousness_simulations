
import sys
import time

VERBOSE = True
def echo_start(msg):
    t = time.time()
    if VERBOSE:
        sys.stdout.write(msg)
        sys.stdout.flush()
    return t

def echo_end(t, opt_text=""):
    elapsed = time.time() - t
    if VERBOSE:
        sys.stdout.write("{} [{:.2f}s]\n".format(opt_text, elapsed))
        sys.stdout.flush()
