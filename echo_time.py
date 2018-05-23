
import sys
import time

ECHO_OFF = True
def echo_start(msg):
    t = time.time()
    if not ECHO_OFF:
        sys.stdout.write(msg)
        sys.stdout.flush()
    return t

def echo_end(t):
    elapsed = time.time() - t
    if not ECHO_OFF:
        sys.stdout.write(" [{:.2f}s]\n".format(elapsed))
        sys.stdout.flush()
