
import sys
import time

ECHO_OFF = False
def echo_start(msg):
    t = time.time()
    if not ECHO_OFF:
        sys.stdout.write(msg)
        sys.stdout.flush()
    return t

def echo_end(t, opt_text=""):
    elapsed = time.time() - t
    if not ECHO_OFF:
        sys.stdout.write("{} [{:.2f}s]\n".format(opt_text, elapsed))
        sys.stdout.flush()
