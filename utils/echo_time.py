"""
Utility file to display timing information during simulations.

Example usage:

    echo = echo_time('Doing some slow calculation...')
    slow_calculation()
    echo_end(echo)

Output:
    Doing some slow calculation... [12.52s]

"""

import sys
import time

VERBOSE = True
#VERBOSE = False
def echo_start(msg):
    """
        Prints and flushes a message without a new line and returns the time.
    """
    t = time.time()
    if VERBOSE:
        sys.stdout.write(msg)
        sys.stdout.flush()
    return t

def echo_end(t, opt_text=""):
    """
        Called with a time variable (e.g. obtained with echo_start), it prints
        the elapsed time and prints a new line.
    """
    elapsed = time.time() - t
    if VERBOSE:
        sys.stdout.write("{} [{:.2f}s]\n".format(opt_text, elapsed))
        sys.stdout.flush()
