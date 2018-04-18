from scipy.signal import welch
import numpy as np

# y = f(t)
# Returns the number of events per time unit averaged over a window of size `dt` and
# at every time point separated by `shift` time units.
# Parameters
#   In:
#       - a: [float] - Array of floats representing the times at which an event occurs
#       - dt: int - Window of time over which events are counted
#       - shift: int - Time shift between every measurement
#       - start: int - Number of time units at which the measurement starts
#       - end: int - Number of time units at which the measurement ends
#   Out:
#       - output: [float] - Moving average of the rate of events. Represents number of 
#                           events per time unit calculated as an average over the window.
#       - ts: [float] - Time steps representing the start time at which every output
#                       measurement is recorded.
def moving_average(a, dt, shift, start, end):
    assert len(a) > 0

    # time steps
    ts = np.linspace(start, end, (end - start) / shift, endpoint=False)
    output = np.zeros(len(ts), dtype=np.float)

    N, M = len(a), len(ts)

    i, j = 0, 0

    while i < N and j < M:
        # Find all spikes in interval [ts[j], ts[j] + dt)
        t0, t1 = ts[j], ts[j] + dt

        # Find first index >= t0
        if a[i] < t0:
            while i + 1 < N and a[i + 1] < t0:
                i += 1
            i += 1

        if i == N:
            break

        # If first index >= t1, then it's outside the current slot
        if a[i] >= t1:
            output[j] = 0
        else:
            # Find the last index inside the time slot
            k = i
            while k + 1 < N and a[k + 1] < t1:
                k += 1
            output[j] = k - i + 1
        j += 1
        #print("[{}, {}) -> [{},{}]: {}".format(t0, t1, t[l], max(t[l], t[r]), ma))

    # Calculate # spikes / time step i.e. the average
    output /= dt
    return output, ts

def power_spectrum(a, dt, shift, start, end):
    ma, _ = moving_average(a, dt, shift, start, end)
    # e.g. Shift = 20ms
    # Therefore the sampling frequency is 1/20ms = 1 / 0.02s = 50Hz
    return welch(ma, int(1.0/(float(shift)/1000)))

