from scipy.signal import welch
import numpy as np

# y = f(t)
def moving_average(t, dt, shift, total_steps, return_time=False):
    if len(t) == 0:
        return []

    N = len(t)
    MA = []
    Time = []
    t0 = 0
    t1 = t0 + dt
    l = 0

    while l < N:
        # Find next interval [t0, t1)

        # Find first index >= t0
        if t[l] < t0:
            while l + 1 < N and t[l + 1] < t0:
                l += 1
            l = l + 1

        if l == N:
            break

        # If first index >= t1, then it's outside the slot
        if t[l] >= t1:
            ma = 0
        else:
            # Find the last index inside the time slot
            r = l
            while r + 1 < N and t[r + 1] < t1:
                r += 1
            ma = r-l+1
        #print("[{}, {}) -> [{},{}]: {}".format(t0, t1, t[l], max(t[l], t[r]), ma))

        MA.append(ma)
        Time.append((t0, t1))
        t0 = t0 + shift
        t1 = t0 + dt

    MA = np.array(MA, dtype=np.float)
    if total_steps > len(MA):
        # Pad array with zeros at the end until a length of 'total_steps'
        MA = np.pad(MA, (0, total_steps-len(MA)), 'constant')
        for step in range(total_steps-len(MA)):
            Time.append((t0, t1))
            t0 = t0 + shift
            t1 = t0 + dt

    # Calculate # spikes / time step i.e. the average
    MA /= dt
    if return_time:
        return (MA, Time)
    else:
        return MA

def power_spectrum(t, dt, shift, total_steps):
    ma = moving_average(t, dt, shift, total_steps)
    # e.g. Shift = 20ms
    # Therefore the sampling frequency is 1/20ms = 1 / 0.02s = 50Hz
    return welch(ma, int(1.0/(float(shift)/1000)))

