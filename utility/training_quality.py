import numpy as np
import scipy as sp
from scipy import stats
from typing import List
from tqdm.auto import tqdm

def variance_from_linear(x: List, y: List):
    slope, intercept, _, p_value, _ = sp.stats.linregress(x, y)
    y_exp = [slope * x_i + intercept for x_i in x]
    avg_residual = np.mean([abs(y_exp_i - y_i) for y_exp_i, y_i in zip(y_exp, y)])
    return avg_residual, p_value

def instability(x: List, y: List, s: int = 5):
    """
    Return a measure of smoothness for a training run, given a segment_length s, which is assumed to be linear.
    Standard deviation for the points on every successive segment is calculated and finally averaged.
    """

    assert len(x) == len(y)
    assert len(x) > s + 2

    sum_ar = 0
    n_segments = len(x) - s + 1

    for i in tqdm(range(n_segments), total = n_segments):
        ar, _ = variance_from_linear(x = x[i:i+s], y = y[i:i+s])
        sum_ar += ar

    return sum_ar / n_segments

def time_to_convergence(x: List, y: List, s: int = 5):
    """
    Assuming a segment length of s, test iteratively whether the final
    area of the curve is flat (has converged). If the slope of a linear
    fit through these points is within 3 sigma of 0, assume convergence.
    Iteratively move the segment to the left, starting on the right, and
    test again. The last point for which convergence is given is the time
    to convergence.
    """
    assert len(x) == len(y)

    tau = len(x) - s

    for i in tqdm(range(len(x) - s + 1)):
        if i != 0:
            slope, _, _, _, slope_err = sp.stats.linregress(x[-s-i:-i-1], y[-s-i:-i-1])
        else:
            slope, _, _, _, slope_err = sp.stats.linregress(x[-s-i:], y[-s-i:])
        if abs(slope) >= 3 * slope_err: # is not flat
            print(i, x[-s-i], y[-s-i], abs(slope), slope_err)
            tau -= (i-1)
            if tau == len(x) - s - i + 1:
                return tau, True
            else:
                return tau, False

    # Return time to convergence, and whether there is a need to worry because nowhere was flat or everywhere was flat
    return tau, True

if __name__ == "__main__":
    
    ar, p = variance_from_linear([1, 2, 3, 4, 5], [1, 2.1, 3, 3.9, 5])
    print(ar, p)

    ins = instability(x = [1, 2, 3, 4, 5, 6, 7, 8, 9], y = [1, 2.1, 3, 3.9, 5, 6.5, 7.1, 7.5, 8.8])
    print(ins)

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    y = [1, 2, 3, 4, 4.8, 5.5, 6.3, 6.6, 6.8, 7.0, 7.1, 7.0, 6.9, 7.1, 7.0, 7.0]
    print(time_to_convergence(x = x, y = y))