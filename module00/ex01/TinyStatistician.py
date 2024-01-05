import numpy as np
from math import sqrt

class TinyStatistician:
    def mean(x: list):
        """computes the mean of a given non-empty list or array x,
        using a for-loop. The method returns the mean as a float, otherwise
        None if x is an empty list or array, or a non expected type object."""
        if isinstance(x, (np.ndarray, list)) and len(x):
            total, count = 0, 0
            for n in x:
                total += n
                count += 1
            return float(total / count)
        return None
    def median(x: list):
        """ computes the median, which is also the 50th percentile, of a
        given nonempty list or array x . The method returns the median as
        a float, otherwise None if x is an empty list or array or a non
        expected type object."""
        if isinstance(x, (np.ndarray, list)) and len(x):
            tmp = x.copy()
            tmp.sort()
            return float(tmp[int(len(tmp) / 2)])
        return None
    def quartile(x: list):
        """computes the 1st and 3rd quartiles, also called the 25th percentile 
        and the 75th percentile, of a given non-empty list or array x. The
        method returns the quartiles as a list of 2 floats, otherwise None if
        x is an empty list or array or a non expected type object."""
        if isinstance(x, (np.ndarray, list)) and len(x):
            tmp = x.copy()
            tmp.sort()
            return [float(tmp[int(len(tmp) / 4)]), float(tmp[int(len(tmp) * 3 / 4)])]
        return None
    def percentile(x: list, p):
        """computes the expected percentile of a given non-empty list or array x.
        The method returns the percentile as a float, otherwise None if x is an
        empty list or array or a non expected type object. The second parameter
        is the wished percentile."""
        if isinstance(x, (np.ndarray, list)) and len(x) and isinstance(p, (int, float)):
            tmp = x.copy()
            tmp.sort()
            return float(tmp[int((len(tmp) + 1) * p / 100)])
        return None
    def var(x: list): 
        """computes the sample variance of a given non-empty list or array x. The
        method returns the sample variance as a float, otherwise None if x is an empty
        list or array or a non expected type object."""
        if isinstance(x, (np.ndarray, list)) and len(x):
            v = 0
            m = TinyStatistician.mean(x)
            for i in x:
                v += (i - m) * (i - m)
            return v / len(x)
        return None
    def std(x: list):
        """computes the sample standard deviation of a given non-empty list or array
        x. The method returns the sample standard deviation as a float, otherwise None if
        x is an empty list or array or a non expected type object."""
        if isinstance(x, (np.ndarray, list)) and len(x):
            return sqrt(TinyStatistician.var(x))
        return None