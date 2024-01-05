import numpy as np
from math import sqrt

class TinyStatistician:

    @staticmethod
    def __is_invalid_input(x):
        self.x = x
        if not isinstance(x, list):
            return True
        elif not all(isinstance(i, (int, float)) for i in x):
            return True
        elif len(x) == 0:
            return True
        return False

    def mean(self, x):
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
    
    def median(self, x):
        """ computes the median, which is also the 50th percentile, of a
        given nonempty list or array x . The method returns the median as
        a float, otherwise None if x is an empty list or array or a non
        expected type object."""
        if isinstance(x, (np.ndarray, list)) and len(x):
            tmp = x.copy()
            tmp.sort()
            return float(tmp[int(len(tmp) / 2)])
        return None
    
    def quartile(self, x):
        """computes the 1st and 3rd quartiles, also called the 25th percentile 
        and the 75th percentile, of a given non-empty list or array x. The
        method returns the quartiles as a list of 2 floats, otherwise None if
        x is an empty list or array or a non expected type object."""
        if isinstance(x, (np.ndarray, list)) and len(x):
            tmp = x.copy()
            tmp.sort()
            return [float(tmp[int(len(tmp) / 4)]), float(tmp[int(len(tmp) * 3 / 4)])]
        return None
    
    def percentile(self, x, p):
        """computes the expected percentile of a given non-empty list or array x.
        The method returns the percentile as a float, otherwise None if x is an
        empty list or array or a non expected type object. The second parameter
        is the wished percentile."""
        if isinstance(x, (np.ndarray, list)) and len(x) and isinstance(p, (int, float)):
            tmp = x.copy()
            tmp.sort()
            return float(tmp[int((len(tmp) + 1) * p / 100)])
        return None
    
    def var(self,x): 
        """computes the sample variance of a given non-empty list or array x. The
        method returns the sample variance as a float, otherwise None if x is an empty
        list or array or a non expected type object."""
        if isinstance(x, (np.ndarray, list)) and len(x):
            v = 0
            m = self.mean(x)
            for i in x:
                v += (i - m) * (i - m)
            return v / len(x)
        return None
    
    def std(self, x):
        """computes the sample standard deviation of a given non-empty list or array
        x. The method returns the sample standard deviation as a float, otherwise None if
        x is an empty list or array or a non expected type object."""
        if isinstance(x, (np.ndarray, list)) and len(x):
            return sqrt(self.var(x))
        return None
    
a = [1, 42, 300, 10, 59]
print(TinyStatistician().mean(a))
# Output:
# 82.4
print(TinyStatistician().median(a))
# Output:
# 42.0
print(TinyStatistician().quartile(a))
# Output:
# [10.0, 59.0]
print(TinyStatistician().percentile(a, 10))
# Output:
# 4.6
print(TinyStatistician().percentile(a, 15))
# Output:
# 6.4
print(TinyStatistician().percentile(a, 20))
# Output:
# 8.2
print(TinyStatistician().var(a))
# Output:
# 15349.3
print(TinyStatistician().std(a))