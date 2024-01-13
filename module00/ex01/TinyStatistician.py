import unittest


class TinyStatistician:

    @staticmethod
    def __is_invalid_input(x):
        if not isinstance(x, list):
            return True
        elif not all(isinstance(i, (int, float)) for i in x):
            return True
        elif len(x) == 0:
            return True
        return False

    def mean(self, x) -> float:
        """
        Computes the mean of a given non-empty list of int or float.
        """
        if self.__is_invalid_input(x):
            return None
        total = 0
        for i in x:
            total += i
        return float(total / len(x))

    def median(self, x) -> float:
        """
        Computes the median of a given non-empty list of int or float.
        """
        if self.__is_invalid_input(x):
            return None
        x.sort()
        middle = len(x) // 2
        if len(x) % 2 == 0:
            return (x[middle - 1] + x[middle]) / 2
        else:
            return float(x[middle])

    def quartile(self, x) -> 'list[float]':
        """
        Computes the quartiles Q1 and Q3 of a given non-empty list of
        int or float.
        """
        if self.__is_invalid_input(x):
            return None
        x.sort()
        middle = len(x) // 2
        Q1_index = middle // 2
        Q3_index = middle + Q1_index
        if len(x) % 2 == 0:
            Q1 = (x[Q1_index - 1] + x[Q1_index]) / 2
            Q3 = (x[Q3_index - 1] + x[Q3_index]) / 2
        else:
            Q1 = x[Q1_index]
            Q3 = x[Q3_index]
        return [float(Q1), float(Q3)]

    def percentile(self, x, percentile) -> float:
        """
        Computes the percentile p of a given non-empty list of int or float.
        Note:
        uses a different definition of percentile, it does linear
        interpolation between the two closest list element to the percentile.
        """
        if self.__is_invalid_input(x):
            return None
        elif not isinstance(percentile, int):
            return None
        elif percentile < 0 or percentile > 100:
            return None
        x.sort()
        length = len(x)
        percentile_index = (length - 1) * percentile / 100
        floor_index = int(percentile_index)
        diff = percentile_index - floor_index
        return (x[floor_index]
                + ((x[floor_index + 1] - x[floor_index])
                   / 100 * diff * 100))

    def var(self, x) -> float:
        """
        Computes the variance of a given non-empty list of int or float.
        Note:
        uses the unbiased estimator (divides by n - 1).
        """
        if self.__is_invalid_input(x):
            return None
        mean = TinyStatistician.mean(self, x)
        diff = sum((i - mean) ** 2 for i in x)
        return float(diff / (len(x) - 1))

    def std(self, x):
        """
        Computes the standard deviation of a given non-empty list of
        int or float.
        """
        var = TinyStatistician.var(self, x)
        if var is None:
            return None
        return var ** 0.5

