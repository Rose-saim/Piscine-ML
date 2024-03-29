class Matrix:

    # The matrix object can be initialized with either:
    # - the elements of the matrix as a list of lists
    #   Matrix([[1.0, 2.0], [3.0, 4.0]])
    # - the shape of the matrix as a tuple
    #   Matrix((2, 2)) -> [[0.0, 0.0], [0.0, 0.0]]
    def __init__(self,
                 data: 'list[list[float]]' = None,
                 shape: 'tuple[int, int]' = None):
        if data is None and shape is None:
            raise TypeError(
                "Matrix() missing 1 required argument : 'data' or 'shape'")
        elif data is not None and shape is not None:
            raise TypeError(
                "Matrix() takes 1 positional argument but 2 were given")
        if data is not None:
            self.__init_by_data(data)
        elif shape is not None:
            self.__init_by_shape(shape)

    def __init_by_data(self, data):
        if (not isinstance(data, list)
                or not all(isinstance(x, list) for x in data)):
            raise TypeError("Data must be a list of lists")
        elif len(data) == 0:
            raise ValueError("Data must not be empty")
        elif not all(len(x) == len(data[0]) for x in data):
            raise ValueError(
                "Data must be a matrix, all rows must have the same length")
        elif len(data[0]) == 0:
            raise ValueError("Data must not be empty")
        elif not all(
                isinstance(x, (int, float)) for row in data for x in row):
            raise TypeError("Data must contain only integers or floats")
        self.data = [[float(x) for x in row] for row in data]
        self.shape = (len(data), len(data[0]))
        if self.shape[0] == 1 or self.shape[1] == 1:
            self.__class__ = Vector

    def __init_by_shape(self, shape):
        if not isinstance(shape, tuple):
            raise TypeError("Shape must be a tuple")
        elif len(shape) != 2:
            raise ValueError("Shape must be a tuple of length 2")
        elif not all(isinstance(x, int) for x in shape):
            raise TypeError("Shape must contain only integers")
        elif not all(x > 0 for x in shape):
            raise ValueError("Shape must contain only positive integers")
        self.data = [[0.0 for __ in range(shape[1])]
                     for _ in range(shape[0])]
        self.shape = shape
        if self.shape[0] == 1 or self.shape[1] == 1:
            self.__class__ = Vector

    # add : only matrices/vectors of same dimensions.
    # __add__
    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Can only add a Matrix to a Matrix")
        elif self.shape != other.shape:
            raise ValueError("Can only add matrices of same shape")
        return Matrix([[a + b for a, b in zip(x, y)]
                       for x, y in zip(self.data, other.data)])

    # __radd__
    def __radd__(self, other):
        return self + other

    # sub : only matrices/vectors of same dimensions.
    # __sub__
    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Can only subtract a Matrix from a Matrix")
        elif self.shape != other.shape:
            raise ValueError("Can only subtract matrices of same shape")
        return Matrix([[a - b for a, b in zip(x, y)]
                       for x, y in zip(self.data, other.data)])

    # __rsub__
    def __rsub__(self, other):
        return self - other

    # Multplies a matrix by a vector.
    def mul_by_vector(self, vector):
        if not isinstance(vector, Vector):
            raise TypeError("Can only multiply a Matrix by a Vector")
        elif self.shape[1] != vector.shape[0]:
            raise ValueError("Matrix and vector shapes do not match")
        data = []
        for i in range(self.shape[0]):
            val = 0
            for j in range(self.shape[1]):
                val += self.data[i][j] * vector.data[j][0]
            data.append([val])
        return Vector(data)

    # Multiplies a matrix by a scalar.
    def scale(self, factor):
        if not isinstance(factor, (int, float)):
            raise TypeError("Can only scale a Matrix by a scalar")
        return Matrix([[x * factor for x in row] for row in self.data])

    # Multiplies a matrix by a matrix.
    def mul_by_matrix(self, matrix):
        if not isinstance(matrix, Matrix):
            raise TypeError("Can only multiply a Matrix by a Matrix")
        elif self.shape[1] != matrix.shape[0]:
            raise ValueError("Matrix shapes do not match")
        data = []
        for i in range(self.shape[0]):
            row = []
            for j in range(matrix.shape[1]):
                val = 0
                for k in range(self.shape[1]):
                    val += self.data[i][k] * matrix.data[k][j]
                row.append(val)
            data.append(row)
        return Matrix(data)

    # mul : scalars, vectors and matrices,
    # can have errors with vectors and matrices,
    # returns a Vector if we perform Matrix * Vector mutliplication.
    # __mul__
    def __mul__(self, other):
        if isinstance(other, Vector):
            return self.mul_by_vector(other)
        elif isinstance(other, Matrix):
            return self.mul_by_matrix(other)
        elif isinstance(other, (int, float)):
            return self.scale(other)
        else:
            raise TypeError("Can only multiply a Matrix by a scalar, "
                            "a Vector or a Matrix")

    # __rmul__
    def __rmul__(self, other):
        return self * other

    # div : only scalars.
    # __truediv__
    def __truediv__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError("Can only divide a Matrix by a scalar")
        elif other == 0:
            raise ZeroDivisionError("Cannot divide a Matrix by 0")
        return self * (1 / other)

    # __rtruediv__
    def __rtruediv__(self, other):
        raise TypeError("Cannot divide a scalar by a Matrix")

    # __str__ : print the matrix in a nice way.
    def __str__(self):
        ret = type(self).__name__ + "(\n"
        for row in self.data:
            ret += " " + str(row) + "\n"
        ret += ")"
        return ret

    # __repr__
    # (More precise than __str__)
    def __repr__(self):
        if self.shape[0] == 1 or self.shape[1] == 1:
            return "Vector(" + str(self.data) + ")"
        else:
            return "Matrix(" + str(self.data) + ")"

    # Transpose the matrix
    def T(self):
        if self.shape[0] == 0 or self.shape[0] == 1:
            return Matrix(self.shape)
        res = [[self.data[row][column] for row in range(len(self.data))] for column in range(len(self.data[0]))]
        return Matrix(res)

    # ==
    def __eq__(self, other) -> bool:
        if isinstance(other, Matrix):
            return (self.data == other.data
                    and self.shape == other.shape)
        return False

    # !=
    def __ne__(self, other) -> bool:
        return not self == other


class Vector(Matrix):

    """
    A vector is a matrix with only one column.
    """

    def __init__(self,
                 data: 'list[list[float]]' = None,
                 shape: 'tuple[int, int]' = None):
        super().__init__(data=data, shape=shape)
        if self.shape[0] != 1 and self.shape[1] != 1:
            raise ValueError("A vector must have only one column")

    def dot(self, other):
        if not isinstance(other, Vector):
            raise TypeError("Can only dot a Vector with a Vector")
        elif self.shape != other.shape:
            raise ValueError("Can only dot vectors of same shape")
        return sum(self.data[i][j] * other.data[i][j]
                   for i in range(self.shape[0])
                   for j in range(self.shape[1]))

    # Cross product
    def cross(self, other):
        if not isinstance(other, Vector):
            raise TypeError("Can only cross a Vector with a Vector")
        elif self.shape != other.shape:
            raise ValueError("Can only cross vectors of same shape")
        if self.shape == (1, 3):
            return (self.T().cross(other.T())).T()
        elif self.shape == (3, 1):
            return Vector([
                [self.data[1][0] * other.data[2][0]
                 - self.data[2][0] * other.data[1][0]],
                [self.data[2][0] * other.data[0][0]
                 - self.data[0][0] * other.data[2][0]],
                [self.data[0][0] * other.data[1][0]
                 - self.data[1][0] * other.data[0][0]]
            ])
        raise ValueError("Can only cross vectors of shape (3, 1) or (1, 3)")
