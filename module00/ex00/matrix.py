import numpy as np

class Matrix:
    def __init__(self, data, shape=None):
        arr = np.shape(data)
        rows, cols = shape
        if shape is None:
            shape = len(data)
        data = [col + [0.0] * (cols - arr[1]) for col in data]
        full = [[float(0)] * cols] * ((rows - np.shape(data)[0]))
        self.matrix_data = data+full
    # add : only matrices of same dimensions.
    def __add__(self, other):
        if np.shape(self) != np.shape(other):
            print('ERROR')
        return self.matrix_data + other
    def __radd__(self, other):
        return other + self.matrix_data 
    # sub : only matrices of same dimensions.
    def __sub__(self, other):
        if np.shape(self) != np.shape(other):
            print('ERROR')
        return self.matrix_data - other 
    def __rsub__(self, other):
        if np.shape(self) != np.shape(other):
            print('ERROR')
        return  other - self.matrix_data
    # div : only scalars.
    def __truediv__(self, other):
        if not isinstance(other, (int, float)) or other == 0:
            print('ERROR')
        return self.matrix_data / other
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            print('ERROR')
        result_data = [[other / element for element in row] for row in self.matrix_data]
        return Matrix(result_data)
    # mul : scalars, vectors and matrices , can have errors with vectors and matrices,
    # returns a Vector if we perform Matrix * Vector mutliplication.
    
    def __mul__(self, other):
        if isinstance(other, Matrix):
            # Matrix multiplication
            if self.shape[1] != other.shape[0]:
                raise ValueError("Number of columns in the first matrix must be equal to the number of rows in the second matrix.")
            
            result_data = []
            for i in range(self.shape[0]):
                row_result = [sum(self.data[i][k] * other.data[k][j] for k in range(self.shape[1])) for j in range(other.shape[1])]
                result_data.append(row_result)

            return Matrix(result_data)

        elif isinstance(other, (int, float)):
            # Scalar multiplication
            result_data = [[element * other for element in row] for row in self.data]
            return Matrix(result_data)

        elif isinstance(other, list) and all(isinstance(item, (int, float)) for item in other):
            # Vector multiplication
            if len(other) != self.shape[1]:
                raise ValueError("Length of the vector must be equal to the number of columns in the matrix.")

            result_vector = [sum(self.data[i][j] * other[j] for j in range(self.shape[1])) for i in range(self.shape[0])]
            return result_vector

        else:
            raise TypeError("Multiplication with the provided type is not supported.")

    def __rmul__(self, other):
        # Scalar multiplication on the left
        if isinstance(other, (int, float)):
            result_data = [[element * other for element in row] for row in self.data]
            return Matrix(result_data)
        else:
            raise TypeError("Left multiplication with the provided type is not supported.")

    def __repr__(self):
        return f"Matrix({self.data})"

    def __str__(self):
        return "\n".join([" ".join(map(str, row)) for row in self.data])
    def T(self):
            # Transpose the matrix
            transposed_data = [[self.data[j][i] for j in range(self.shape[0])] for i in range(self.shape[1])]
            return Matrix(transposed_data)

class Vector(Matrix):
    def __init__(self, data):
        super().__init__(data)
        if self.shape[0] != 1 and self.shape[1] != 1:
            raise ValueError("Invalid vector. It must be either a row or a column vector.")
    
    
    def dot(self, v):
        # Check if the shapes match for dot product
        if self.shape != v.shape:
            raise ValueError("Shapes of vectors do not match for dot product.")

        # Calculate dot product
        dot_product = sum(self.data[0][i] * v.data[0][i] for i in range(self.shape[1]))
        return dot_product


m1 = Matrix([[1.0, 2.0], [3.0, 4.0]], [3, 4])
print(m1.matrix_data)
