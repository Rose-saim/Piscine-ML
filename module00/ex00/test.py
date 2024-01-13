from matrix import Matrix, Vector


m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
print(m1.shape)
# Output:
# (3, 2)
print(m1.T())
# Output:
# Matrix([[0., 2., 4.], [1., 3., 5.]])
print(m1.T().shape)
# Output:
# (2, 3)
m1 = Matrix([[0., 2., 4.], [1., 3., 5.]])
print(m1.shape)
# Output:
# (2, 3)
print(m1.T())
# Output:
# Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
print(m1.T().shape)
# Output:
# (3, 2)
m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
[0.0, 2.0, 4.0, 6.0]])
m2 = Matrix([[0.0, 1.0],
[2.0, 3.0],
[4.0, 5.0],
[6.0, 7.0]])
print(m1 * m2)
# Output:
# Matrix([[28., 34.], [56., 68.]])
m1 = Matrix([[0.0, 1.0, 2.0],
[0.0, 2.0, 4.0]])
v1 = Vector([[1], [2], [3]])
print(m1 * v1)
# Output:
# Matrix([[8], [16]])
# Or: Vector([[8], [16]
v1 = Vector([[1], [2], [3]])
v2 = Vector([[2], [4], [8]])
print(v1 + v2)
# Output:
# Vector([[3],[6],[11]])