import numpy as np


def gauss_elimination(A, b):
    n = len(b)

    # Etapa de eliminação
    for pivot_row in range(n-1):  # Para cada linha pivô
        for row in range(pivot_row+1, n):  # Para cada linha abaixo da linha pivô
            factor = A[row, pivot_row] / \
                A[pivot_row, pivot_row]  # Fator de eliminação
            A[row, pivot_row:] -= factor * A[pivot_row,
                                             pivot_row:]  # Operação de eliminação
            b[row] -= factor * b[pivot_row]  # Operação de eliminação

    # Etapa de substituição de volta
    x = np.zeros(n) # Vetor de solução
    for i in range(n-1, -1, -1): # Para cada linha, de baixo para cima
        x[i] = (b[i] - np.sum(A[i, i+1:] * x[i+1:])) / A[i, i] # Operação de substituição

    return x


# Sistema de equações a)
A_a = np.array([[3, 2, 4], [1, 1, 2], [4, 3, -2]], dtype=float)
b_a = np.array([1, 2, 3], dtype=float)
x_a = gauss_elimination(A_a, b_a)

print("Solução do sistema a):")
print("x =", x_a[0])
print("y =", x_a[1])
print("z =", x_a[2])

# Sistema de equações b)
A_b = np.array([[3, 2, 0, 1], [9, 8, -3, 4],
               [-6, 4, -8, 0], [3, -8, 3, -4]], dtype=float)
b_b = np.array([3, 6, 16, 18], dtype=float)
x_b = gauss_elimination(A_b, b_b)

print("\nSolução do sistema b):")
print("x =", x_b[0])
print("y =", x_b[1])
print("z =", x_b[2])
print("w =", x_b[3])


# TESTANDO AS SOLUÇÕES

# Sistema de equações a)
print("\nTestando as soluções do sistema a):")
print("3x + 2y + 4z =", 3*x_a[0] + 2*x_a[1] + 4*x_a[2])
print("x + y + 2z =", x_a[0] + x_a[1] + 2*x_a[2])
print("4x + 3y - 2z =", 4*x_a[0] + 3*x_a[1] - 2*x_a[2])

# Sistema de equações b)
print("\nTestando as soluções do sistema b):")
print("3x + 2y + w =", 3*x_b[0] + 2*x_b[1] + x_b[3])
print("9x + 8y - 3z + 4w =", 9*x_b[0] + 8*x_b[1] - 3*x_b[2] + 4*x_b[3])
print("-6x + 4y - 8z =", -6*x_b[0] + 4*x_b[1] - 8*x_b[2])
print("3x - 8y + 3z - 4w =", 3*x_b[0] - 8*x_b[1] + 3*x_b[2] - 4*x_b[3])


"""

Solução do sistema a):
x = -3.0
y = 5.0
z = 5.551115123125783e-17

Solução do sistema b):
x = 2.0
y = 7.0
z = 0.0
w = -17.0

Testando as soluções do sistema a):
3x + 2y + 4z = 1.0000000000000002
x + y + 2z = 2.0
4x + 3y - 2z = 3.0

Testando as soluções do sistema b):
3x + 2y + w = 3.0
9x + 8y - 3z + 4w = 6.0
-6x + 4y - 8z = 16.0
3x - 8y + 3z - 4w = 18.0

"""
