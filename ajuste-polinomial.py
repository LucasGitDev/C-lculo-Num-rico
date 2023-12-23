import numpy as np

# Dados fornecidos
x = np.array([1, 2, 3, 4, 5])
y = np.array([3.5, 2.0, 1.1, 2.3, 4.1])

# Criando as matrizes necessárias
n = len(x)
A = np.vstack([x**2, x, np.ones(n)]).T
B = np.vstack([y]).T

print("Matriz A:")
print(A)
print("\nMatriz B:")
print(B)

# Resolvendo o sistema de equações normais
coefficients = np.linalg.lstsq(A, B, rcond=None)[0]


# Extraindo os coeficientes
a, b, c = coefficients.flatten()

# Imprimindo os resultados
print("Coeficientes:")
print(f"a = {a}")
print(f"b = {b}")
print(f"c = {c}")

# Apresentando a expressão final
print("\nModelo de ajuste polinomial de segundo grau:")
print(f"Y = {a:.2f}x^2 + {b:.2f}x + {c:.2f}")

