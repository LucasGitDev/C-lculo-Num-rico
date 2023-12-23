import numpy as np

solucao_final = np.array([1, -2, 1], dtype=float)


def gaus_jacobi():

    def gauss_jacobi(A, b, x0, epsilon, max_iterations):
        n = len(b)
        x = np.copy(x0)  # Vetor de solução inicial

        for iteration in range(max_iterations):
            x_old = np.copy(x)  # Copia o vetor de solução anterior
            for i in range(n):  # Para cada linha da matriz
                # Somatório dos elementos da linha
                sigma = np.dot(A[i, :n], x_old[:n])
                x[i] = (b[i] - sigma + A[i, i] * x_old[i]) / A[i, i]

            if np.linalg.norm(x - x_old, np.inf) < epsilon:  # Critério de parada
                return x  # Retorna o vetor de solução

        return x

    # Sistema de equações
    A = np.array([[10, 2, 1], [1, 5, 1], [2, 3, 10]], dtype=float)
    b = np.array([7, -8, 6], dtype=float)

    # Vetor de solução inicial
    x0 = np.array([0, 0, 0], dtype=float)

    # Critério de parada e número máximo de iterações
    epsilon = 0.01
    max_iterations = 10

    # Resolver o sistema utilizando o método de Gauss-Jacobi
    solution = gauss_jacobi(A, b, x0, epsilon, max_iterations)

    # Imprimir a solução
    print("Solução do sistema: | Método de Gauss-Jacobi |")
    print("x =", round(solution[0], 4))
    print("y =", round(solution[1], 4))
    print("z =", round(solution[2], 4))

    x = round(solution[0], 4)
    y = round(solution[1], 4)
    z = round(solution[2], 4)

    print("Avaliando a solução:")
    print("10x + 2y + z =", 10*x + 2*y + z)
    print("x + 5y + z =", x + 5*y + z)
    print("2x + 3y + 10z =", 2*x + 3*y + 10*z)

    print("Diferença entre a solução encontrada e a solução real:")
    print("x =", x - solucao_final[0])
    print("y =", y - solucao_final[1])
    print("z =", z - solucao_final[2])


def gaus_seidel():

    def gauss_seidel(A, b, x0, epsilon, max_iterations):
        n = len(b)
        x = np.copy(x0)

        for iteration in range(max_iterations):
            x_old = np.copy(x)  # Copia o vetor de solução anterior
            for i in range(n):
                # Somatório dos elementos anteriores a x[i]
                sigma1 = np.dot(A[i, :i], x[:i])
                # Somatório dos elementos posteriores a x[i]
                sigma2 = np.dot(A[i, i+1:], x_old[i+1:])
                # Calcula o novo valor de x[i]
                x[i] = (b[i] - sigma1 - sigma2) / A[i, i]

            if np.linalg.norm(x - x_old, np.inf) < epsilon:  # Critério de parada
                return x

        return x

    # Sistema de equações
    A = np.array([[10, 2, 1], [1, 5, 1], [2, 3, 10]], dtype=float)
    b = np.array([7, -8, 6], dtype=float)

    # Vetor de solução inicial
    x0 = np.array([0, 0, 0], dtype=float)

    # Critério de parada e número máximo de iterações
    epsilon = 0.01
    max_iterations = 10

    # Resolver o sistema utilizando o método de Gauss-Seidel
    solution = gauss_seidel(A, b, x0, epsilon, max_iterations)

    # Imprimir a solução
    print("Solução do sistema: | Método de Gauss-Seidel |")
    print("x =", round(solution[0], 4))
    print("y =", round(solution[1], 4))
    print("z =", round(solution[2], 4))

    x = round(solution[0], 4)
    y = round(solution[1], 4)
    z = round(solution[2], 4)
    
    print("Avaliando a solução:")
    print("10x + 2y + z =", 10*x + 2*y + z)
    print("x + 5y + z =", x + 5*y + z)
    print("2x + 3y + 10z =", 2*x + 3*y + 10*z)
    
    print("Diferença entre a solução encontrada e a solução real:")
    print("x =", x - solucao_final[0])
    print("y =", y - solucao_final[1])
    print("z =", z - solucao_final[2])
    


gaus_jacobi()
print('='*50)
gaus_seidel()
