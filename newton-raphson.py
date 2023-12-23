def newton_raphson(f, f_prime, x0, tol):
    """
    Implementa o método de Newton-Raphson.

    Args:
      f: A função a ser resolvida.
      f_prime: A derivada da função a ser resolvida.
      x0: A aproximação inicial.
      tol: O critério de parada.

    Returns:
      Uma aproximação da raiz.
    """

    x = x0
    while True:  # Loop infinito
        # Calcula a próxima aproximação -> significa que x' = x - f(x)/f'(x)
        x_prime = x - (f(x) / f_prime(x))
        if abs(f(x_prime)) < tol:  # Critério de parada
            return x_prime  # Retorna a aproximação da raiz
        x = x_prime  # Atualiza a aproximação


if __name__ == "__main__":
    def f(x): return 0.5 * x ** 2 - 1
    def f_prime(x): return x  # Derivada de f(x)
    x0 = (-2 + -1) / 2  # Ponto médio do intervalo [-2, -1]
    tol = 0.01  # Tolerância desejada
    x_star = newton_raphson(f, f_prime, x0, tol)  # Aproximação da raiz
    print("A raiz é", x_star)
