def regra_do_trapezio(func, a, b):
    return (b - a) * (func(a) + func(b)) / 2.0

def regra_do_trapezio_repetida(func, a, b, n):
    h = (b - a) / n
    integral = 0.5 * (func(a) + func(b))
    for i in range(1, n):
        integral += func(a + i * h)
    return integral * h

def regra_de_simpson_repetida(func, a, b, n):
    if n % 2 != 0:
        raise ValueError("O número de intervalos deve ser par para a regra de Simpson")
    
    h = (b - a) / n
    integral = func(a) + func(b)
    
    for i in range(1, n):
        if i % 2 == 0:
            integral += 2 * func(a + i * h)
        else:
            integral += 4 * func(a + i * h)
    
    return integral * h / 3


import math

# Definindo a função a ser integrada (exemplo: função exponencial e^x)
def f(x):
    return math.exp(x)

# Limites de integração
a = 0
b = 1

n_intervalos = 4

integral_trapezio_single = regra_do_trapezio(f, a, b)
integral_trapezio_multiple = regra_do_trapezio_repetida(f, a, b, n_intervalos)
integral_simpson = regra_de_simpson_repetida(f, a, b, n_intervalos)

print("Integral usando a Regra do Trapézio (um intervalo):", integral_trapezio_single)
print("Integral usando a Regra dos Trapézios Repetidos ({} intervalos):".format(n_intervalos), integral_trapezio_multiple)
print("Integral usando a Regra de Simpson Repetida ({} intervalos):".format(n_intervalos), integral_simpson)
