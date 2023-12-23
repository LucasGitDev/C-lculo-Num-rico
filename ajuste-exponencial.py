import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Dados fornecidos
x_data = np.array([0, 1, 2, 3, 4, 5])
y_data = np.array([1, 2, 4, 12, 27, 69])

# Definição da função exponencial para ajuste
def exponential_func(x, a, b):
    return a * np.exp(b * x)

# Ajuste da curva exponencial aos dados
params, covariance = curve_fit(exponential_func, x_data, y_data)

# Parâmetros 'a' e 'b'
a, b = params

# Gerar os valores estimados usando os parâmetros ajustados
y_fit = exponential_func(x_data, a, b)

# Plotar os resultados
plt.scatter(x_data, y_data, label='Dados reais')
plt.plot(x_data, y_fit, label='Ajuste exponencial', color='red')
plt.xlabel('Horas')
plt.ylabel('Número de bactérias')
plt.legend()
plt.show()

# Exibir os parâmetros encontrados
print(f"Parâmetro 'a': {a}")
print(f"Parâmetro 'b': {b}")
