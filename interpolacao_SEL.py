import numpy as np
from scipy.interpolate import lagrange

def q1():
    def interpolar_polinomio(velocidades, consumos):
        # Criar o polinômio interpolador de Lagrange
        polinomio = lagrange(velocidades, consumos)
        return polinomio


    def estimar_consumo(polinomio, velocidade):
        consumo_estimado = polinomio(velocidade)
        return consumo_estimado


    def call():
        # Dados da tabela
        velocidades = [80, 100, 120]
        consumos = [14.2, 12.1, 10.4]

        # (a) Encontrar o polinômio interpolador
        polinomio_interpolador = interpolar_polinomio(velocidades, consumos)
        print("Polinômio interpolador:\n")
        print(polinomio_interpolador)
        print("\n")

        # (b) Estimar o consumo para as velocidades de 90 km/h e 110 km/h
        velocidade_1 = 90
        consumo_estimado_1 = estimar_consumo(polinomio_interpolador, velocidade_1)
        print(f"Estimativa de consumo em {velocidade_1} km/h: {consumo_estimado_1:.2f} km/l")

        velocidade_2 = 110
        consumo_estimado_2 = estimar_consumo(polinomio_interpolador, velocidade_2)
        print(f"Estimativa de consumo em {velocidade_2} km/h: {consumo_estimado_2:.2f} km/l")

    call()

def q2():
    def interpolacao_lagrange(x, y, x_estimado):
        """
        Realiza interpolação de Lagrange para estimar o valor de y no ponto x_estimado.
        x: Lista de valores de x (bimestres).
        y: Lista de valores de y (lucros).
        x_estimado: O ponto no qual queremos estimar o lucro.
        """
        n = len(x)
        resultado = 0

        for i in range(n):
            termo = y[i]
            for j in range(n):
                if i != j:
                    termo *= (x_estimado - x[j]) / (x[i] - x[j])
            resultado += termo

        return resultado
    
    def call():
        # Dados da tabela
        bimestres = [1, 2, 3]
        lucros = [10000, 12000, 7000]

        # Bimestre que desejamos estimar o lucro
        bimestre_estimado = 4

        # Realizar a interpolação e estimar o lucro
        lucro_estimado = interpolacao_lagrange(bimestres, lucros, bimestre_estimado)

        print(f"Estimativa de lucro no {bimestre_estimado}º bimestre: R${lucro_estimado:.2f}")

    call()
    
    
q1()

print("\n\n")

q2()