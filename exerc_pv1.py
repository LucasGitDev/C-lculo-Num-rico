

def q1():
    def erro_absoluto(valor_real, valor_aproximado) -> float:
        return abs(valor_real - valor_aproximado)

    def erro_relativo(valor_real, valor_aproximado):
        return erro_absoluto(valor_real, valor_aproximado) / float(valor_real)

    valor_real = 2.718281828
    valor_aproximado = 2.718

    calc_erro_absoluto = erro_absoluto(valor_real, valor_aproximado)
    calc_erro_relativo = erro_relativo(valor_real, valor_aproximado)

    print("Erro absoluto:", calc_erro_absoluto)
    print("Erro relativo:", calc_erro_relativo)


def q2():
    def erro_absoluto(valor_real, valor_aproximado):
        return abs(valor_real - valor_aproximado)

    def erro_relativo(valor_real, valor_aproximado):
        return erro_absoluto(valor_real, valor_aproximado) / valor_real

    valor_real = 96485.33289
    valor_aproximado = 96485

    calc_erro_absoluto = erro_absoluto(valor_real, valor_aproximado)
    calc_erro_relativo = erro_relativo(valor_real, valor_aproximado)

    print("Erro absoluto:", calc_erro_absoluto)
    print("Erro relativo:", calc_erro_relativo)


def q3():

    def f(x):
        return x**3 - 2*x**2 - 3*x + 1

    def encontrar_intervalos_com_raizes(a, b, passo):
        intervalos_com_raizes = []

        while a < b:
            fa = f(a)
            fb = f(a + passo)

            if fa * fb < 0:
                intervalos_com_raizes.append((a, a + passo))

            a += passo

        return intervalos_com_raizes

    a_inicial = -4
    b_inicial = 4
    passo = 1  # Ajuste conforme necessário

    intervalos = encontrar_intervalos_com_raizes(a_inicial, b_inicial, passo)

    if len(intervalos) > 0:
        print("Intervalos com raízes encontrados:")
        for intervalo in intervalos:
            print(f"Intervalo: [{intervalo[0]}, {intervalo[1]}]")
    else:
        print("Não foram encontrados intervalos com mudança de sinal no intervalo dado.")


def q4():

    def f(x):
        return x**3 - 2*x**2 - 3*x + 1

    def bisseccao(a, b, tolerancia, max_repeticoes):
        if f(a) * f(b) > 0:
            print(
                "Não há mudança de sinal no intervalo. O método da Bissecção não se aplica.")
            return None

        for i in range(max_repeticoes):
            c = (a + b) / 2.0
            if f(c) == 0 or (b - a) / 2.0 < tolerancia:
                return c
            elif f(a) * f(c) < 0:
                b = c
            else:
                a = c

        return (a + b) / 2.0

    # Intervalo [a, b] com mudança de sinal
    a = -4
    b = 0
    tolerancia = 0.0001  # Tolerância desejada
    max_repeticoes = 4  # Máximo de repetições

    raiz_aproximada = bisseccao(a, b, tolerancia, max_repeticoes)

    if raiz_aproximada is not None:
        print(f"Raiz aproximada encontrada: {raiz_aproximada:.5f}")
    else:
        print("Não foi possível encontrar uma raiz no intervalo dado ou atingir a tolerância desejada.")


def q5():

    def f(x):
        return x**3 - 2*x**2 - 3*x + 1

    def df(x):
        return 3*x**2 - 4*x - 3

    def newton_raphson(x0, tolerancia, max_repeticoes):
        for i in range(max_repeticoes):
            fx = f(x0)
            dfx = df(x0)

            if abs(fx) < tolerancia:
                return x0

            x1 = x0 - fx / dfx
            x0 = x1

        return x0

    # Valor inicial igual ao ponto médio do intervalo [0, 4]
    x0 = 2
    tolerancia = 0.0001  # Tolerância desejada
    max_repeticoes = 4  # Máximo de repetições

    raiz_aproximada = newton_raphson(x0, tolerancia, max_repeticoes)

    if raiz_aproximada is not None:
        print(f"Segunda raiz aproximada encontrada: {raiz_aproximada:.5f}")
    else:
        print(
            "Não foi possível encontrar uma segunda raiz ou atingir a tolerância desejada.")


def q6():

    def f(x):
        return -2*x**2 + 4*x + 2

    def encontrar_intervalos_com_raizes(a, b, passo):
        intervalos_com_raizes = []

        while a < b:
            fa = f(a)
            fb = f(a + passo)

            if fa * fb < 0:
                intervalos_com_raizes.append((a, a + passo))

            a += passo

        return intervalos_com_raizes

    # Intervalo inicial [-3, 3]
    a_inicial = -3
    b_inicial = 3
    passo = 0.1  # Ajuste conforme necessário

    intervalos = encontrar_intervalos_com_raizes(a_inicial, b_inicial, passo)

    if len(intervalos) > 0:
        print("Intervalos com raízes encontrados:")
        for intervalo in intervalos:
            print(f"Intervalo: [{intervalo[0]:.2f}, {intervalo[1]:.2f}]")
    else:
        print("Não foram encontrados intervalos com mudança de sinal no intervalo dado.")


def q7():
    def f(x):
        return -2*x**2 + 4*x + 2

    def bisseccao(a, b, tolerancia, max_repeticoes):
        if f(a) * f(b) > 0:
            print(
                "Não há mudança de sinal no intervalo. O método da Bissecção não se aplica.")
            return None

        for i in range(max_repeticoes):
            c = (a + b) / 2.0
            if f(c) == 0 or (b - a) / 2.0 < tolerancia:
                return c
            elif f(a) * f(c) < 0:
                b = c
            else:
                a = c

        return (a + b) / 2.0

    # Intervalo [a, b] com mudança de sinal
    a = -3
    b = 0
    tolerancia = 0.0001  # Tolerância desejada
    max_repeticoes = 4  # Máximo de repetições

    raiz_aproximada = bisseccao(a, b, tolerancia, max_repeticoes)

    if raiz_aproximada is not None:
        print(f"Raiz aproximada encontrada: {raiz_aproximada:.5f}")
    else:
        print("Não foi possível encontrar uma raiz no intervalo dado ou atingir a tolerância desejada.")


def q8():
    def f(x):
        return -2*x**2 + 4*x + 2

    def df(x):
        return -4*x + 4

    def newton_raphson(x0, tolerancia, max_repeticoes):
        for i in range(max_repeticoes):
            fx = f(x0)
            dfx = df(x0)

            if abs(fx) < tolerancia:
                return x0

            x1 = x0 - fx / dfx
            x0 = x1

        return x0

    # Valor inicial igual ao ponto médio do intervalo [0, 3]
    x0 = 1.5
    tolerancia = 0.0001  # Tolerância desejada
    max_repeticoes = 4  # Máximo de repetições

    raiz_aproximada = newton_raphson(x0, tolerancia, max_repeticoes)

    if raiz_aproximada is not None:
        print(f"Segunda raiz aproximada encontrada: {raiz_aproximada:.5f}")
    else:
        print(
            "Não foi possível encontrar uma segunda raiz ou atingir a tolerância desejada.")


"""
1ª)Na solução de diversos problemas matemáticos, uma constante bastante utilizada é o número de Neper, 
um número irracional representado por e. Considerando apenas as nove casas decimais da calculadora Cassio fx-82,
o mesmo pode ser  representado  por  2,718281828. Ao  realizar  certo cálculo, se  eu  aproximar  este  valor  dado 
por  2,718,  qual  o  erro absoluto e qual o erro relativo introduzido nessa aproximação? 
"""
print("Questão 1")
q1()

"""
2ª)No estudo da eletroquímica, o valor 96485,33289 C/mol é conhecido por constante de Faraday. Ao realizar certo 
cálculo,  se  eu  aproximar  este  valor  dado  por 96485,qual  o  erro  absoluto  
e  qual  o  erro  relativo  introduzido  nessa aproximação? 
"""
print("\nQuestão 2")
q2()

"""
(Enunciado para as questões 3a 5)Seja a função f(x) = x^3–2x^2–3x + 1. 
Sabendo que esta função possui suas raízes no intervalo [-4, 4], responda:
"""

"""
3ª)Usando o método T.E.U., localize os intervalos que se encontram cada uma das raízes reais da função.
"""
print("\nQuestão 3")
q3()

"""
4ª)Usando o Método da  Bissecção e  um dos  intervalos  encontrados na  questão 3,
encontre  uma  raiz aproximada da função.Considerando no máximo 4repetições (critério de parada).
"""
print("\nQuestão 4")
q4()
"""
5ª)Usando o Método de Newton-Raphson e um outro dos intervalos encontrados na questão 3, 
encontre uma segunda raiz aproximada da função, utilizando como valor inicial o 
ponto médio do intervalo usado.Considerando no máximo 4repetições (critério de parada).
"""
print("\nQuestão 5")
q5()

"""
(Enunciado para as questões 6a 8)Seja a função f(x) = -2^x2+ 4x + 2. Sabendo que esta função possui suas raízes no intervalo [-3, 3], responda:
"""

"""
6ª)Usando o método T.E.U., localize os intervalos que se encontram cada uma das raízes reais da função.
"""
print("\nQuestão 6")
q6()

"""
7ª)Usando o Método da Bissecção e o primeiro dos intervalos encontrados na questão 6, encontre uma raiz aproximada da função.
Considere no máximo4repetições (critério de parada).
"""
print("\nQuestão 7")
q7()

"""
8ª) Usando o Método de Newton-Raphson e o segundo dos intervalos encontrados na questão 6, encontre uma segunda raiz aproximada da função,
utilizando como valor inicial o ponto médio do intervalo usado.Considerando no máximo 4repetições (critério de parada).
"""
print("\nQuestão 8")
q8()

"""
Questão 1
Erro absoluto: 0.00028182799999987296
Erro relativo: 0.00010367872716392709

Questão 2
Erro absoluto: 0.33289000000513624
Erro relativo: 3.4501616985107365e-06

Questão 3
Intervalos com raízes encontrados:
Intervalo: [-2, -1]
Intervalo: [0, 1]
Intervalo: [2, 3]

Questão 4
Raiz aproximada encontrada: -1.12500

Questão 5
Segunda raiz aproximada encontrada: 3.20867

Questão 6
Intervalos com raízes encontrados:
Intervalo: [-0.50, -0.40]
Intervalo: [2.40, 2.50]

Questão 7
Raiz aproximada encontrada: -0.46875

Questão 8
Segunda raiz aproximada encontrada: 2.41423
"""
