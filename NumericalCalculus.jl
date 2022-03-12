#=
  Implementacao de varios metodos de calcnum ... agora estou enrolando falando da nossa biblioteca
=#

using LinearAlgebra

#=
Essa funcao eh uma funcao de teste e serve para mostrar como montar uma funcao

x: O que seu coracao mandar

Retorno: A funcao sempre vai retornar 42
=#
function teste(x)
    return 42
end 



#============================== Funções dos problemas de 2 a 8 ===========================#

## 2. Problema: Achar um valor aproximado de uma função com informações de derivadas

@doc raw"""
Objetivo
------------------------------
Achar um valor aproximado de uma função com informações de derivadas utilizando o método de Taylor

Especificação
------------------------------
|y - f(x)|<= E


Parâmetros de entrada
------------------------------
    x : Number
        Valor aproximado que queremos calcular a aproximação em y (f(x))

    a : Number
        Ponto fixo que será calculado a aproximação
        Idealmente é próximo de x e é conhecida as derivadas no ponto

    derivatives : Vector{Number}
        Vetor com informações das derivadas no ponto a: f(a), f'(a), f''(a) ...

    M: Number
        Maior valor da n-derivada no intervalo (teto)

    n: Int64, optional
        Número de termos do polinômio de Taylor
        Se nenhum valor for passado será calculado o polinômio de ordem 2


Retorno
------------------------------
    y : Flotat64
        Retorna o valor aproximado de f(x)

    E : Float64
        Retorna o erro da aproximação

Exceções
------------------------------

    AssertionError
        Caso a quantidade de derivadas no vetor seja insuficiente para realizar o 
        cálculo, levanta exeção de domínio

"""
function value_approximation(x::Number, a::Number, derivatives::Vector, M::Number, n::Int64=2)
    factor = 1
    error = (x-a)^factor/factor
    
    @assert(size(derivatives)[1] > n, "Informe um conjunto de derivadas que seja maior que o valor n (ordem do polinômio)")
    
    sum = derivatives[1]
    
    for factor=factor:n
        sum += derivatives[factor+1] * (((x-a)^factor)/factorial(factor))
        error = ((x-a)^(factor+1))/factorial(factor+1)
    end
    
    return sum, error
    
end

## 3. Problema: Encontrar raíz aproximadamente (Resolver equações não-lineares) (uma variável e uma equação)


@doc raw"""
Objetivo
------------------------------
Encontrar zero de função (raiz) para calcular aproximação de valores numéricos com o método da Bisseção.
Retorna um aviso caso o intervalo passado pelo usuário não possua sinais trocados.

Especificação
------------------------------
f(r)=0 e |x-r| <= erro


Parâmetros
------------------------------
    f : Function
        Recebe uma função 

    a : Number
        Início do intervalo

    b : Number
        Fim do intervalo

    error: Number
        Erro no domínio


Retorno
------------------------------
    root : Float64
        Retorna uma aproximação para a raiz calculada pelo método da Bisseção

Exceções
------------------------------
    AssertionError
        Caso o intervalo passado não tenha troca de valores no intervalo (sinais opostos) para o método da Bisseção

"""
function find_root(f::Function, a::Number, b::Number, error::Number) 
    
    @assert(f(a)*f(b) < 0, "Aviso!! A função no intervalo passado [$a, $b] não possui sinais opostos")
    
    n = floor(log2((b - a) / error)) + 1
    
    for i=1:n
        average = (b + a)/2
        
        if f(a) * f(average) < 0 # Verifica se o valor da função nos pontos são opostos
            b = average # Estreitando o resultado pelo lado esquerdo
        else
            a = average # Estreitando o resultado pelo lado direito
        end
    end
    
    aprox = (b + a)/2
    return aprox

end


@doc raw"""
Objetivo
------------------------------
Encontrar zero de função (raiz) para calcular aproximação de valores numéricos com o método de Newton

Especificação
------------------------------
f(r)=0

Parâmetros
------------------------------
    f : Function
        Recebe uma função 

    derivative: Function
        Derivada da função f

    qtty_iterations: Int64, optional
        Quantidade de interações para ser utilizada no método 
        Caso nenhum valor seja passado será calculado 10 iterações

    kick: Float64
        Chute da função inicial para começar a aplicar o método

    method: String, optional
        Método que será aplicado a aproximação da raiz da função (:bisecion ou :newton)
        Caso nenhum método seja escolhido o padrão é o da Bisseção

Retorno
------------------------------
    root : Float64
        Retorna uma aproximação para a raiz calculada pelo método de Newton


"""
function find_root(f::Function, derivative::Function, kick::Number, qtty_iterations::Int64=10)
    
    for i=1:qtty_iterations
       
        kick = kick - f(kick) / derivative(kick) # Método de newton utilizando a equação da reta
    end
    
    return kick

end

## 4. Problema: Interpolação Polinomial

@doc raw"""
Objetivo
------------------------------
Monta uma matriz de Vandermonde de dado grau

Parâmetros
------------------------------
    x: Vector{Number}
        Vetor usado como base
    qtd_rows: Number
        Quantidade de pontos
    degree: Number
        Grau do polinomio

Retorno
------------------------------
    Retorno: Matriz de Vandermonde
"""
function vandermonde(x::Vector, qtd_rows::Number, degree::Number) 
    # Cria uma matriz vazia 
    V = zeros(qtd_rows, degree+1)
    
    # Para cada coluna adiciona uma potencia de u
    for i = 1:degree+1
         V[:, i] = x.^(i-1)
    end
    
    return V
end

function vandermonde_interpolation(x::Vector, y::Vector, degree::Number)

    qtd_rows = length(y)
    
    V = vandermonde(x, qtd_rows, degree)

    c = resolve_sistema(V, y)
    
    f(x) = sum(c[n+1]*x^n for n in 0:degree)
    
    return f
end

@doc raw"""
Objetivo
------------------------------
Transforma um conjunto de pontos discretos em uma função contínua.

Especificação
------------------------------
Para todo 1<=i<=n, F(x_i)=y_i

Parâmetros
------------------------------
    points : Vector{Tuple{Number, Number}}
        Vetor com coordenadas (x,y). Formato: [(x1, y1), (x2,y2)]

    method: Symbol, optional
        Nome do método utilizado para a interpolação

Retorno
------------------------------
    function : function
        Retorna um polinômio com grau no máximo n-1 (tamanho do vetor de pontos - 1)

"""
function interpolation(points::Vector, method::Symbol=:vandermond)
    
    size_points, = size(points)
    
    degree = size_points - 1
    
    x = zeros(0)
    y = zeros(0)
    
    for point in points
        push!(x, point[1])
        push!(y, point[2])
    end
    
        
    if method == :vandermond
        return vandermonde_interpolation(x, y, degree)
    end

end

## 5 e 6. Problema: Regressão

@doc raw"""
Objetivo
------------------------------
Realizar a regressão com coeficientes lineares

Especificação
------------------------------
Para todo 1<=i<=n, F(x_i) aproximadamente y_i

Parâmetros
------------------------------
    points : Vector{Tuple{Number, Number}}
        Vetor com coordenadas (x,y). Formato: [(x1, y1), (x2,y2)]

    degree : Int64
        Grau da interpolação

    functions: Vector{function}, optional
        Caso esse vetor for passado, será calculado a regressão generalizada

Retorno
------------------------------
    function : function
        Retorna uma função no seu formato linear tradicional ou generalizada (com multiplifcação de funções)

"""
function linear_regression(points::Vector, degree::Int64, functions=nothing)
    
    x = zeros(0)
    y = zeros(0)
    
    for point in points
        push!(x, point[1])
        push!(y, point[2])
    end
    
    qtd_rows = length(y)
    
    V = vandermonde(x,qtd_rows,degree)
    
    c = minimos_quadrados(V,y)
    
    if functions == nothing
        return lr(x) = sum(c[n+1]*x^n for n in 0:degree) # linear regression
    else
        return lrg(x) = sum(c[n+1]*functions[n+1](x) for n in 0:degree) # linear regression generalized
    end
end


## 7. Problema: Regressão com coeficientes não lineares


# 7.1 Exponencial

@doc raw"""
Objetivo
------------------------------
Realizar a regressão com coeficientes não lineares do modelo exponencial

Especificação
------------------------------
Para todo 1<=i<=n, F(x_i) aproximadamente y_i
Calculada com a linearização da forma ln(y) = ln(c1) + c2*x

Parâmetros
------------------------------
    points : Vector{Tuple{Number, Number}}
        Vetor com coordenadas (x,y). Formato: [(x1, y1), (x2,y2)]

Retorno
------------------------------
    function : function
        Retorna uma função com o modelo da forma y = c1*e^(c2*x)

"""
function exponential_regression(points::Vector)
    # modelo: y = c1*e^(c2*x)
    # linearização: ln(y) = ln(c1) + c2*x
    
    # Etapa 1: converte o vetor de coordenadas em dois vetores de x e y
    
    x = zeros(0)
    y = zeros(0)
    
    for point in points
        push!(x, point[1])
        push!(y, point[2])
    end
    
    # Etapa 2: troca de variável (linearização)
    
    x_barra=x
    y_barra=log.(y)
    
    # Etapa 3: regressão linear com grau 1
    
    qtd_rows = length(y)
    V = vandermonde(x_barra,qtd_rows,1)
    
    c_barra= V\y_barra # TODO: trocar
    
    # Etapa 4: troca de variável (modelo)

    c1=exp(c_barra[1])
    c2=c_barra[2]
    
    modelo(x) = c1*exp(c2*x)
    
    return modelo
    
end

# 7.2 Potência

@doc raw"""
Objetivo
------------------------------
Realizar a regressão com coeficientes não lineares do modelo de potência

Especificação
------------------------------
Para todo 1<=i<=n, F(x_i) aproximadamente y_i
Calculada com a linearização da forma ln(y) = ln(c1) + c2*ln(x)

Parâmetros
------------------------------
    points : Vector{Tuple{Number, Number}}
        Vetor com coordenadas (x,y). Formato: [(x1, y1), (x2,y2)]

Retorno
------------------------------
    function : function
        Retorna uma função com o modelo da forma y = c1*x^(c2)

"""
function potency_regression(points::Vector)
    # modelo: y = c1*x^(c2)
    # linearização: ln(y) = ln(c1) + c2*ln(x)
    
    # Etapa 1: converte o vetor de coordenadas em dois vetores de x e y
    
    x = zeros(0)
    y = zeros(0)
    
    for point in points
        push!(x, point[1])
        push!(y, point[2])
    end
    
    # Etapa 2: troca de variável (linearização)
    
    x_barra=log.(x)
    y_barra=log.(y)
    
    # Etapa 3: regressão linear com grau 1
    
    qtd_rows = length(y)
    V = vandermonde(x_barra,qtd_rows,1)
    
    c_barra= V\y_barra # TODO: trocar
    
    # Etapa 4: troca de variável (modelo)

    c1=exp(c_barra[1])
    c2=c_barra[2]
    
    modelo(x) = c1*(x^c2)
    
    return modelo
    
end


# 7.3 Geométrico

@doc raw"""
Objetivo
------------------------------
Realizar a regressão com coeficientes não lineares do modelo geométrico

Especificação
------------------------------
Para todo 1<=i<=n, F(x_i) aproximadamente y_i
Calculada com a linearização da forma 1/y = c1 + c2*x

Parâmetros
------------------------------
    points : Vector{Tuple{Number, Number}}
        Vetor com coordenadas (x,y). Formato: [(x1, y1), (x2,y2)]

Retorno
------------------------------
    function : function
        Retorna uma função com o modelo da forma y = 1/(c1 + c2*x)

"""
function geometric_regression(points::Vector)
    # modelo: y = 1/(c1 + c2*x)
    # linearização: 1/y = c1 + c2*x
    
    # Etapa 1: converte o vetor de coordenadas em dois vetores de x e y
    
    x = zeros(0)
    y = zeros(0)
    
    for point in points
        push!(x, point[1])
        push!(y, point[2])
    end
    
    # Etapa 2: troca de variável (linearização)
    
    x_barra=x
    
    temp_y(x) = 1/x
    y_barra=temp_y.(y)
    
    # Etapa 3: regressão linear com grau 1
    
    qtd_rows = length(y)
    V = vandermonde(x_barra,qtd_rows,1)
    
    c_barra= V\y_barra # TODO: trocar
    
    # Etapa 4: troca de variável (modelo)

    c1=c_barra[1]
    c2=c_barra[2]
    
    modelo(x) = 1/(c1 + c2*x)
    
    return modelo
    
end


## 8. Problema: Interpolação 2D

@doc raw"""
Objetivo
------------------------------
Realiza a interpolacao 2D (bilinear) dado 4 pontos geométricos e suas respectivas alturas

Especificação
------------------------------
Para todo 1<=i<=n, F(x_i,y_j)=zij

Parâmetros
------------------------------
    points : Vector{Tuple{Number, Number}}
        Vetor com 2 coordenadas (x,y). Formato: [(x1, y1), (x2,y2)]

    z : Vector{Float64}
        Vetor com alturas 

Retorno
------------------------------
    function : function
        Retorna uma função de grau dois de duas variáveis

"""
function interpolation_2d(points::Vector, z::Vector)
    x = zeros(0)
    y = zeros(0)
    
    for point in points
        push!(x, point[1])
        push!(y, point[2])
    end
    
    x1, x2 = x
    y1, y2 = y
    
    f(x, y) = (z[1]*(x2 - x)*(y2 - y) + z[2]*(x2 - x)*(y - y1) + z[3]*(x - x1)*(y2 - y) + z[4]*(x - x1)*(y - y1))/((x2 - x1)*(y2 - y1))
    
    # Modelo do polinômio: a + bx + cy + dxy
    a = f(0, 0) # = a + 0 + 0 + 0
    b = f(1, 0) - a # = a + b + 0 + 0
    c = f(0, 1) - a # = a + 0 + c + 0
    d = f(1, 1) - a - b - c # = a + b + c + d
    
    return lagrange_2d(x,y) = a + b*x + c*y + d*x*y
    
end



#============================== Fim Funções de 2 a 8 ===========================#


## 9. Calcular a norma de um vetor v

@doc raw"""
Objetivo
----------
Calcular a norma de um vetor.

Parâmetros
----------
    v : Vector{Float64}
        Recebe um vetor

Retorno
----------
    z : Float64
        Retorna a norma do vetor v

"""
function vector_norm(v::Vector{Float64})
    v = v.^2 # Elevo os elementos ao quadrado
    soma = sum(v) # Faço o somatório dos elementos ao quadrado
    z = sqrt(soma) # Faço a raiz quadrada dessa soma    
    return z
end 
 

## 10. Problema: Resolver um sistema linear denso aproximadamente

@doc raw"""
Objetivo
----------
Aproximar a solução do sistema Ax=b utilizando o método de mínimos quadrados.

Dado uma matriz densa mxn (m>n) A e um vetor b resolve o sistema A’Ax=A’b onde Ax* aproximadamente b ( x*=argmin ||Ax-b|| )

Parâmetros
----------
    A : Matrix{Float64}
        Recebe uma matriz densa

    b : Vector{Float64}
        Um vetor b tal que Ax=b

Retorno
----------
    x : Vector{Float64}
        Retorna uma aproximação da solução do sistema A’Ax=A’b

"""
function least_squares(A::Matrix{Float64}, b::Vector{Float64})
        return solve_system(A'*A, A'*b) # Resolvo o sistema A’Ax=A’b
end


## 11. Problema: Resolver exatamente uma sistema linear denso

@doc raw"""
Objetivo
----------
Resolver um sistema Ax = b onde A eh uma matriz nxn e b uma matriz nx1 utilizando decomposição LU.

Parâmetros
----------
    A : Matrix{Float64}
        Recebe uma matriz nxn

    b : Vector{Float64}
        Um vetor tal que Ax=b

Retorno
----------
    x : Vector{Float64}
        Retorna a solução do sistema Ax=b

"""
function solve_system(A::Matrix{Float64}, b::Vector{Float64})
    n,m = size(A)
    @assert(n == m, string("Matrix must be a square. Received a (",n,",",m,")"))
    L, U = lu_decomposition(A) # Faz a decomposição LU
    Y = lower_triangular_solve(L, b) # Resolve o sistema LY=b
    x = upper_triangular_solve(U, Y) # Resolve o sistema Ux=Y
    return x
end


## 12. Problema: Resolver um sistema triangular superior

@doc raw"""
Objetivo
----------
Resolver um sistema Ax = y onde A eh uma matriz triangular superior e y uma matriz nx1

Parâmetros
----------
    A : Matrix{Float64}
        Recebe uma matriz triangular superior no formato (n,n)

    y : Vector{Float64}
        Um vetor tal que Ax=y

Retorno
----------
    x : Vector{Float64}
        Retorna a solução do sistema Ax=y

"""
function upper_triangular_solve(A::Matrix{Float64}, y::Vector{Float64})
    # Podemos pegar o tamanho de y, pois eh o mesmo que A
    n = length(y)   
    x = zeros(n)
    
    # Na triangular superior, comecamos de baixo para cima, substituindo as variaveis anteriores nas equacoes
    for i = n:(-1):1
        x[i] = (y[i] - sum([A[i, k] * x[k] for k = i+1:n])) / A[i,i] 
    end
    
    return x
end


## 13. Problema: Resolver um sistema triangular inferior

@doc raw"""
Objetivo
----------
Resolver um sistema Ax = y onde A eh uma matriz triangular inferior e y uma matriz nx1

Parâmetros
----------
    A : Matrix{Float64}
        Recebe uma Matrix triangular inferior no formato (n,n)

    y : Vector{Float64}
        Um vetor tal que Ax=y

Retorno
----------
    x : Vector{Float64}
        Retorna a solução do sistema Ax=y

"""
function lower_triangular_solve(A::Matrix{Float64}, y::Vector{Float64})
    # Podemos pegar o tamanho de y, pois eh o mesmo que A
    n = length(y)
    x = zeros(n)
    
    # Na triangular inferior, comecamos de cima para baixo, substituindo as variaveis anteriores nas equacoes
    for i = 1:n
        x[i] = (y[i] - sum([A[i, k] * x[k] for k = 1:i-1])) / A[i,i] 
    end
    
    return x
end


## 14. Problema: Resolver um sistema diagonal

@doc raw"""
Objetivo
----------
Resolver um sistema Ax = y onde A eh uma matriz triangular inferior e y uma matriz nx1

Parâmetros
----------
    A : Matrix{Float64}
        Recebe uma Matrix triangular inferior no formato (n,n)

    y : Vector{Float64}
        Um vetor tal que Ax=y

Retorno
----------
    x : Vector{Float64}
        Retorna a solução do sistema Ax=y

"""
function solve_diagonal(A::Matrix{Float64}, y::Vector{Float64})
    # Podemos pegar o tamanho de y, pois eh o mesmo que A
    n = length(y)
    
    # Divide o lado direito pelo coeficiente de cada variavel
    x = [y[i]/A[i,i] for i = 1:n]
    
    return x
end


## 15. Problema: achar a inversa de uma matriz

@doc raw"""
Objetivo
----------
Achar a inversa de uma matriz A utilizando o método da decomposição LU.

Parâmetros
----------
    A : Matrix{Float64}
        Recebe uma Matriz no formato (n,n)

Retorno
----------
    inv_A : Matrix{Float64}
        Retorna a inversa da matriz A

"""
function inverse_LU(A::Matrix{Float64})
    # Decompoe em LU
    L, U = decomposicao_lu(A)
    n, = size(A)
    
    # Inicializa a inversa
    inv_A = zeros(n,n)
    
    # Para cada coluna
    for i = 1:n
        # Cria um vetor one-hot (Identidade final)
        y = zeros(n)
        y[i] = 1
        
        # Resolve o sistema para a coluna i
        Y = resolve_triangular_inferior(L, y)
        x = resolve_triangular_superior(U, Y)
        
        # Substitui a coluna i da inversa
        inv_A[:,i] = x
    end
    
    return inv_A
end




#=

function numerical_integration(f, a, b, n=1, error=nothing, M=nothing)
    if(any(error, M))
end
=#

#=
Realiza a derivada continua usando diferenca para frente, para tras e centradas e retorna a derivada de uma funcao num ponto x

Problema: Derivada continua

f: Funcao a ser derivada
x: Ponto a ser calculada a derivada
h: Tamanho dos intervalos a serem percorridos
option:
 - :front, para usar diferenca para frente (Padrao caso nao entre)
 - :back, para usar diferenca para tras
 - :center, para usar diferenca centrada

Retorno: A derivada no ponto x

Especificacao: 
=#
function continuous_derivative(f, x, h, option=:front)
    @assert(option == :front || option == :back || option == :center, "Invalid option, possible options are: :front, :back, and :center")
    if(option == :front) return (f(x+h) - f(x))/h end
    if(option == :back) return (f(x) - f(x-h))/h end
    if(option == :center) return (f(x+h) - f(x-h))/2h end
end