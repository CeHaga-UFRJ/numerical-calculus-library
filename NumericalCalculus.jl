using LinearAlgebra

## 2. Problema: Achar um valor aproximado de uma função com informações de derivadas

@doc raw"""
Objetivo
------------------------------
Achar um valor aproximado de uma função com informações de derivadas utilizando o método de Taylor

Especificação
------------------------------
``|y - f(x)|<= E``


Parâmetros de entrada
------------------------------
    x : Number
        Valor aproximado que queremos calcular a aproximação em y (f(x))

    a : Number
        Ponto fixo que será calculado a aproximação
        Idealmente é próximo de x e é conhecida as derivadas no ponto

    derivatives : Vector
        Vetor com informações das derivadas no ponto a: f(a), f'(a), f''(a) ...

    M: Number
        Maior valor da n-derivada no intervalo (teto)

    n: Int64, default 2
        Número de termos do polinômio de Taylor
        Se nenhum valor for passado será calculado o polinômio de ordem 2


Retorno
------------------------------
    y : Float64
        Retorna o valor aproximado de f(x)

    E : Float64
        Retorna o erro da aproximação

Exceções
------------------------------

    AssertionError
        Caso a quantidade de derivadas no vetor seja insuficiente para realizar o 
        cálculo, levanta exceção de domínio

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
``f(r)=0`` e ``|x-r| <= erro``


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
``f(r)=0``

Parâmetros
------------------------------
    f : Function
        Recebe uma função 

    derivative: Function
        Derivada da função f

    kick: Float64
        Chute da função inicial para começar a aplicar o método

    qtty_iterations: Int64, default 10
        Quantidade de interações para ser utilizada no método 
        Caso nenhum valor seja passado será calculado 10 iterações

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
Monta uma matriz de Vandermonde (matriz de potência)

Parâmetros
------------------------------
    x: Vector
        Vetor usado como base da potência

    qtd_rows: Number
        Quantidade de linhas

    degree: Number
        Grau do polinômio

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

@doc raw"""
Objetivo
------------------------------
Realiza interpolação utilizando a matriz de vandermonde

Parâmetros
------------------------------
    x: Vector
        Vetor de x

    y: Vector
        Vetor de y

    degree: Number
        Grau do polinomio

Retorno
------------------------------
    Retorno: Polinômio interpolador
"""
function vandermonde_interpolation(x::Vector, y::Vector, degree::Number)

    qtd_rows = length(y)
    
    V = vandermonde(x, qtd_rows, degree)
    c = solve_system(V, y)
    
    f(x) = sum(c[n+1]*x^n for n in 0:degree)
    
    return f
end

@doc raw"""
Objetivo
------------------------------
Realiza interpolação utilizando o polinômio de Lagrange.
É utilizado a partir de combinações lineares.

Parâmetros
------------------------------
    x: Vector
        Vetor de x

    y: Vector
        Vetor de y

Retorno
------------------------------
    Retorno: Polinômio interpolador
"""
function lagrange_interpolation(x::Vector, y::Vector)
    
    # Ideia: montar combinações lineares ("peças de legos") 
    # para que a gente consiga interpolar sem resolver um sistema linear
    
    n, = size(y)
    
    # Monta as peças de lego e soma no formato
    # soma[numerador * (x-x[j]) / denominador *(x[i]-x[j])]
    
    f(var) = sum(
        [y[j] * prod([var - x[i] for i = 1:n if i != j])/prod([x[j] - x[i] for i = 1:n if i != j ]) for j = 1:n]
    )
    
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
        Vetor com coordenadas (x,y). Formato: ``[(x1, y1), (x2,y2)]``

    method: Symbol, {:vandermonde, :lagrange}, padrão :vandermonde
        Nome do método utilizado para a interpolação.


Retorno
------------------------------
    function : function
        Retorna um polinômio com grau no máximo n-1 (tamanho do vetor de pontos - 1)

Exceções
------------------------------

    AssertionError
        Caso o método escolhido seja diferente de :vandermonde ou :lagrange

"""
function interpolation(points::Vector, method::Symbol=:vandermonde)
    
    size_points, = size(points)
    
    degree = size_points - 1
    
    x = zeros(0)
    y = zeros(0)
    
    for point in points
        push!(x, point[1])
        push!(y, point[2])
    end
    
    @assert(method == :vandermonde || method == :lagrange, "Método de interpolação inválido, valores possíveis são: :vandermonde, ou :lagrange")
        
    if method == :vandermonde
        return vandermonde_interpolation(x, y, degree)
    elseif method ==:lagrange
        return lagrange_interpolation(x, y)
    end

end

## 5 e 6. Problema: Regressão

@doc raw"""
Objetivo
------------------------------
Monta uma matriz genérica de funções passadas pelo usuário

Parâmetros
------------------------------
    x: Vector
        Vetor usado como base

    qtd_rows: Number
        Quantidade de linhas

    degree: Number
        Grau do polinômio

    functions: Vector{function}
        Conjunto de funções

Retorno
------------------------------
    Retorno: Matriz onde as funções estão aplicadas em cada ponto x
"""
function functions_matrix(x::Vector, qtd_rows::Number, degree::Number, functions::Vector)
    # Cria uma matriz vazia 
    V = zeros(qtd_rows, degree+1)
    
    # Para cada coluna adiciona uma potencia de u e multiplica pela função
    for i = 1:degree+1
         V[:, i] = functions[i].(x)
    end
    
    return V
end

@doc raw"""
Objetivo
------------------------------
Realizar a regressão com coeficientes lineares

Especificação
------------------------------
Para todo ``1<=i<=n, F(x_i)`` aproximadamente ``y_i``

Parâmetros
------------------------------
    points : Vector{Tuple{Number, Number}}
        Vetor com coordenadas (x,y). Formato: ``[(x1, y1), (x2,y2)]``

    degree : Int64
        Grau da interpolação

    functions: Vector{function}, default nothing
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
    
    if functions == nothing
        # linear regression
        
        V = vandermonde(x,qtd_rows,degree)
        c =  least_squares(V,y)

        return lr(x) = sum(c[n+1]*x^n for n in 0:degree) 

    else
        # linear regression generalized

        V = functions_matrix(x,qtd_rows,degree,functions)
        c =  least_squares(V,y)

        return lrg(x) = sum(c[n+1]*functions[n+1](x) for n in 0:degree)
    end
end


## 7. Problema: Regressão com coeficientes não lineares


# 7.1 Exponencial

@doc raw"""
Objetivo
------------------------------
Realizar a regressão com coeficientes não lineares para o modelo exponencial

Especificação
------------------------------
Para todo ``1<=i<=n, F(x_i)`` aproximadamente ``y_i``
Calculada com a linearização da forma ``ln(y) = ln(c1) + c2*x``

Parâmetros
------------------------------
    points : Vector{Tuple{Number, Number}}
        Vetor com coordenadas (x,y). Formato: ``[(x1, y1), (x2,y2)]``

Retorno
------------------------------
    function : function
        Retorna uma função com o modelo da forma ``y = c1*e^(c2*x)``

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
    
    c_barra= solve_system(V, y_barra)
    
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
Realizar a regressão com coeficientes não lineares para o modelo de potência

Especificação
------------------------------
Para todo ``1<=i<=n, F(x_i)`` aproximadamente ``y_i``
Calculada com a linearização da forma ``ln(y) = ln(c1) + c2*ln(x)``

Parâmetros
------------------------------
    points : Vector{Tuple{Number, Number}}
        Vetor com coordenadas (x,y). Formato: ``[(x1, y1), (x2,y2)]``

Retorno
------------------------------
    function : function
        Retorna uma função com o modelo da forma ``y = c1*x^(c2)``

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
    
    c_barra= solve_system(V, y_barra)
    
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
Realizar a regressão com coeficientes não lineares para o modelo geométrico

Especificação
------------------------------
Para todo ``1<=i<=n, F(x_i)`` aproximadamente y_i
Calculada com a linearização da forma ``1/y = c1 + c2*x``

Parâmetros
------------------------------
    points : Vector{Tuple{Number, Number}}
        Vetor com coordenadas (x,y). Formato: ``[(x1, y1), (x2,y2)]``

Retorno
------------------------------
    function : function
        Retorna uma função com o modelo da forma ``y = 1/(c1 + c2*x)``

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
    
    c_barra= solve_system(V, y_barra)
    
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
Para todo ``1<=i<=n, F(x_i,y_j)=zij``

Parâmetros
------------------------------
    points : Vector{Tuple{Number, Number}}
        Vetor com 2 coordenadas (x,y). Formato: ``[(x1, y1), (x2,y2)]``

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

## 9. Calcular a norma de um vetor v

@doc raw"""
Objetivo
------------------------------
Calcular a norma de um vetor.

Especificação
------------------------------
``z=\sqrt{v_1^2 + v_2^2 + ... + v_n^2}``

Parâmetros
------------------------------
    v : Vector{Float64}
        Recebe um vetor

Retorno
------------------------------
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
------------------------------
Aproximar a solução do sistema Ax=b utilizando o método de mínimos quadrados

Dado uma matriz densa mxn (m>n) A e um vetor b resolve o sistema A’Ax=A’b onde Ax* aproximadamente b ( x*=argmin ||Ax-b|| )

Especificação
------------------------------
``Ax* \approx b ( x*=argmin ||Ax-b|| )``

Parâmetros
------------------------------
    A : Matrix{Float64}
        Recebe uma matriz densa

    b : Vector{Float64}
        Um vetor b tal que Ax=b

Retorno
------------------------------
    x : Vector{Float64}
        Retorna uma aproximação da solução do sistema A’Ax=A’b

"""
function least_squares(A::Matrix{Float64}, b::Vector{Float64})
        return solve_system(A'*A, A'*b) # Resolvo o sistema A’Ax=A’b
end


## 11. Problema: Resolver exatamente uma sistema linear denso

@doc raw"""
Objetivo
------------------------------
Resolver um sistema Ax = b onde A é uma matriz nxn e b uma matriz nx1 utilizando decomposição LU

Especificação
------------------------------
``Ax=b``

Parâmetros
------------------------------
    A : Matrix{Float64}
        Recebe uma matriz nxn

    b : Vector{Float64}
        Um vetor tal que Ax=b

Retorno
------------------------------
    x : Vector{Float64}
        Retorna a solução do sistema Ax=b

"""
function solve_system(A::Matrix{Float64}, b::Vector{Float64})
    n,m = size(A)
    @assert(n == m, string("Matriz deve ser quadrada. Matriz recebida (",n,",",m,")"))
    L, U = lu_decomposition(A) # Faz a decomposição LU
    Y = lower_triangular_solve(L, b) # Resolve o sistema LY=b
    x = upper_triangular_solve(U, Y) # Resolve o sistema Ux=Y
    return x
end


## 12. Problema: Resolver um sistema triangular superior

@doc raw"""
Objetivo
------------------------------
Resolver um sistema Ux = b onde U é uma matriz triangular superior e b uma matriz nx1

Especificação
------------------------------
``Ux=b``

Parâmetros
------------------------------
    U : Matrix{Float64}
        Recebe uma matriz triangular superior no formato (n,n)

    b : Vector{Float64}
        Um vetor tal que Ax=b

Retorno
------------------------------
    x : Vector{Float64}
        Retorna a solução do sistema Ax=b

"""
function upper_triangular_solve(U::Matrix{Float64}, b::Vector{Float64})
    # Podemos pegar o tamanho de b, pois eh o mesmo que U
    n = length(b)   
    x = zeros(n)
    
    # Na triangular superior, comecamos de baixo para cima, substituindo as variaveis anteriores nas equacoes
    for i = n:(-1):1
        x[i] = (b[i] - sum([U[i, k] * x[k] for k = i+1:n])) / U[i,i] 
    end
    
    return x
end


## 13. Problema: Resolver um sistema triangular inferior

@doc raw"""
Objetivo
------------------------------
Resolver um sistema Lx = b onde L é uma matriz triangular inferior e b uma matriz nx1

Especificação
------------------------------
``Lx=b``

Parâmetros
------------------------------
    L : Matrix{Float64}
        Recebe uma Matrix triangular inferior no formato (n,n)

    b : Vector{Float64}
        Um vetor tal que Lx=b

Retorno
------------------------------
    x : Vector{Float64}
        Retorna a solução do sistema Lx=b

"""
function lower_triangular_solve(L::Matrix{Float64}, b::Vector{Float64})
    # Podemos pegar o tamanho de b, pois eh o mesmo que L
    n = length(b)
    x = zeros(n)
    
    # Na triangular inferior, comecamos de cima para baixo, substituindo as variaveis anteriores nas equacoes
    for i = 1:n
        x[i] = (b[i] - sum([L[i, k] * x[k] for k = 1:i-1])) / L[i,i] 
    end
    
    return x
end


## 14. Problema: Resolver um sistema diagonal

@doc raw"""
Objetivo
------------------------------
Resolver um sistema Dx = b onde D é uma matriz diagonal e b uma matriz nx1

Especificação
------------------------------
``Dx=b``

Parâmetros
------------------------------
    D : Matrix{Float64}
        Recebe uma Matrix diagonal no formato (n,n)

    b : Vector{Float64}
        Um vetor tal que Dx=b

Retorno
------------------------------
    x : Vector{Float64}
        Retorna a solução do sistema Dx=b

"""
function solve_diagonal(D::Matrix{Float64}, b::Vector{Float64})
    # Podemos pegar o tamanho de b, pois eh o mesmo que D
    n = length(b)
    
    # Divide o lado direito pelo coeficiente de cada variavel
    x = [b[i]/D[i,i] for i = 1:n]
    
    return x
end


## 15. Problema: achar a inversa de uma matriz

@doc raw"""
Objetivo
------------------------------
Achar a inversa de uma matriz A utilizando o método da decomposição LU.

Especificação
------------------------------
``A\cdotB=B\cdotA=I``

Parâmetros
------------------------------
    A : Matrix{Float64}
        Recebe uma Matriz no formato (n,n)

Retorno
------------------------------
    B : Matrix{Float64}
        Retorna a inversa da matriz A

"""
function inverse_LU(A::Matrix{Float64})
    # Decompoe em LU
    L, U = lu_decomposition(A)
    n, = size(A)
    
    # Inicializa a inversa
    B = zeros(n,n)
    
    # Para cada coluna
    for i = 1:n
        # Cria um vetor one-hot (Identidade final)
        y = zeros(n)
        y[i] = 1
        
        # Resolve o sistema para a coluna i
        Y = lower_triangular_solve(L, y)
        x = upper_triangular_solve(U, Y)
        
        # Substitui a coluna i da inversa
        B[:,i] = x
    end
    
    return B
end

## 16. Problema: Decomposição LU

@doc raw"""
Objetivo
------------------------------
Realizar a decomposição LU

Especificação
------------------------------
``A = LU``

Parâmetros
------------------------------
    A : Matrix
        Matriz no formato (n,n)

Retorno
------------------------------
    L : Matrix{Float64}
        Matriz triangular inferior

    U : Matrix{Float64}
        Matriz triangular superior

"""
function lu_decomposition(A::Matrix)::Tuple{Matrix{Float64},Matrix{Float64}}
    # Podemos salvar apenas um tamanho pois sao o mesmo
    n, = size(A)

    # U comeca como uma copia de A, enquanto L comeca como uma matriz identidade
    U = copy(Float64.(A))
    L = Matrix(1.0I, n, n)

    for i = 1:n
        for j = i+1:n
            # Eh calculado o coeficiente dividindo o numero da matriz pelo pivot
            l = U[j,i] / U[i,i]
            # O coeficiente eh o elemento de L
            L[j,i] = l
            # E o coeficiente eh usado para alterar a linha de U
            U[j,:] -= l * U[i,:]
        end
    end
    return L, U
end

## 17. Problema: PVC

@doc raw"""
Objetivo
------------------------------
Resolve um Problema de Valores no Contorno dado ``y'' = c[1] + c[2]y + c[3]y', y(xi) = yi e y(xf) = yf``

Especificação
------------------------------
``y(x_i) ≈ y\_between[i]``

Parâmetros
------------------------------
    coeff : Vector
        Coeficientes da segunda derivada
        
    xi : Number
        Ponto inicial conhecido

    xf : Number
        Ponto final conhecido

    yi : Number
        Valor da função no ponto inicial

    yf : Number
        Valor da função no ponto final

    n : Int64
        Quantidade de pontos no meio a ser encontrados
        

Retorno
------------------------------
    y_between : Vector{Float64}
        Vetor de pontos internos entre xi e xf

"""
function generic_bvp(coeff::Vector, xi::Number, xf::Number, yi::Number, yf::Number, n::Int64)::Vector{Float64}
    # Calcula o tamanho do intervalo
    h = (xf - xi)/(n-1)
    
    A = generic_bvp_matrix(n-2, coeff[2], coeff[3], h) 
    b = zeros(n-2)
    
    # Cada equação vai possuir esse termo
    for i = 1:n-2
        b[i] = 2 * coeff[1] * h^2
    end
    
    # Com excecao do inicio e final, que possuem os limites ja conhecidos
    b[1] -= yi * (1 + coeff[3] * h)
    b[n-2] -= yf * (1 - coeff[3]) * h
    
    # Resolve e retorna
    y_between = solve_system(A, b)
    return [yi; y_between; yf]
end
function generic_bvp_matrix(n::Int64, b::Number, c::Number, h::Number)::Matrix
    A = zeros(n,n)
    # Manual no inicio
    A[1, 1] = (-2 - 2 * b * h^2)
    A[1, 2] = (1 - c * h)
    
    # Manual no final
    A[n, n-1] = (1 + c * h)
    A[n, n] = (-2 - 2 * b * h^2)
    
    # Monta a tridiagonal
    for i = 2:n-1
        A[i, i-1] = (1 + c * h)
        A[i, i] = (-2 - 2 * b * h^2)
        A[i, i+1] = (1 - c * h)
    end
    
    return A
end


## 18, 19 & 20. Problema: Integração Numérica

@doc raw"""
Objetivo
------------------------------
Realiza a integral numérica com um determinado número de intervalos

Especificação
------------------------------
``S \approx \int_a^b f(x)dx``

Parâmetros
------------------------------
    f : Function
        Função a ser derivada
        
    a : Number
        Limite inferior da integral

    b : Number
        Limite superior da integral

    n : Number
        Número de intervalos
        

Retorno
------------------------------
    S : Float64
        Aproximação da integral

"""
function numerical_integration(f::Function, a::Number, b::Number, n::Int64 = 1000)::Float64
    # Calcula distancia entre cada ponto
    h = (b-a)/n
    
    # Calcula a soma usando a formula extendida das areas
    S = 0
    xi = a
    # O meio sera somado duas vezes, as pontas uma
    for i = 1:n
        S += f(xi)
        xi += h
        S += f(xi)
    end
    S *= h/2
    
    return S
end

@doc raw"""
Objetivo
------------------------------
Realiza a integral numérica com erro máximo

Especificação
------------------------------
``\left| S - \int_a^b f(x)dx \right| < \text{erro}``

Parâmetros
------------------------------
    f : Function
        Função a ser derivada
        
    a : Number
        Limite inferior da integral

    b : Number
        Limite superior da integral

    error : Number
        Erro máximo

    M : Number
        Limite superior para a segunda derivada
        

Retorno
------------------------------
    S : Float64
        Aproximação da integral

"""
function numerical_integration(f::Function, a::Number, b::Number, error::Number, M::Number)::Float64
    n = ceil(sqrt((M * (b-a)^3) / (12 * error)))
    return numerical_integration(f, a, b, n)
end

@doc raw"""
Objetivo
------------------------------
Realiza a integral dupla numérica com um determinado número de intervalos

Especificação
------------------------------
``S \approx \int_a^b \int_{h(x)}^{g(x)} f(x,y) dx dy``

Parâmetros
------------------------------
    f : Function
        Função a ser derivada
        
    a : Number
        Limite inferior da integral

    b : Number
        Limite superior da integral

    n : Number
        Número de intervalos
        

Retorno
------------------------------
    S : Float64
        Aproximação da integral

"""
function numerical_integration(f::Function, g::Function, h::Function, a::Number, b::Number, n::Int64)::Float64
    # Calcula a integral para um y especifico
    function outer_integral(y)
        return numerical_integration(x -> f(x,y), h(y), g(y), n)
    end
    
    # Calcula a integral para cada y
    return numerical_integration(outer_integral, a, b)
end


## 21. Derivada Numérica com uma função contínua

@doc raw"""
Objetivo
------------------------------
Realiza a derivada continua usando diferenca para frente, para tras e centradas e retorna a derivada de uma funcao num ponto x

Especificação
------------------------------
``dx \approx f'(x)``

Parâmetros
------------------------------
    f : Function
        Função a ser derivada
        
    x : Number
        Ponto da derivação

    h : Number
        Tamanho do intervalo

    option : Symbol, {:front, :back, :center}, padrão :center
        Estratégia de derivação
        

Retorno
------------------------------
    dx : Float64
        Derivada no ponto x

"""
function continuous_derivative(f::Function, x::Number, h::Number, option::Symbol = :center)::Float64
    @assert(option == :front || option == :back || option == :center, "Opção inválida, possíveis opções são: :front, :back e :center")
    if(option == :front) return (f(x+h) - f(x))/h end
    if(option == :back) return (f(x) - f(x-h))/h end
    if(option == :center) return (f(x+h) - f(x-h))/2h end
end

## 22. Derivada Numérica com uma função discreta

@doc raw"""
Objetivo
------------------------------
Realiza a derivada de uma função discreta em um intervalo

Especificação
------------------------------
``fy(x_i) \approx y'(x_i)``

Parâmetros
------------------------------
    f : Vector
        Vetor dos valores da função
        
    x : Vector
        Vetor dos valores do domínio da função

    degree : Int64
        Grau da derivação

    option : Symbol, {:front, :back, :center}, padrão :center
        Estratégia de derivação
        

Retorno
------------------------------
    fy : Vector{Float64}
        Vetor das derivadas em cada ponto do domínio

"""
function discrete_derivative(f::Vector, x::Vector, degree::Int64, option::Symbol = :center)::Vector
    n = length(f)
    fy = copy(f)
    for k = 1:degree
        y = zeros(n)
        
        # Nas pontas so tem uma opcao
        y[1] = (fy[2] - fy[1])/(x[2] - x[1])
        y[n] = (fy[n] - fy[n-1])/(x[n] - x[n-1])
        
        # No meio e escolha do usuario
        for i = 2:n-1
            deriv = 0
            if(option == :front) deriv = (fy[i+1] - fy[i])/(x[i+1] - x[i]) end
            if(option == :back) deriv = (fy[i] - fy[i-1])/(x[i] - x[i-1]) end
            if(option == :center) deriv = (fy[i+1] - fy[i-1])/(x[i+1] - x[i-1]) end
            y[i] = deriv
        end
        
        fy = copy(y)
    end
    
    return fy
end