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



function erro(A, x ,b)
    return norm(A * x - b)
end 

#=
Função que calcula a norma de um vetor v.

Dado um vetor v, a sua norma será a raiz quadrada do somatório dos seus elementos elevados ao quadrado

Entrada: Um vetor v
Saída: A norma do vetor v

=#
function norma(v)
    v = v.^2 # Elevo os elementos ao quadrado
    soma = sum(v) # Faço o somatório dos elementos ao quadrado
    z = sqrt(soma) # Faço a raiz quadrada dessa soma    
    return z
end 
 

#=
Função que resolve um sistema linear denso aproximadamente

Dado uma matriz densa nxn A e um vetor b resolve o sistema A’Ax=A’b onde Ax* aproximadamente b ( x*=argmin ||Ax-b|| )

Entrada: Uma matriz densa nxn A e um vetor b
Saída: Um vetor x* nx1

=#
function minimos_quadrados(A, b)
        return resolve_sistema(A'*A, A'*b) # Resolvo o sistema A’Ax=A’b
end

#=
Resolve um sistema Ax = b onde A eh uma matriz nxn e b uma matriz nx1

Dado um sistema Ax=b podemos decompor A em LU resolvendo o sistema:
L(Ux) = b

Onde podemos substituir (Ux) por Y e resolver o sistema

LY = b

E depois desfazer a substituição e resolver
Ux = Y

A: Matrix no formato (n,n)
y: Lado direito da equacao no formato (n,1)

Retorno: Solucao x
=#
function resolve_sistema(A, b)
    L, U = decomposicao_lu(A) # Faz a decomposição LU
    Y = resolve_triangular_inferior(L, b) # Resolve o sistema b
    x = resolve_triangular_superior(U, Y)
    return x
end

#=
Resolve um sistema Ax = y onde A eh uma matriz triangular superior
A: Matrix triangular superior no formato (n,n)
y: Lado direito da equacao no formato (n,1)

Retorno: Solucao x
=#
function resolve_triangular_superior(A, y)
    # Podemos pegar o tamanho de y, pois eh o mesmo que A
    n = length(y)   
    x = zeros(n)
    
    # Na triangular superior, comecamos de baixo para cima, substituindo as variaveis anteriores nas equacoes
    for i = n:(-1):1
        x[i] = (y[i] - sum([A[i, k] * x[k] for k = i+1:n])) / A[i,i] 
    end
    
    return x
end

#=
Resolve um sistema Ax = y onde A eh uma matriz triangular inferior
A: Matrix triangular inferior no formato (n,n)
y: Lado direito da equacao no formato (n,1)

Retorno: Solucao x
=#
function resolve_triangular_inferior(A, y)
    # Podemos pegar o tamanho de y, pois eh o mesmo que A
    n = length(y)
    x = zeros(n)
    
    # Na triangular inferior, comecamos de cima para baixo, substituindo as variaveis anteriores nas equacoes
    for i = 1:n
        x[i] = (y[i] - sum([A[i, k] * x[k] for k = 1:i-1])) / A[i,i] 
    end
    
    return x
end

#=
Resolve um sistema Ax = y onde A eh uma matriz diagonal
A: Matrix diagonal no formato (n,n)
y: Lado direito da equacao no formato (n,1)

Retorno: Solucao x
=#
function resolve_diagonal(A, y)
    # Podemos pegar o tamanho de y, pois eh o mesmo que A
    n = length(y)
    
    # Divide o lado direito pelo coeficiente de cada variavel
    x = [y[i]/A[i,i] for i = 1:n]
    
    return x
end

#=
Realiza a decomposicao LU de uma matriz quadrada A
A: Matrix no formato (n,n)

Retorno: Matrizes L & U da decomposicao LU
=#
function decomposicao_lu(A)
    # Podemos salvar apenas um tamanho pois sao o mesmo
    n, = size(A)
    
    # U comeca como uma copia de A, enquanto L comeca como uma matriz identidade
    U = copy(A)
    L = Matrix(1.0I, n, n)
    
    for i = 1:n
        for j = i+1:n
            # Eh calculado o coeficiente dividindo o numero da matriz pelo pivot
            l = U[j,i] / U[i,i]
            # O coeficiente eh o elemento de L
            L[j,i] = l
            # E o coeficiente eh usado para alterar a linha de U
            U[j,:] += -l * U[i,:]
        end
    end
    return L, U
end

#=
Calcula a matriz inversa usando decomposicao LU
A: Matriz no formato (n,n) para encontrar a inversa

Retorno: Inversa de A
=#
function inverse_LU(A)
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
Realiza a decomposicao LU de uma matriz quadrada A e retorna as matrizes L e U

Problema: Decomposicao LU

Parâmetro A: Matrix no formato (n,n)

Retorno: Matrizes L (triangular inferior) e U (triangular superior) da decomposicao LU

Especificacao: A = LU
=#
function lu_decomposition(A)
    # Podemos salvar apenas um tamanho pois sao o mesmo
    n, = size(A)
    
    # U comeca como uma copia de A, enquanto L comeca como uma matriz identidade
    U = copy(A)
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
        Obrigatório caso o método seja Bisseção

    b : Number
        Fim do intervalo
        Obrigatório caso o método seja Bisseção

    error: Number
        Erro no domínio
        Obrigatório caso o método seja Bisseção

    method: String, optional
        Método que será aplicado a aproximação da raiz da função (:bisecion ou :newton)
        Caso nenhum método seja escolhido o padrão é o da Bisseção


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
----------
Monta uma matriz de Vandermonde de dado grau

Parâmetros
----------
    x: Vector{Number}
        Vetor usado como base
    qtd_rows: Number
        Quantidade de pontos
    degree: Number
        Grau do polinomio

Retorno
----------
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

    c = V \ y #resolve_sistema(V, y) FIXME: quando utilizo o resolve_sistema meu exemplo 2 não funciona
    
    f(x) = sum(c[n+1]*x^n for n in 0:degree)
    
    return f
end

@doc raw"""
Objetivo
----------
Transforma um conjunto de pontos discretos em uma função contínua.

Parâmetros
----------
    points : Vector{Tuple{Number, Number}}
        Vetor com coordenadas (x,y)

    degree : Number, optional
        Grau da interpolação

    method: Symbol, optional
        Nome do método utilizado para a interpolação

Retorno
----------
    function : function
        Retorna uma função

"""
function interpolation(points::Vector, degree::Number=0, method::Symbol=:vandermond)
    
    if degree == 0
        degree, = size(points)
    end
    
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



#============================== Fim Funções de 2 a 8 ===========================#

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