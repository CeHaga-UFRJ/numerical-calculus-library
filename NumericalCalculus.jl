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