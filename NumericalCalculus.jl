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