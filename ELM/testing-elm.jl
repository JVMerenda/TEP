using LinearAlgebra, StatsBase, Flux, ProgressMeter, CUDA
using Flux: train!
using Random: seed!
using BSON,Dates


function gen_adj(N; p=0.4)
    @assert p < 1e0
    return Int8.(Hermitian(reshape(sample( [0,1],Weights([1e0-p,p]),N*N),(N,N))))
end

function gen_tep(A,X0; steps=110, pars=Dict("a"=>0.0004 , "b"=>0.01))
    N = size(X0)[1]
    X = zeros(Int8,(N,steps)) 
    X[:,1] = X0
    Y = zeros(Int8,N) 
    for k=1:steps-1
        Y .= rand(N) .< (pars["a"]*(1 .- X[:,k]) .* (A * X[:,k]) + pars["b"]*X[:,k])
        X[:,k+1] .= X[:,k] .+ Y  .- 2*(X[:,k].* Y)
    end
    return X' 
end

function gen_data(;N=100,p=nothing,steps=110, pars=Dict("a"=>0.0004 , "b"=>0.01))
    if isnothing(p)
        p = rand(1)
    end
    data_size = length(p)
    X = rand(N) .< 0.2  # Initial condition
    data = []
    @showprogress "Generating data..." for k = 1:data_size
        A = gen_adj(N, p=p[k])
        C = get_mutual_entropy_matrix(gen_tep(A, X, steps=steps))
        append!(data, [(vec(C), vec(A)) ])
    end
    return data
end


function gen_data_multiples(;N=100,p=nothing,steps=200,data_multi=1, pars=Dict("a"=>0.0004 , "b"=>0.01))
    if isnothing(p)
        p = rand(1)
        data_multi = 1
    end
    data_size = length(p)
    X = rand(N) .< 0.2  # Initial condition
    data = []
    @showprogress "Generating data..." for k = 1:data_size
        A = gen_adj(N, p=p[k])
        for j=1:data_multi
            C = get_mutual_entropy_matrix(gen_tep(A, X, steps=steps))
            append!(data, [(vec(C), vec(A)) ])
        end
    end
    return data
end



function get_entropy(X)
    p = [x[2] for x in countmap(X)]
    x0= sum(p)
    return -sum( [(x/x0)*log2(x/x0) for x in p])
end

function get_mutual_entropy(X,Y)
    p = [x[2] for x in countmap(zip(X,Y))]
    x0= sum(p)
    return get_entropy(X) + get_entropy(Y) + sum( [(x/x0)*log2(x/x0) for x in p])
end

function get_mutual_entropy_matrix(X)
    T,N = size(X)
    M = zeros(Float32,(N,N))
    for j = 1:N
        for i = j:N
            M[i,j] = get_mutual_entropy(X[:,i],X[:,j])
        end
    end
    return M+M'
end


activation(x) = sigmoid.(x)
function ELM(X, T ; num_hidden = 2048)    
    input_size = size(X, 2)
    output_size = size(T, 2)
    W_hidden = randn(Float32, input_size, num_hidden) 
    b_hidden = randn(Float32, 1, num_hidden)         
    H = activation(X * W_hidden .+ b_hidden)  
    β = pinv(H) * T
    return (x) -> activation(x * W_hidden .+ b_hidden) * β
end

function experiment_elm(; N = 100 , ntrain = [100,5], ntest=[10,1] ,num_hidden = 2048,steps=1000)
    # if isnothing(seed)
    #     seed = 812753712
    # end
    # seed!(seed)
    
    data_train = zeros(Float32,prod(ntrain),N*N,2)
    X = (rand(N) .< 0.2)
    kk = 0 ; A = zeros(Int8,(N,N))   
    for k=1:ntrain[1]
        A = gen_adj(N, p = mod(0.5 + 0.1*randn(),1e0))
        for j=1:ntrain[2]
            kk += 1
            data_train[kk,:,2] = vec(A)
            data_train[kk,:,1] = vec(get_mutual_entropy_matrix(gen_tep(A,X,steps=steps )))
        end
    end
    data_test = zeros(Float32,prod(ntest),N*N,2)
    jj = 0 ; B = zeros(Int8,(N,N))   
    for k=1:ntest[1]
        B = gen_adj(N, p = mod(0.5 + 0.1*randn(),1e0))
        for j=1:ntest[2]
            jj += 1
            data_test[jj,:,2] = vec(B)
            data_test[jj,:,1] = vec(get_mutual_entropy_matrix(gen_tep(B,X )))
        end
    end 
        
    model = ELM(data_train[:,:,1],data_train[:,:,2],num_hidden=num_hidden)
    u = now()
    BSON.@save "exp-elm-$num_hidden-$u.bson" model data_train data_test

    avg_train = mean([count((model(data_train[k,:,1]') .> 0.5) .== data_train[k,:,2]') for k=1:prod(ntrain)])
    avg_test  = mean([count((model(data_test[k,:,1]') .> 0.5) .== data_test[k,:,2]') for k=1:prod(ntest)])
    
    
    return avg_train, avg_test
end
