using LinearAlgebra, StatsBase, Flux, ProgressMeter, CUDA
using Flux: train!


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
        for i = j+1:N
            M[i,j] = get_mutual_entropy(X[:,i],X[:,j])
        end
    end
    return M+M'
end



function experiment1(;N=100, steps = 1000, epochs = 5000, ntrain = 100)

    data = []
    A = zeros(Int8,(N,N))
    X = zeros(Int8,N)
    B = zeros(Int8,(steps,N))
    C = zeros((N,N))
    @showprogress "generating data..." for k = 1:ntrain
        A = gen_adj(N) ; 
        X = rand(N) .< 0.2 ; # initial condition
        B = gen_tep(A,X,steps=steps);
        C = get_mutual_entropy_matrix(B)
        append!(data,[(C,A)])
        #data = [(C,A)]
    end
        
    model = Chain( Dense( N, 128, relu), Dense( 128, N) )
    loss(model,x,y) = mean(abs2.( model(x) .- y))
    opt = Flux.setup(Adam(), model)
    v = zeros(epochs) ; 
    @showprogress for epoch in 1:epochs
        v[epoch] = loss(model,data[1][1],data[1][2])
        train!(loss,model, data, opt)
    end
    return model, v,data

    # test the model on different data 
end



function experiment2(;N=100, steps = 1000, epochs = 5000, ntrain = 100)

    data = []
    A = zeros(Int8,(N,N))
    X = zeros(Int8,N)
    B = zeros(Int8,(steps,N))
    C = zeros((N,N))
    @showprogress "generating data..." for k = 1:ntrain
        A = gen_adj(N,p=rand()) ; 
        X = rand(N) .< 0.2 ; # initial condition
        B = gen_tep(A,X,steps=steps);
        C = get_mutual_entropy_matrix(B)
        append!(data,[(C,A)])
        #data = [(C,A)]
    end
        
    model = Chain( Dense( N, 128, relu), Dense( 128, N) )
    loss(model,x,y) = mean(abs2.( model(x) .- y))
    opt = Flux.setup(Adam(), model)
    v = zeros(epochs) ; 
    @showprogress for epoch in 1:epochs
        v[epoch] = mean([loss(model,data[k][1],data[k][2]) for k=1:ntrain])
        train!(loss,model, data, opt)
    end
    return model, v,data

    # test the model on different data 
end


function experiment3(;N=100, steps = 1000, epochs = 5000, ntrain = 10)

    data = []
    A = zeros(Int8,(N,N))
    X = zeros(Int8,N)
    B = zeros(Int8,(steps,N))
    C = zeros((N,N))
    @showprogress "generating data..." for k = 1:ntrain
        A = gen_adj(N,p=rand()) ; 
        X = rand(N) .< 0.2 ; # initial condition
        B = gen_tep(A,X,steps=steps);
        C = get_mutual_entropy_matrix(B)
        append!(data,[(C,A)])
        #data = [(C,A)]
    end
        
    model = Chain( Dense( N, 128, relu),Dense(128,32,relu),Dense( 32, N) )
    loss(model,x,y) = mean(abs2.( model(x) .- y))
    opt = Flux.setup(Adam(), model)
    v = zeros(epochs) ; 
    @showprogress for epoch in 1:epochs
        v[epoch] = mean([loss(model,data[k][1],data[k][2]) for k=1:ntrain])
        train!(loss,model, data, opt)
    end
    return model, v,data

    # test the model on different data 
end


function experiment4(; N=100, steps=1000, epochs=5000, ntrain=10 , m=64,model=nothing,data=[])
    @showprogress "Generating data..." for k = 1:ntrain
        A = gen_adj(N, p=rand())
        X = rand(N) .< 0.2  # Initial condition
        B = gen_tep(A, X, steps=steps)
        C = get_mutual_entropy_matrix(B)
        append!(data, [(C[:,m], A[:,m]) for m=1:N])
    end
    if isnothing(model)
        model = Chain(Dense(N, m, relu),Dense(m, N))
    end
    loss(model, x, y) = mean(abs2.(model(x) .- y))
    opt = Flux.setup(Adam(0.001), model)
    v = zeros(Float32, epochs)
    @showprogress "Training..." for epoch in 1:epochs
        v[epoch] = mean([loss(model, x, y) for (x, y) in data])
        Flux.train!(loss, model, data, opt)
    end
    return model, v, data
end



function experiment5(; N=100, steps=1000, epochs=5000, ntrain=10,m = 128, model=nothing,data=[])
    @showprogress "Generating data..." for k = 1:ntrain
        A = gen_adj(N, p=rand())
        X = rand(N) .< 0.2  # Initial condition
        B = gen_tep(A, X, steps=steps)
        C = get_mutual_entropy_matrix(B)
        append!(data, [(reshape(C,N*N), reshape(A,N*N))])
    end
    if isnothing(model)
        model = Chain(Dense(N*N, m, relu),Dense(m, N*N))
    end
    loss(mm, x, y) = mean(abs2.(mm(x) .- y))
    opt = Flux.setup(Adam(), model)
    v = zeros(Float32, epochs)
    @showprogress "Training..." for epoch in 1:epochs
        v[epoch] = mean([loss(model, x, y) for (x, y) in data])
        Flux.train!(loss, model, data, opt)
    end
    return model, v, data
end



function experiment6(; N=100, steps=1000, epochs=5000, p=nothing,m = [128,64], model=nothing,data=[])
    if isnothing(p)
        p = rand(1)
    end
    ntrain = length(p)
    @showprogress "Generating data..." for k = 1:ntrain
        A = gen_adj(N, p=p[k])
        X = rand(N) .< 0.2  # Initial condition
        B = gen_tep(A, X, steps=steps)
        C = get_mutual_entropy_matrix(B)
        append!(data, [(reshape(C,N*N), reshape(A,N*N))])
    end
    if isnothing(model)
        model = Chain(Dense(N*N, m[1], relu),Dense(m[1],m[2],relu),Dense(m[2], N*N,sigmoid))
    end
    
    #loss(mm, x, y) = mean(abs2.(mm(x) .- y))
    loss(mm, x, y) = Flux.binarycrossentropy(mm(x),y) #if only 0,1
    opt = Flux.setup(Adam(), model)
    v = zeros(Float32, epochs)
    @showprogress "Training..." for epoch in 1:epochs
        v[epoch] = mean([loss(model, x, y) for (x, y) in data])
        Flux.train!(loss, model, data, opt)
    end
    return model, v, data
end




function experiment7(; N=100, steps=1000, epochs=5000, p=nothing,m = [128,64], model=nothing,data=[],
                     learning_rate=0.01)
    if isnothing(model)
        model = Chain(Dense(N*N, m[1], relu),Dense(m[1],m[2],relu),Dense(m[2], N*N,sigmoid)) |> gpu
    end
    
    if isnothing(p)
        p = rand(1)
    end
    ntrain = length(p)
    @showprogress "Generating data..." for k = 1:ntrain
        A = gen_adj(N, p=p[k])
        X = rand(N) .< 0.2  # Initial condition
        B = gen_tep(A, X, steps=steps)
        C = get_mutual_entropy_matrix(B)
        append!(data, [(vec(C), vec(A)) |> gpu ])
    end
    
    loss(mm, x, y) = Flux.binarycrossentropy(mm(x),y)  #if only 0,1
    opt = Flux.setup(Adam(learning_rate), model)
    #v =  zeros(Float32, epochs)
    v = [] 
    @showprogress "Training..." for epoch in 1:epochs
        #v[epoch] = mean([loss(model, x, y) for (x, y) in data])
        append!(v, [ mean([loss(model, x, y) for (x, y) in data]) |> gpu ])
        Flux.train!(loss, model, data, opt)
    end
    
    return model |> cpu, v |> cpu, data |> cpu
end



function experiment8(data; epochs=5000,m = [128,64], model=nothing,learning_rate=0.001)
    NN = length(data[1][1])
    if isnothing(model)
        model = Chain(Dense(NN, m[1], relu),Dense(m[1],m[2],relu),Dense(m[2], NN,sigmoid)) |> gpu
    end    
    loss(mm, x, y) = Flux.binarycrossentropy(mm(x),y)  #if only 0,1
    opt = Flux.setup(Adam(learning_rate), model)
    #v =  zeros(Float32, epochs)
    v = [] 
    @showprogress "Training..." for epoch in 1:epochs
        #v[epoch] = mean([loss(model, x, y) for (x, y) in data])
        append!(v, [ mean([loss(model, x, y) for (x, y) in data])  ])
        Flux.train!(loss, model, data, opt)
    end
    
    return model , v 
end
