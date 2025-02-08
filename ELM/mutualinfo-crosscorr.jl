include("tools.jl")
using Flux: train!
using Random: seed!
using BSON
using StatisticalMeasures

# ====================================================================
# ====================================================================
#activation(x) = sigmoid.(x)
function ELM(X, T ; num_hidden = 2048 , activation=sigmoid,λ=1e-3)    
    input_size = size(X, 2)
    output_size = size(T, 2)
    factor = sqrt(2e0 / (input_size+num_hidden))
    W_hidden = CUDA.randn(Float32, input_size, num_hidden) * factor  
    b_hidden = CUDA.randn(Float32, 1, num_hidden)          
    H = activation.(X * W_hidden .+ b_hidden)  |> gpu    
    #β = (pinv(H |>cpu) |> gpu) * T
    β = (H' * H + λ*I) \ (H' * T) |> gpu
    return (x) -> activation.(x * W_hidden .+ b_hidden) * β   
end


function build_mlp(data,val_data;lr=1e-2,epochs=1_000,layers=[1024,512])
    input_size  = length(data[1][1])
    output_size = length(data[1][2])
    model = Chain(
        Dense(input_size, layers[1], relu),
        Dense( layers[1],  layers[2], relu),
        Dense( layers[2], output_size, sigmoid)
    ) |> gpu 
    loss(m,x,y) = Flux.binarycrossentropy(m(x),y)
    opt = Flux.setup(Adam(lr),model)   
    @showprogress for epoch=1:epochs
        Flux.train!(loss,model,data,opt)
        println()
    end
    return model,loss
end


function experiment_elm(; N = 100 , ntrain = [100,5], ntest=[5,1] ,num_hidden = 2048,steps=1000)
    # if isnothing(seed)
    #     seed = 812753712
    # end
    # seed!(seed)
    
    data_train = zeros(Float32,prod(ntrain),N*N,2)
    X = (rand(N) .< 0.2)
    kk = 0 ; A = zeros(Int8,(N,N))
    
    @info "Generating train data"
    for k=1:ntrain[1]
        A = gen_adj(N, p = mod(0.5 + 0.1*randn(),1e0))
        for j=1:ntrain[2]
            kk += 1
            data_train[kk,:,2] = vec(A)
            data_train[kk,:,1] = vec(get_mutual_entropy_matrix(cpu(gen_tep(A,X,steps=steps ))))
        end
    end

    @info "Generating test data"
    data_test = zeros(Float32,prod(ntest),N*N,2)
    jj = 0 ; B = zeros(Int8,(N,N))   
    for k=1:ntest[1]
        B = gen_adj(N, p = mod(0.5 + 0.1*randn(),1e0))
        for j=1:ntest[2]
            jj += 1
            data_test[jj,:,2] = vec(B)
            data_test[jj,:,1] = vec(get_mutual_entropy_matrix(cpu(gen_tep(B,X ))))
        end
    end 
    
    @info "training"
    model = ELM(gpu(data_train[:,:,1]),gpu(data_train[:,:,2]),num_hidden=num_hidden)
    BSON.@save "exp-elm-$num_hidden.bson" model data_train data_test

    avg_train_acc = mean(accuracy(model(gpu(data_train[:,:,1]')) .> .5, gpu(data_train[:,:,2]')))
    avg_test_acc  = mean(accuracy((cpu(model(gpu(data_test[k,:,1]'))) .> 0.5) .== cpu(data_test[k,:,2])') for k=1:prod(ntest))
    
    return avg_train, avg_test
end

avg_train, avg_test = experiment_elm()
