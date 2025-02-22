using Flux , LinearAlgebra, StatsBase, Random, CUDA, Logging, Zygote
include("tools.jl")



function ELM(X, T ; num_hidden = 2048 , activation=relu,λ=1e-3)    
    input_size = size(X, 1)
    output_size = size(T, 1)
    factor = sqrt(2e0 / (input_size+num_hidden))
    W_hidden = CUDA.randn(Float32, num_hidden, input_size) * factor  
    b_hidden = CUDA.randn(Float32, num_hidden, 1)          
    H = activation.( W_hidden * X .+ b_hidden)  |> gpu
    β = transpose((H * H' + λ*I) \ (   H * (T')   )) |> gpu   
    #@info "Finished training ELM" size(β)
    return (x) -> β * activation.( W_hidden * x .+ b_hidden) 
end



loss(m,x,y) = mean( abs2.( m(x) .- y ))    

function experiment_cor_cov(; N = 20 , pars= Dict("a"=>0.045, "b"=> 0.1) , ρ = 0.4  , steps = 125 ,
                            ntrain = 100, ntest = 20,epochs = 100)
    num_hidden = 2*N*N+1
    activation = relu
    X0 = CUDA.rand(N) .< ρ
    A = CUDA.zeros(Int16,(N,N))
    B = CUDA.zeros(Float32,(steps,N))
    M = CUDA.zeros(Float32,(N,N))
    C = CUDA.zeros(Float32,(N,N))
    data1 = CUDA.zeros(Float32,(N*N,ntrain))
    data2 = CUDA.zeros(Float32,(N*N,ntrain,6))
    @showprogress "Generating training data..." for k = 1:ntrain
        A .= gen_adj(N , p = 0.5 + 0.1*randn()) |> gpu
        B .= gen_tep(A, X0, pars= pars ,  steps= steps)
        M .= get_mutual_entropy_matrix( B |> cpu) |> gpu
        C .= B'B
        data1[:,k] .= vec(A)  ;
        data2[:,k,1] .= vec(M)
        data2[:,k,2] .= vec(cor(M |> cpu, A |> cpu)) |> gpu
        data2[:,k,3] .= vec(cov(M |> cpu, A |> cpu)) |> gpu
        data2[:,k,4] .= vec(C)
        data2[:,k,5] .= vec(cor(C |> cpu,A |> cpu)) |> gpu
        data2[:,k,6] .= vec(cov(C |> cpu,A |> cpu)) |> gpu
    end
    
    datat1 = CUDA.zeros(Int16,(N*N,ntest))
    datat2 = CUDA.zeros(Float32,(N*N,ntest,6))
    @showprogress "Generating validation data..." for j=1:ntest
        A .= gen_adj(N, p = 0.5 +0.1*randn()) |> gpu
        B .= gen_tep(A, X0, pars= pars ,  steps= steps)
        M .= get_mutual_entropy_matrix( B |> cpu) |> gpu
        C .= B'B
        datat1[:,j] = vec( A )
        datat2[:,j,1] .= vec( M )
        datat2[:,j,2] .= vec(cor(M |> cpu, A |> cpu)) |> gpu
        datat2[:,j,3] .= vec(cor(M |> cpu, A |> cpu)) |> gpu
        datat2[:,j,4] .= vec(C)
        datat2[:,j,5] .= vec(cor(C |> cpu, A |> cpu)) |> gpu
        datat2[:,j,6] .= vec(cov(C |> cpu, A |> cpu)) |> gpu
    end

    
    m = [ELM(data2[:,:,k], data1,num_hidden = num_hidden,λ = 1e-3, activation=activation) |> gpu for k=1:6]
    a = ones(Float32,6)
    @info "Training entry layers... done" size(m)
    output_training     = replace( [ m[k](data2[:,:,k])  |> cpu for k=1:6], NaN => 0)
    output_validation   = replace( [ m[k](datat2[:,:,k]) |> cpu for k=1:6], NaN => 0)


    
    combined_training   = CUDA.zeros(Float32,N*N)
    combined_validation = CUDA.zeros(Float32,N*N)
    
    combined_training   = replace(reduce(+,[output_training[k]*a[k]   for k=1:6]),NaN => 0) |> gpu
    combined_validation = replace(reduce(+,[output_validation[k]*a[k] for k=1:6]),NaN => 0) |> gpu

    # combined_training   = reduce(+,[output_training[k]*a[k]   for k=1:6]) |> gpu
    # combined_validation = reduce(+,[output_validation[k]*a[k] for k=1:6]) |> gpu


    
    prediction = ELM(combined_training,data1,num_hidden=num_hidden,λ=1e-3,activation=activation)        
    @info "training   loss::" loss(prediction, combined_training  , data1)    
    @info "validation loss::" loss(prediction, combined_validation, datat1)

    #return output_training, output_validation, m, prediction,data1,data2,datat1,datat2


    a_final = zeros(6)
    current_loss = N*N*1e0
    #@showprogress
    for epoch = 1:epochs
        #combined_training   = reduce(+,[output_training[k]*a[k]   for k=1:6]) |> gpu
        #combined_validation = reduce(+,[output_validation[k]*a[k] for k=1:6]) |> gpu
        a = randn(6)
        combined_training   = replace(reduce(+,[output_training[k]*a[k]   for k=1:6]),NaN => 0) |> gpu
        combined_validation = replace(reduce(+,[output_validation[k]*a[k] for k=1:6]),NaN => 0) |> gpu
        prediction = ELM(combined_training,data1,num_hidden=num_hidden,λ=1e-3,activation=activation) 
        #grads = Zygote.gradient(a -> loss(prediction,combined_training,data1) )        
        #a -= 0.01*grads
        training_loss = loss(prediction,combined_training,data1)+loss(prediction,combined_validation,datat1)
        if training_loss < current_loss
            a_final .= a*1e0
            current_loss = training_loss
            println("Epoch $epoch :: loss $training_loss")
            println(a_final)
        end
    end
    println("=> ", a_final)
    combined_training   = replace(reduce(+,[output_training[k]*a_final[k] for k=1:6]),NaN => 0) |> gpu
    combined_validation = replace(reduce(+,[output_validation[k]*a_final[k] for k=1:6]),NaN => 0) |> gpu
    prediction = ELM(combined_training,data1,num_hidden=num_hidden,λ=1e-3,activation=activation)
    training_loss = loss(prediction,combined_training,data1)+loss(prediction,combined_validation,datat1)
    println("loss $training_loss")
end


function experiment_cor_cov_cpu(; N = 20 , pars= Dict("a"=>0.045, "b"=> 0.1) , ρ = 0.4  , steps = 125 ,
                            ntrain = 100, ntest = 20,epochs = 100)
    num_hidden = 2*N*N+1
    activation = tanh
    X0=rand(N) .< ρ
    A = zeros(Int16,(N,N))
    B = zeros(Float32,(steps,N))
    M = zeros(Float32,(N,N))
    C = zeros(Float32,(N,N))
    data1 = zeros(Float32,(N*N,ntrain))
    data2 = zeros(Float32,(N*N,ntrain,6))
    @showprogress "Generating training data..." for k = 1:ntrain
        A .= gen_adj(N , p = 0.5 + 0.1*randn())  |> cpu
        B .= gen_tep(A, X0, pars= pars ,  steps= steps) |> cpu
        M .= get_mutual_entropy_matrix( B ) 
        C .= B'B
        data1[:,k] .= vec(A)  ;
        data2[:,k,1] .= vec(M)
        data2[:,k,2] .= vec(cor(M , A ))
        data2[:,k,3] .= vec(cov(M , A ))
        data2[:,k,4] .= vec(C)
        data2[:,k,5] .= vec(cor(C ,A )) 
        data2[:,k,6] .= vec(cov(C ,A )) 
    end
    
    datat1 = zeros(Int16,(N*N,ntest))
    datat2 = zeros(Float32,(N*N,ntest,6))
    @showprogress "Generating validation data..." for j=1:ntest
        A .= gen_adj(N, p = 0.5 +0.1*randn())  |> cpu
        B .= gen_tep(A, X0, pars= pars ,  steps= steps) |> cpu
        M .= get_mutual_entropy_matrix( B ) 
        C .= B'B
        datat1[:,j] = vec( A )
        datat2[:,j,1] .= vec( M )
        datat2[:,j,2] .= vec(cor(M , A ))
        datat2[:,j,3] .= vec(cor(M , A ))
        datat2[:,j,4] .= vec(C)
        datat2[:,j,5] .= vec(cor(C , A )) 
        datat2[:,j,6] .= vec(cov(C , A )) 
    end

    
    m = [ELM_cpu(data2[:,:,k], data1,num_hidden = num_hidden,λ = 1e-3, activation=activation) for k=1:6]
    a = ones(Float32,6)
    @info "Training entry layers... done" size(m)
    output_training     = replace( [ m[k](data2[:,:,k])   for k=1:6], NaN => 0)
    output_validation   = replace( [ m[k](datat2[:,:,k])  for k=1:6], NaN => 0)


    
    combined_training   = zeros(Float32,N*N)
    combined_validation = zeros(Float32,N*N)
    
    combined_training   = replace(reduce(+,[output_training[k]*a[k]   for k=1:6]),NaN => 0) 
    combined_validation = replace(reduce(+,[output_validation[k]*a[k] for k=1:6]),NaN => 0) 

    # combined_training   = reduce(+,[output_training[k]*a[k]   for k=1:6]) |> gpu
    # combined_validation = reduce(+,[output_validation[k]*a[k] for k=1:6]) |> gpu


    
    prediction = ELM_cpu(combined_training,data1,num_hidden=num_hidden,λ=1e-3,activation=activation)        
    @info "training   loss::" loss(prediction, combined_training  , data1)    
    @info "validation loss::" loss(prediction, combined_validation, datat1)

    #return output_training, output_validation, m, prediction,data1,data2,datat1,datat2


    a_final = zeros(6)
    current_loss = N*N*1e0
    #@showprogress
    for epoch = 1:epochs
        #combined_training   = reduce(+,[output_training[k]*a[k]   for k=1:6]) |> gpu
        #combined_validation = reduce(+,[output_validation[k]*a[k] for k=1:6]) |> gpu
        a = randn(6)
        combined_training   = replace(reduce(+,[output_training[k]*a[k]   for k=1:6]),NaN => 0) 
        combined_validation = replace(reduce(+,[output_validation[k]*a[k] for k=1:6]),NaN => 0) 
        prediction = ELM_cpu(combined_training,data1,num_hidden=num_hidden,λ=1e-3,activation=activation) 
        #grads = Zygote.gradient(a -> loss(prediction,combined_training,data1) )        
        #a -= 0.01*grads
        training_loss = loss(prediction,combined_training,data1)+loss(prediction,combined_validation,datat1)
        if training_loss < current_loss
            a_final .= a*1e0
            current_loss = training_loss
            println("Epoch $epoch :: loss $training_loss")
            #println(a_final)
        end
    end
    combined_training   = replace(reduce(+,[output_training[k]*a_final[k] for k=1:6]),NaN => 0) 
    combined_validation = replace(reduce(+,[output_validation[k]*a_final[k] for k=1:6]),NaN => 0) 
    prediction = ELM_cpu(combined_training,data1,num_hidden=num_hidden,λ=1e-3,activation=activation)
    training_loss = loss(prediction,combined_training,data1)+loss(prediction,combined_validation,datat1)
    println("loss $training_loss")
end



function ELM_cpu(X, T ; num_hidden = 2048 , activation=relu,λ=1e-3)    
    input_size = size(X, 1)
    output_size = size(T, 1)
    factor = sqrt(2e0 / (input_size+num_hidden))
    W_hidden = randn(Float32, num_hidden, input_size) * factor  
    b_hidden = randn(Float32, num_hidden, 1)          
    H = activation.( W_hidden * X .+ b_hidden)  
    β = transpose((H * H' + λ*I) \ (   H * (T')   )) 
    #@info "Finished training ELM" size(β)
    return (x) -> β * activation.( W_hidden * x .+ b_hidden) 
end


