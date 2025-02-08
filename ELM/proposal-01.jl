using Flux , LinearAlgebra, StatsBase, Random, CUDA, Logging, Dates
include("tools.jl")
using StatisticalMeasures

loss(m,x,y) = mean( abs2.( m(x) .- y ))
acc(m,x,y) = accuracy(y, m(x))
f1(m,x,y) = f1score(cpu(y), cpu(m(x)))
prec(m,x,y) = StatisticalMeasures.precision(cpu(y), cpu(m(x)))
rec(m,x,y) = recall(cpu(y), cpu(m(x)))

function gen_inter_elm(X,Y;num_hidden = 1048,λ = 1e-3,activation=tanh)
    num_metrics = size(X,3)
    models = [ELM(X[:,:,k],Y,num_hidden=num_hidden,λ = λ, activation=activation) |> gpu for k=1:num_metrics]
    return models
end

function eval_inter_elm(models,X)
    num_metrics = size(X,3)
    return [models[k](X[:,:,k]) for k=1:num_metrics]
end

function combine(a,b)
    out = CUDA.zeros(Float32,size(a[1]))
    for k = 1:length(a)
        out += a[k]*b[k]
    end
    return out
end

function experiment_combine_cor_mif_uniform(; N = 50, ntrain = 40, ntest = 5,epochs = 100, λ=1e-3,activation = tanh)
    num_hidden  = 2*N*N+1
    data1,data2 = gen_data(N = N,ntrain=ntrain)
    data3,data4 = gen_data(N = N,ntrain=ntest)
    num_metrics = size(data2,3) 
    inter_elms = gen_inter_elm(data2,data1,num_hidden = num_hidden, activation = activation, λ = λ) 
    out_train = eval_inter_elm(inter_elms,data2)
    out_valid = eval_inter_elm(inter_elms,data4)
   
    w = ones(num_metrics)

    model_ =  ELM(combine(out_train,w),data1,num_hidden = num_hidden,activation=activation) |> gpu
    
    model(x) = model_( combine(x,w) ) .> 0.5
    acc(m,x,y) = mean( m(x) .== y)
    
    training_acc,validation_acc = acc(model,out_train,data1),acc(model,out_valid,data3)
    training_f1, validation_f1 = f1(model,out_train,data1),f1(model,out_valid,data3)
    recall_train, recall_valid = rec(model,out_train,data1),rec(model,out_valid,data3)
    precision_train, precision_valid = prec(model,out_train,data1),prec(model,out_valid,data3)
    println("=========== REPORT FOR COMBINING corr and mutual =============")
    println("date : ",Dates.format(now(),"YYYY-mm-dd-HHhMM"))
    println("results for N =$N, num_hidden = $num_hidden")
    println("Training   acc = $training_acc")
    println("Validation acc = $validation_acc")
    println("Training   f1  = $training_f1")
    println("Validation f1  = $validation_f1")
    println("Training   recall = $recall_train")
    println("Validation recall = $recall_valid")
    println("Training   precision = $precision_train")
    println("Validation precision = $precision_valid")
end

experiment_combine_cor_mif_uniform()

function experiment_combine_cor_mif_ensemble(; N = 20 , ntrain = 100, ntest = 20, λ=1e-3,activation = tanh,
                                             num_ensemble = 100)
    num_hidden  = 2*N*N+1
    data1,data2 = gen_data(N = N,ntrain=ntrain)
    data3,data4 = gen_data(N = N,ntrain=ntest)
    num_metrics = size(data2,3)
    inter_elms = gen_inter_elm(data2,data1,num_hidden = num_hidden, activation = activation, λ = λ)
    out_train = eval_inter_elm(inter_elms,data2)
    out_valid = eval_inter_elm(inter_elms,data4)

    #loss(m,x,y) = mean( m(x) .== y )
    train_f1s = []
    valid_f1s = [] 
    losses = []
    coeffs = []
    w = ones(num_metrics)
    for k=1:num_ensemble
        model_ =  ELM(combine(out_train,w),data1,num_hidden = num_hidden,activation=activation) |> gpu
        CUDA.synchronize()
        model(x) = model_( combine(x,w) ) .> 0.5
        
        training_loss   = loss(model,out_train,data1)
        validation_loss = loss(model,out_valid,data3)
        CUDA.synchronize()
        push!(losses,validation_loss+training_loss)
        push!(coeffs,copy(w))
        push!(train_f1s,f1(model,out_train,data1))
        push!(valid_f1s,f1(model,out_valid,data3))
        w = randn(num_metrics)
    end

    println("=========== REPORT FOR COMBINING corr and mutual =============")
    println("date : ",Dates.format(now(),"YYYY-mm-dd-HHhMM"))
    println("results for N =$N, num_hidden = $num_hidden")
    println("minimal loss : ",minimum(losses))
    return losses,coeffs, train_f1s, valid_f1s
    # plot(sort(losses))
end

losses, coeffs, train_f1s, valid_f1s = experiment_combine_cor_mif_ensemble()

    
    # for epoch = 1:epochs
    #     training_loss,grads = Flux.withgradient(w) do            
    #         combined_train = combine(out_train, w[1] |> cpu)
    #         combined_valid = combine(out_valid, w[1] |> cpu)
    #         prediction = ELM(combined_training,data1,num_hidden=num_hidden,λ=1e-3,activation=activation)
    #         return loss(prediction, combined_train,data1)+loss(prediction, combined_valid,data3)
    #     end
    #     Flux.update!(opt,w,grads[1])
    #     println("Epoch $epoch :: loss $training_loss")        
    # end
    # return prediction
# end


# function experiment_cor_cov_cpu(; N = 20 , pars= Dict("a"=>0.045, "b"=> 0.1) , ρ = 0.4  , steps = 125 ,
#                             ntrain = 100, ntest = 20,epochs = 100)
#     num_hidden = 2*N*N+1
#     activation = tanh
#     X0=rand(N) .< ρ
#     A = zeros(Int16,(N,N))
#     B = zeros(Float32,(steps,N))
#     M = zeros(Float32,(N,N))
#     C = zeros(Float32,(N,N))
#     data1 = zeros(Float32,(N*N,ntrain))
#     data2 = zeros(Float32,(N*N,ntrain,6))
#     @showprogress "Generating training data..." for k = 1:ntrain
#         A .= gen_adj(N , p = 0.5 + 0.1*randn())  |> cpu
#         B .= gen_tep(A, X0, pars= pars ,  steps= steps) |> cpu
#         M .= get_mutual_entropy_matrix( B ) 
#         C .= B'B
#         data1[:,k] .= vec(A)  ;
#         data2[:,k,1] .= vec(M)
#         data2[:,k,2] .= vec(cor(M , A ))
#         data2[:,k,3] .= vec(cov(M , A ))
#         data2[:,k,4] .= vec(C)
#         data2[:,k,5] .= vec(cor(C ,A )) 
#         data2[:,k,6] .= vec(cov(C ,A )) 
#     end
    
#     datat1 = zeros(Int16,(N*N,ntest))
#     datat2 = zeros(Float32,(N*N,ntest,6))
#     @showprogress "Generating validation data..." for j=1:ntest
#         A .= gen_adj(N, p = 0.5 +0.1*randn())  |> cpu
#         B .= gen_tep(A, X0, pars= pars ,  steps= steps) |> cpu
#         M .= get_mutual_entropy_matrix( B ) 
#         C .= B'B
#         datat1[:,j] = vec( A )
#         datat2[:,j,1] .= vec( M )
#         datat2[:,j,2] .= vec(cor(M , A ))
#         datat2[:,j,3] .= vec(cor(M , A ))
#         datat2[:,j,4] .= vec(C)
#         datat2[:,j,5] .= vec(cor(C , A )) 
#         datat2[:,j,6] .= vec(cov(C , A )) 
#     end

    
#     m = [ELM_cpu(data2[:,:,k], data1,num_hidden = num_hidden,λ = 1e-3, activation=activation) for k=1:6]
#     a = ones(Float32,6)
#     @info "Training entry layers... done" size(m)
#     output_training     = replace( [ m[k](data2[:,:,k])   for k=1:6], NaN => 0)
#     output_validation   = replace( [ m[k](datat2[:,:,k])  for k=1:6], NaN => 0)


    
#     combined_training   = zeros(Float32,N*N)
#     combined_validation = zeros(Float32,N*N)
    
#     combined_training   = replace(reduce(+,[output_training[k]*a[k]   for k=1:6]),NaN => 0) 
#     combined_validation = replace(reduce(+,[output_validation[k]*a[k] for k=1:6]),NaN => 0) 

#     # combined_training   = reduce(+,[output_training[k]*a[k]   for k=1:6]) |> gpu
#     # combined_validation = reduce(+,[output_validation[k]*a[k] for k=1:6]) |> gpu


    
#     prediction = ELM_cpu(combined_training,data1,num_hidden=num_hidden,λ=1e-3,activation=activation)        
#     @info "training   loss::" loss(prediction, combined_training  , data1)    
#     @info "validation loss::" loss(prediction, combined_validation, datat1)

#     #return output_training, output_validation, m, prediction,data1,data2,datat1,datat2


#     a_final = zeros(6)
#     current_loss = N*N*1e0
#     #@showprogress
#     for epoch = 1:epochs
#         #combined_training   = reduce(+,[output_training[k]*a[k]   for k=1:6]) |> gpu
#         #combined_validation = reduce(+,[output_validation[k]*a[k] for k=1:6]) |> gpu
#         a = randn(6)
#         combined_training   = replace(reduce(+,[output_training[k]*a[k]   for k=1:6]),NaN => 0) 
#         combined_validation = replace(reduce(+,[output_validation[k]*a[k] for k=1:6]),NaN => 0) 
#         prediction = ELM_cpu(combined_training,data1,num_hidden=num_hidden,λ=1e-3,activation=activation) 
#         #grads = Zygote.gradient(a -> loss(prediction,combined_training,data1) )        
#         #a -= 0.01*grads
#         training_loss = loss(prediction,combined_training,data1)+loss(prediction,combined_validation,datat1)
#         if training_loss < current_loss
#             a_final .= a*1e0
#             current_loss = training_loss
#             println("Epoch $epoch :: loss $training_loss")
#             #println(a_final)
#         end
#     end
#     combined_training   = replace(reduce(+,[output_training[k]*a_final[k] for k=1:6]),NaN => 0) 
#     combined_validation = replace(reduce(+,[output_validation[k]*a_final[k] for k=1:6]),NaN => 0) 
#     prediction = ELM_cpu(combined_training,data1,num_hidden=num_hidden,λ=1e-3,activation=activation)
#     training_loss = loss(prediction,combined_training,data1)+loss(prediction,combined_validation,datat1)
#     println("loss $training_loss")
# end



# function ELM_cpu(X, T ; num_hidden = 2048 , activation=relu,λ=1e-3)    
#     input_size = size(X, 1)
#     output_size = size(T, 1)
#     factor = sqrt(2e0 / (input_size+num_hidden))
#     W_hidden = randn(Float32, num_hidden, input_size) * factor  
#     b_hidden = randn(Float32, num_hidden, 1)          
#     H = activation.( W_hidden * X .+ b_hidden)  
#     β = transpose((H * H' + λ*I) \ (   H * (T')   )) 
#     #@info "Finished training ELM" size(β)
#     return (x) -> β * activation.( W_hidden * x .+ b_hidden) 
# end


