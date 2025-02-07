
gpu_id     = parse(Int8,ARGS[1])
N          = parse(Int8,ARGS[2])
size_train = parse(Int16,ARGS[3])
size_test  = parse(Int16,ARGS[4])
epochs     = parse(Int16,ARGS[5])
m          = (parse(Int16,ARGS[6]),parse(Int16,ARGS[7]))
learning_rate = 0.001

println(epochs)

using BSON, Dates
include("testing.jl")
CUDA.device!(gpu_id)
p_train,p_test = mod.(0.5 .+ 0.1*rand(size_train),1e0),mod.(0.5 .+ 0.1*rand(size_test),1e0)
data_train = gen_data(N=N, p = p_train) |> gpu;
data_test  = gen_data(N=N, p = p_test ) ;

model = Chain(Dense(N*N, m[1], relu),Dense(m[1],m[2],relu),Dense(m[2], N*N,sigmoid)) |> gpu
loss(mm, x, y) = Flux.binarycrossentropy(mm(x),y)  |> gpu
opt = Flux.setup(Adam(learning_rate), model) |> gpu
v = [] |>gpu
@showprogress "Training..." for epoch in 1:epochs
    append!(v, [ mean([loss(model, x, y) for (x, y) in data])  ])
    Flux.train!(loss, model, data, opt)
end

model_ = model |> cpu
q = [sum((model_(x) .> 0.5) .== y)/10_000 for (x,y) in data_test] ;
q_avg = mean(q)
q_std = std(q)
u = Dates.format(now(),"yyyy-mm-dd::HHhMM")
open("results_$u.txt","a") do file
    println("================================================")
    println(u)
    println("The test concluded with accuracy $q_avg ± $q_std")
    println("------------------------------------------------")
    println("ARGS :: ",ARGS...)
end
BSON.@save "results_$u.bson" model_,( v|> cpu ), data_train, data_test




