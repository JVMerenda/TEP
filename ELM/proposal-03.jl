include("tools.jl")
using Random, CUDA

function experiment_patches()
    Random.seed!(1276312)
    N,ntrain,multi,npatches,L = 20,100,2,100,5
    d1,d2 = gen_data(N = N,multi = multi,ntrain=ntrain)

    X = zeros(Float32,3*L*L,ntrain*multi*npatches)
    Y = zeros(Float32,1*L*L,ntrain*multi*npatches)
    for k=1:ntrain*multi
        cols  = get_patch_idx(N, L = L,num_patches = npatches)
        A,M   = reshape(d1[:,k],N,N), reshape(d2[:,k,1],N,N)
        AA,MM = get_patches(A |> cpu,cols) ,get_patches(M |> cpu,cols)
        i1 = (k-1)*npatches
        for j = 1:npatches 
            Q = MM[j] / (sqrt(tr(MM[j]'MM[j])) + 1e-12)
            X[:,i1+j] = vcat((Q)...,(Q^2)...,(Q^3)...)
            Y[:,i1+j] = vec(AA[j])
        end
    end
    model = ELM(X |> gpu,Y |> gpu,num_hidden=4*3*L*L+1,activation=tanh)
    return model,X,d2,Y,d1
end

model, X, d2, Y, d1 = experiment_patches()