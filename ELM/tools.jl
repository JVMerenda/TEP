using LinearAlgebra, StatsBase, CUDA, ProgressMeter , Flux



# ====================================================================
# Extreme Learning Machine (ELM) implementation for regression/classification
function ELM(X, T; num_hidden=2048, activation=relu, λ=1e-3)
    # Inputs:
    # X: Input data matrix (features x samples)
    # T: Target output matrix (outputs x samples)
    # num_hidden: Number of hidden neurons in the single hidden layer
    # activation: Activation function for the hidden layer (default: relu)
    # λ: Regularization parameter for ridge regression (default: 1e-3)
    
    input_size = size(X, 1)  # Number of input features
    output_size = size(T, 1) # Number of output targets    
    factor = sqrt(2e0 / (input_size + num_hidden))    
    # Randomly initialize hidden layer weights and biases on GPU
    W_hidden = CUDA.randn(Float32, num_hidden, input_size) * factor  # Weight matrix
    b_hidden = CUDA.randn(Float32, num_hidden, 1)                   # Bias vector    
    # Compute hidden layer output (H = activation(W_hidden * X + b_hidden))
    H = activation.(W_hidden * X .+ b_hidden) |> gpu    
    # Solve for output weights (β) using ridge regression:
    # β = (H * H' + λ * I) \ (H * T') 
    # Ridge regression adds λ * I for regularization to handle overfitting.
    β = transpose((H * H' + λ * I) \ (H * T')) |> gpu    
    # Return the trained ELM model as a function
    # The function computes outputs for new inputs using the trained weights
    return (x) -> β * activation.(W_hidden * x .+ b_hidden)
end



# Generate a random adjacency matrix
function gen_adj(N; p=0.4)
    # Ensure the probability p is less than 1
    @assert p < 1e0    
    # Generate a symmetric adjacency matrix with values 0 or 1
    A = Int16.(Hermitian(reshape(sample([0, 1], Weights([1e0 - p, p]), N * N), (N, N))))    
    # Remove self-loops by subtracting the diagonal
    return (A - diagm(diag(A))) 
end

# Generate time-evolution data based on an adjacency matrix and initial state
function gen_tep(A, X0; steps=110, pars=Dict("a" => 0.0004, "b" => 0.01))
    # Number of nodes in the network
    N = size(X0)[1]    
    # Move matrices and vectors to GPU for faster computation
    A_gpu = CuArray(A)
    X_gpu = CUDA.zeros(Int16, N, steps)  # Storage for time evolution
    X_gpu[:, 1] = CuArray(X0)           # Initialize first state
    Y_gpu = CUDA.zeros(Int16, N)        # Temporary storage for updates    
    # Extract parameters from the dictionary
    a = pars["a"]
    b = pars["b"]    
    # Iterate through time steps
    for k in 1:steps-1
        # Calculate the probability of flipping binary states
        Y_gpu .= CUDA.rand(N) .< (a .* (1 .- X_gpu[:, k]) .* (A_gpu * X_gpu[:, k]) + b .* X_gpu[:, k])        
        # Update the state based on flips
        X_gpu[:, k+1] .= X_gpu[:, k] .+ Y_gpu .- 2 .* (X_gpu[:, k] .* Y_gpu)
    end    
    # Return the time evolution matrix, transposed
    return X_gpu'
end

# Generate training data for a binary network model
function gen_data(; N=20, ρ=0.3, steps=120, pars=Dict("a" => 0.045, "b" => 0.1),
                  ntrain=100, multi=1)
    # Initialize the initial state with a density ρ
    X0 = CUDA.rand(N) .< ρ    
    # Initialize GPU arrays for adjacency matrices and data
    A = CUDA.zeros(Int16, (N, N))
    tep = CUDA.zeros(Float32, (steps, N))
    M = CUDA.zeros(Float32, (N, N))
    C = CUDA.zeros(Float32, (N, N))
    data1 = CUDA.zeros(Float32, (N * N, ntrain * multi))         # Adjacency matrices
    data2 = CUDA.zeros(Float32, (N * N, ntrain * multi, 4))      # Feature matrices    
    # TODO: Implement parallel threads for faster generation
    # TODO: Allow `p` to be an input parameter    
    # Loop through the number of training samples
    @showprogress "Generating training data..." for k = 1:ntrain
        # Generate a random adjacency matrix with noise in `p`
        A .= gen_adj(N, p=0.5 + 0.1 * randn()) |> gpu        
        # Generate multiple instances for each training sample
        for i = 1:multi    
            tep .= gen_tep(A, X0, pars=pars, steps=steps)                
            M .= get_mutual_entropy_matrix(tep |> cpu) |> gpu
            C .= get_corr_matrix(tep |> cpu, tep |> cpu) |> gpu
            #C .= B'B            
            # Index for storing results
            kk = (k-1)*multi + i            
            # Store adjacency matrix
            data1[:, kk] .= vec(A)            
            # Store features in data2
            data2[:, kk, 1] .= vec(M)                                      # Mutual entropy
            #data2[:, kk, 2] .= vec(get_corr_matrix(M |> cpu, A |> cpu)) |> gpu  # Correlation with M
            data2[:, kk, 2] .= vec(M * M)
            data2[:, kk, 3] .= vec(C)
            #data2[:, kk, 4] .= vec(get_corr_matrix(C |> cpu, A |> cpu)) |> gpu  # Correlation with A
            data2[:, kk, 4] .= vec(C * C)
        end
    end    
    # Return the generated datasets
    return data1, data2
end
# ====================================================================
# ====================================================================
# Calculate entropy of a discrete distribution
function get_entropy(X)
    # Compute probabilities of each unique value in X
    p = [x[2] for x in countmap(X)] # NOTE: CPU-only implementation; consider modifying for GPU    
    x0 = sum(p)    
    # Compute entropy using the formula -Σ(p * log2(p))
    return -sum([(x / x0) * log2(x / x0) for x in p])
end

# Calculate mutual entropy between two variables
function get_mutual_entropy(X, Y)
    # Compute joint probabilities of paired values in (X, Y)
    p = [x[2] for x in countmap(zip(X, Y))]
    x0 = sum(p)
    # Use the formula: H(X) + H(Y) - H(X, Y)
    return get_entropy(X) + get_entropy(Y) - sum([(x / x0) * log2(x / x0) for x in p])
end

# Generate a mutual entropy matrix from a dataset
function get_mutual_entropy_matrix(X; delay=0)
    T, N = size(X) # T: Number of time steps, N: Number of variables    
    # Initialize a matrix to store mutual entropies
    M = zeros(Float32, (N, N))   
    # Calculate mutual entropy for each pair of variables
    Threads.@threads for j = 1:N
        for i = j:N
            M[i, j] = get_mutual_entropy(X[1+delay:T, i], X[1:T-delay, j])
            M[j, i] = M[i, j] # Symmetric matrix
        end
    end
    return M
end

# ====================================================================
# Calculate the correlation matrix for a dataset
function get_corr_matrix(matrix; delay=0, eps=1e-12)
    T, N = size(matrix) # T: Number of time steps, N: Number of variables    
    # Initialize the correlation matrix
    C = zeros(Float32, N, N)    
    factor = N / (N + 1e0)    
    # Compute correlation for each pair of variables
    Threads.@threads for j in 1:N
        for i in j:N
            x, y = matrix[1+delay:T, i], matrix[1:T-delay, j]
            C[i, j] = (mean(x .* y) - mean(x) * mean(y)) / (sqrt(var(x) * var(y)) + eps)
            C[j, i] = C[i, j] # Symmetric matrix
        end
    end
    return C * factor
end

# Calculate the correlation matrix between two datasets
function get_corr_matrix(A, B; delay=0, eps=1e-12)
    T, N = size(A) # T: Number of time steps, N: Number of variables    
    # Initialize the correlation matrix
    C = zeros(Float32, N, N)    
    factor = N / (N - 1e0)    
    # Compute correlation for each pair of variables between A and B
    for i = 1:N
        for j = i:N
            x, y = A[1+delay:T, i], B[1:T-delay, j]
            C[j, i] = (mean(x .* y) - mean(x) * mean(y)) / (sqrt(var(x) * var(y)) + eps)
            C[i, j] = C[j, i] # Symmetric matrix
        end
    end
    return C * factor
end

# ====================================================================
# Compute eigenvalues of reshaped matrices within a dataset
function get_eigs(N, X)
    s = size(X) # Shape of input dataset
    idx = CartesianIndices(s[2:end]) # Indices for slicing higher dimensions    
    # Initialize storage for eigenvalues
    Y = zeros(Float32, N, s[2:end]...)    
    # Calculate eigenvalues for each reshaped matrix
    for k = 1:prod(s[2:end])
        Y[:, idx[k]] .= eigvals(reshape(view(X, :, idx[k]), (N, N))) # eigvals is CPU-only
    end
    return Y
end

# Generate random indices for patches of a matrix
function get_patch_idx(N; L=5, num_patches=10)
    # Generate `num_patches` sets of `L` unique indices
    return [sort(sample(1:N, L, replace=false)) for k = 1:num_patches]
end

# Extract patches (submatrices) of size LxL from a matrix
function get_patches(A, cols; L=5)
    num_patches = size(cols, 1) # Number of patches to extract    
    # Initialize a list to store the patches
    samples = Vector{Matrix{Float32}}(undef, num_patches)    
    # Extract patches based on provided column indices
    Threads.@threads for k = 1:num_patches
        samples[k] = A[cols[k], cols[k]]
    end
    return samples
end


