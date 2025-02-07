include("tools.jl")
using LaTeXStrings

function get_spacings(N,X)
    A = get_eigs(N,X)
    return A[2:end,:] - A[1:end-1,:]
end

function experiment_spacings()
    N,ntrain,multi=20,50,100
    d1,d2 = gen_data(N=N,ntrain=ntrain,multi=multi)
    spacings_adj = get_spacings(N,d1        |> cpu)
    spacings_mif = get_spacings(N,d2[:,:,1] |> cpu)
    #average over multiples with same adj
    avg_mif = hcat([mean(spacings_mif[:,(1+(k-1)*multi):k*multi],dims=2) for k=1:ntrain]...)
    var_mif = hcat([ var(spacings_mif[:,(1+(k-1)*multi):k*multi],dims=2) for k=1:ntrain]...)
    
    avg_adj = hcat([mean(spacings_adj[:,(1+(k-1)*multi):k*multi],dims=2) for k=1:ntrain]...)
    var_adj = hcat([ var(spacings_adj[:,(1+(k-1)*multi):k*multi],dims=2) for k=1:ntrain]...)
    
    avg_deg = mean(d1,dims=1)[1:multi:end]'
    # "integral" over mif
    
    sum_adj = sortslices([avg_deg' sum(avg_mif,dims=1)' sqrt.(sum(var_mif,dims=1))'] |> cpu,dims=1) 
    sum_mif = sortslices([avg_deg' sum(avg_mif,dims=1)' sqrt.(sum(var_mif,dims=1))'] |> cpu,dims=1)
    sum_mif2= sortslices([avg_deg' sum(avg_mif.^2,dims=1)' sqrt.(sum(var_mif.^2,dims=1))'] |> cpu,dims=1) 
    

    plot(sum_mif[:,1],sum_mif[:,2],ribbon=sum_mif[:,3], fillalpha=0.2,linealpha=0, c=:1,label=nothing)
    plot!(sum_mif[:,1],sum_mif[:,2],marker=:circle,c=:1,linealpha=0,label=nothing)
    plot!(title="Acc mean eigenspacing for MIF (ER-network)",xlabel=L"p",ylabel=L"\sum_k{[\langle \omega_{k+1}\rangle-\langle \omega_k \rangle ]}")
    savefig("results-spectral-spacing.pdf")
    
    plot(sum_mif2[:,1],sum_mif2[:,2],ribbon=sum_mif2[:,3], fillalpha=0.2,linealpha=0, c=:1,label=nothing)
    plot!(sum_mif2[:,1],sum_mif2[:,2],marker=:circle,c=:1,linealpha=0,label=nothing)
    plot!(title="Acc squared-eigenspacing for MIF (ER-network)",xlabel=L"p",ylabel=L"\sum_k{[\langle \omega_{k+1}\rangle-\langle \omega_k \rangle ]^2}")
    savefig("results-spectral-spacing-squared.pdf")    
end



