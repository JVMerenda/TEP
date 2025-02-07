using StatsPlots, Turing

include("proposal-01.jl")



function experiment_bayesian_01()

    @model function mif_corr(x, y, z ; m = [0,0], s2= [100,100] )
        n = length(y)
        λ ~ Normal(m[1],s2[1])
        β ~ Normal(m[2],s2[2])
        for k=1:n
            x[k] ~ Bernoulli( 1e0/(1e0+exp(λ * y[k]+ β *z[k])) )
        end
        return x
    end
    
    N = 20 ; ntrain = 100 ; ntest = 20 ; d1,d2 = gen_data(N = N, ntrain = ntrain,multi = 2) ;  
    X,Y,Z = vec(d1 |> cpu), vec(d2[:,:,1] |> cpu), vec(d2[:,:,3] |> cpu)
    chain = sample(mif_corr(X,y=Y,z=Z),NUTS(), 5_000)
    println(mean(chain[:λ])," ± ",std(chain[:λ])); println(mean(chain[:β])," ± ",std(chain[:β]));
    plot(chain)
end



function experiment_bayesian_02()

    @model function mif_corr(x,y ; m = [0,0,0], s2= [100,100,100] )
        n = length(y)
        a ~ Normal(m[1],s2[1])
        b ~ Normal(m[2],s2[2])
        c ~ Normal(m[3],s2[3])
        #β ~ Normal(m[4],s2[4])
        for k=1:n
            x[k] ~ Bernoulli( 1e0/(1e0+exp( ( a + b * y[k] + c *log(y[k]+1e-12)  ) ) ))     
        end
        return x
    end
    
    N = 20 ; ntrain = 10 ; multi = 2 ;
    d1,d2 = gen_data(N = N, ntrain = ntrain,multi = multi) ;  
    X,Y = vec(d1 |> cpu), vec(d2[:,:,1] |> cpu) 
    chain = sample(mif_corr(X,Y),NUTS(), 5_000)

    plot(chain)
    return X,Y,mif_corr,chain
end


