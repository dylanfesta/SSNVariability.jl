```@meta
EditURL = "../../examples/linear_mean_and_variance.jl"
```

# Mean and variance in a linear 2D system

Here I don't make use of the algorithm by Hennequin et al, but simply show how to capture
mean and variance analytically in a linear system.

The main sources are:

# Initialization

````@example linear_mean_and_variance
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors,LaTeXStrings ; theme(:default) ; gr()

using InvertedIndices
using Random
Random.seed!(0)

using SSNVariability; const global S = SSNVariability

#= ##  Step 1: create a linear system

For simplicity I just consider a 2D model with one excitatory and one
inhibitory unit.

I simulate a dynamics of the form:
```math
\frac{d\, r(t)}{dt} = - r + W \, r + h
```
where $r(t)$ and $h$ are 2D vectors. The results however work for any linear system.
=#

ntw = let
  iofun=S.ReLU(1.0)
  weight_matrix = 0.2 .* [2.0 -0.4 ;
                          4. -0.1]
  time_membrane = [ 0.1 , 0.05]
  input = [25. , 15.0]
  s1,s2,ρ = 6.0,3.0,0.1 # variance and correlation of noise
  sigma_noise = Symmetric([ s1  s1*s2*ρ ; s1*s2*ρ s2 ])
  @assert isposdef(sigma_noise)
  S.RecurrentNeuralNetwork(iofun,weight_matrix,time_membrane,input,sigma_noise)
end;
nothing #hide
````

This function comutes mean and covariance analytically

````@example linear_mean_and_variance
mu_an,sigma_an = S.mean_cov_linear_ntw(ntw);
@info mu_an
````

## Step 2: simulate the system and compare the results
Now we simulate the system and compare the results.
It will work for pretty much any initial condition.

````@example linear_mean_and_variance
r_start = mu_an .* 2.0 .* rand(2)
````

N.B. the ODE solver determines the actual time step
this is just the timestep used for *saving* the result

````@example linear_mean_and_variance
dt_save = 0.01
Ttot = 200.0
t,_,ei=S.run_network_noise(ntw,r_start,Ttot;stepsize=dt_save)
````

I will consider the first half as warmup

````@example linear_mean_and_variance
idx_t_keep =findfirst(>(Ttot/2),t) : length(t)
ei_less = view(ei,:,idx_t_keep)
sigma_num = cov(ei_less;dims=2)
mu_num = mean(ei_less;dims=2)[:]
````

Plot rate Vs analytic

````@example linear_mean_and_variance
theplot = let plt=plot()
  hline!(plt,[mu_an[1]],label=L"analytic $\mu_{\mathrm{exc}}$",
    color=colorant"dark blue",linestyle=:dash)
  plot!(plt,t,ei[1,:],label=L"numeric $r_{\mathrm{exc}}(t)$",color=:blue)
  ylims!(plt,10,60)
  xlims!(plt,0,10)
  hline!(plt,[mu_an[2]],label=L"analytic $\mu_{\mathrm{inh}}$",
    color=colorant"dark red",linestyle=:dash)
  plot!(plt,t,ei[2,:],label=L"numeric $r_{\mathrm{inh}}(t)$",color=:red)
end
````

Plot distribution Vs analytic for excitatory units

````@example linear_mean_and_variance
theplot = let plt=plot()
  histogram!(plt,ei_less[1,:];
    label=L"numeric $r_{\mathrm{exc}}(t)$",color=:blue,
    alpha=0.5,nbins=100,normalize=:pdf)
  mu_e_an = mu_an[1]
  sigma_e_an = sqrt(sigma_an[1,1])
  xs = range(mu_e_an - 4.0*sigma_e_an,stop=mu_e_an + 4.0*sigma_e_an,length=100)
  ys = pdf.(Normal(mu_e_an,sigma_e_an),xs)
  plot!(plt,xs,ys;
    label=L"analytic $r_{\mathrm{exc}}(t)$",
    color=colorant"dark blue",linestyle=:dash,
    linewidth=3)
  xlabel!(plt,"rate (Hz)")
  ylabel!(plt,"probability density")
  plt
end
````

same for inhibitory units

````@example linear_mean_and_variance
theplot = let plt=plot()
  histogram!(plt,ei_less[2,:];
    label=L"numeric $r_{\mathrm{inh}}(t)$",color=:red,
    alpha=0.5,nbins=100,normalize=:pdf)
  mu_i_an = mu_an[2] # change to the mean of inhibitory units
  sigma_i_an = sqrt(sigma_an[2,2]) # change to the standard deviation of inhibitory units
  xs = range(mu_i_an - 4.0*sigma_i_an,stop=mu_i_an + 4.0*sigma_i_an,length=100)
  ys = pdf.(Normal(mu_i_an,sigma_i_an),xs)
  plot!(plt,xs,ys;
    label=L"analytic $r_{\mathrm{inh}}(t)$",
    color=colorant"dark red",linestyle=:dash,
    linewidth=3)
  xlabel!(plt,"rate (Hz)")
  ylabel!(plt,"probability density")
  plt
end
````

Now, the correlation between the two

````@example linear_mean_and_variance
theplot = let plt=plot()
  k=10
  scatter(ei_less[1,1:k:end],ei_less[2,1:k:end];
    label=L"numeric $r_{\mathrm{exc}}(t)$ vs $r_{\mathrm{inh}}(t)$",
    color=:black,alpha=0.5)
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

