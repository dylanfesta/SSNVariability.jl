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
  s1,s2,ρ = 6.0,3.0,0.15 # variance and correlation of noise
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

### Single unit distribution

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

### Bivariate distribution

Now, I consider the correlation between excitatory and inhibitory units.

````@example linear_mean_and_variance
theplot = let plt=plot()
  k=10
  scatter(ei_less[1,1:k:end],ei_less[2,1:k:end];
    label=L"numeric $r_{\mathrm{exc}}(t)$ vs $r_{\mathrm{inh}}(t)$",
    color=:black,alpha=0.5)
end


#= To compare this with analytic, we can print the analytic bivariate distribution
as a series of ellypses. Code inspired to a blog post by "David Gold" with Python Code
reinterpreted by chatGPT4 and debugged by me.
=#

function plot_bivariate_quantile_ellipse(mean::Vector{Float64}, covariance::Matrix{Float64},
    plot_quantile::Float64 ; npoints::Int64=100)
  # Validate input dimensions
  @assert length(mean) == 2 "Mean must be a 2-element vector"
  @assert size(covariance) == (2, 2) "Covariance must be a 2x2 matrix"
  @assert plot_quantile > 0 && plot_quantile < 1 "Quantile must be between 0 and 1"

  # Calculate quantile of chi-squared distribution with 2 degrees of freedom
  chisq_val = quantile(Chisq(2), plot_quantile)
  # Eigen decomposition
  (Eval, Evec) = eigen(covariance)
  # Calculate radii of the ellipse
  x_radius = sqrt(chisq_val * Eval[1])
  y_radius = sqrt(chisq_val * Eval[2])
  x_vec= [1 ; 0] # vector along x-axis
  # Rotation angle from the eigenvector
  cosrotation = -dot(x_vec,Evec[:,2])/(norm(x_vec)*norm(Evec[:,2]))
  rotation =pi/2-acos(cosrotation)
  # Rotation matrix
  R = [cos(rotation) -sin(rotation); sin(rotation) cos(rotation)]
  # Parametric equations of the ellipse before rotation
  theta = range(0, 2*pi, length=npoints)
  x_ellipse = x_radius * cos.(theta)
  y_ellipse = y_radius * sin.(theta)
  # Rotate and translate the ellipse
  rotated_coords = R * [x_ellipse y_ellipse]'
  x_plot = rotated_coords[1, :] .+ mean[1]
  y_plot = rotated_coords[2, :] .+ mean[2]
  return x_plot, y_plot
end



theplot = let plt=plot()
  k=10
  scatter!(plt,ei_less[1,1:k:end],ei_less[2,1:k:end];
    label=L"numeric $r_{\mathrm{exc}}(t)$ vs $r_{\mathrm{inh}}(t)$",
    color=:black,alpha=0.5)
  # 50% quantile
  (xell,yell) = plot_bivariate_quantile_ellipse(mu_an,sigma_an,0.5)
  plot!(plt,xell,yell;
    label="analytic 50% quantile",
    color=colorant"forest green",
    linewidth=3, alpha=0.5)
  # 95% quantile
  (xell,yell) = plot_bivariate_quantile_ellipse(mu_an,sigma_an,0.95)
  plot!(plt,xell,yell;
    label="analytic 95% quantile",
    color=colorant"forest green",
    linewidth=3,alpha=0.5)
  xlabel!(plt,"rate exc (Hz)")
  ylabel!(plt,"rate inh (Hz)")
  # mean as a big point
  scatter!(plt,[mu_an[1]],[mu_an[2]],label="",
    color=colorant"green",marker=:circle,markersize=8)
end
````

### Covariance density in time

Finally, I show the covariance density in time,
once again comparing analytic and numeric.

````@example linear_mean_and_variance
function compute_covariance_density(X::AbstractVector{R},Y::AbstractVector{R},nsteps::Integer) where R
  @assert length(X) == length(Y) "vector must have same length"
  # forward shift
  ret = Vector{Float64}(undef,2*nsteps-1)
  mean_x = mean(X)
  mean_y = mean(Y)
  ndt_tot = length(X)
  binned_sh = similar(X)
````

0 and forward

````@example linear_mean_and_variance
  @simd for k in 0:nsteps-1
    circshift!(binned_sh,Y,-k)
    ret[nsteps-1+k+1] = dot(X,binned_sh)
  end
````

backward shift

````@example linear_mean_and_variance
  @simd for k in 1:nsteps-1
    circshift!(binned_sh,Y,k)
    ret[nsteps-k] = dot(X,binned_sh)
  end
  @. ret = (ret /ndt_tot) - mean_x*mean_y
end


theplot = let ncov = 500
  _times = range(start=-dt_save*ncov,stop=dt_save*ncov,length=2*ncov-1)
  cov_ei = compute_covariance_density(ei_less[1,:],ei_less[2,:],500)
  cov_ei ./= dt_save^2
  plt=plot()
  plot!(plt,_times,cov_ei;
    label="numeric",
    color=:black,linewidth=3,alpha=0.5,
    xlabel="time (s)",ylabel="covariance exc vs inh")
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

