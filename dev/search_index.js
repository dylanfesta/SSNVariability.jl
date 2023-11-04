var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = SSNVariability","category":"page"},{"location":"#SSNVariability","page":"Home","title":"SSNVariability","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: )","category":"page"},{"location":"","page":"Home","title":"Home","text":":warning: :construction: Work in progress :construction:   see example section for usage","category":"page"},{"location":"#Examples","page":"Home","title":"Examples","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Linear case, match between numerics and analytic","category":"page"},{"location":"#Index","page":"Home","title":"Index","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [SSNVariability]","category":"page"},{"location":"#SSNVariability.diagtozero!-Union{Tuple{AbstractMatrix{T}}, Tuple{T}} where T","page":"Home","title":"SSNVariability.diagtozero!","text":"diagtozero!(M::AbstractMatrix{T}) where T\n\nReplaces the matrix diagonal with zeros\n\n\n\n\n\n","category":"method"},{"location":"#SSNVariability.jacobian-Union{Tuple{R}, Tuple{Vector{R}, SSNVariability.RecurrentNeuralNetwork{R}}} where R","page":"Home","title":"SSNVariability.jacobian","text":"    jacobian(u,rn::RecurrentNetwork)\n\nJacobian matrix of the network dynamics at point u\n\n\n\n\n\n","category":"method"},{"location":"#SSNVariability.norm_sum_rows!-Tuple{Any}","page":"Home","title":"SSNVariability.norm_sum_rows!","text":"norm_sum_rows!(mat)\n\nRescales the matrix by row so that the sum of each row is 1.0\n\n\n\n\n\n","category":"method"},{"location":"#SSNVariability.random_covariance_matrix","page":"Home","title":"SSNVariability.random_covariance_matrix","text":"random_covariance_matrix(dims::Integer,diag_val::Real,k_dims::Integer=5)\n\nReturns a random covariance matrix that is positive definite and has off-diagonal elements.\n\nArguments\n\nd: dimensions\ndiag_val: scaling of the diagonal\nk-dims: to regulate off-diagonal elements\n\n\n\n\n\n","category":"function"},{"location":"#SSNVariability.run_network_nonoise-Union{Tuple{R}, Tuple{SSNVariability.RecurrentNeuralNetwork{R}, Vector{R}, Real}} where R<:Real","page":"Home","title":"SSNVariability.run_network_nonoise","text":"function runnetworknonoise(ntw::RecurrentNeuralNetwork{R},rstart::Vector{R},       tend::Real; verbose::Bool=false,stepsize=0.05) where R<:Real\n\nRuns the network simulation without any external noise.\n\nArguments\n\nntw::RecurrentNeuralNetwork\nr_start::Vector : initial conditions\nt_end::Real: the maximum time considered\n\nOptional arguments\n\nverbose::Bool = false : ODE solver is verbose\nstepsize::Real = 0.05 : stepsize for saving the solution (not solver stepsize)\n\nOutputs\n\ntimes::Vector{R} : the times at which the solution is saved\nu_end::Matrix{R} : the final state\nr_end::Matrix{R} : the final state as rate\n\n\n\n\n\n","category":"method"},{"location":"#SSNVariability.run_network_to_convergence-Union{Tuple{R}, Tuple{SSNVariability.RecurrentNeuralNetwork{R}, Vector{R}}} where R","page":"Home","title":"SSNVariability.run_network_to_convergence","text":"runnetworktoconvergence(u0, rn::RecurrentNeuralNetwork ;                 tend=80. , veltol=1E-4) Runs the network as described in run_network_nonoise, but stops as soon as norm(v) / n < veltol where v is the velocity at time t. If this condition is not satisfied (no convergence to attractor), it runs until t_end and prints a warning.\n\nArguments\n\nrn::RecurrentNeuralNetwork\nr_start::Vector : initial conditions\nt_end::Real: the maximum time considered\nveltol::Real : the norm (divided by num. dimensions) for velocity at convergence\n\nOutputs\n\nu_end::Vector : the final state at convergence\nr_end::Vector : the final state at convergence as rate\n\n\n\n\n\n","category":"method"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"EditURL = \"../../examples/linear_mean_and_variance.jl\"","category":"page"},{"location":"linear_mean_and_variance/#Mean-and-variance-in-a-linear-2D-system","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"","category":"section"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"Here I don't make use of the algorithm by Hennequin et al, but simply show how to capture mean and variance analytically in a linear system.","category":"page"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"The main sources are:","category":"page"},{"location":"linear_mean_and_variance/#Initialization","page":"Mean and variance in a linear 2D system","title":"Initialization","text":"","category":"section"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"using LinearAlgebra,Statistics,StatsBase,Distributions\nusing Plots,NamedColors,LaTeXStrings ; theme(:default) ; gr()\n\nusing InvertedIndices\nusing Random\nRandom.seed!(0)\n\nusing SSNVariability; const global S = SSNVariability\n\n#= ##  Step 1: create a linear system\n\nFor simplicity I just consider a 2D model with one excitatory and one\ninhibitory unit.\n\nI simulate a dynamics of the form:\n```math\n\\frac{d\\, r(t)}{dt} = - r + W \\, r + h\n```\nwhere $r(t)$ and $h$ are 2D vectors. The results however work for any linear system.\n=#\n\nntw = let\n  iofun=S.ReLU(1.0)\n  weight_matrix = 0.2 .* [2.0 -0.4 ;\n                          4. -0.1]\n  time_membrane = [ 0.1 , 0.05]\n  input = [25. , 15.0]\n  s1,s2,ρ = 6.0,3.0,0.15 # variance and correlation of noise\n  sigma_noise = Symmetric([ s1  s1*s2*ρ ; s1*s2*ρ s2 ])\n  @assert isposdef(sigma_noise)\n  S.RecurrentNeuralNetwork(iofun,weight_matrix,time_membrane,input,sigma_noise)\nend;\nnothing #hide","category":"page"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"This function comutes mean and covariance analytically","category":"page"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"mu_an,sigma_an = S.mean_cov_linear_ntw(ntw);\n@info mu_an","category":"page"},{"location":"linear_mean_and_variance/#Step-2:-simulate-the-system-and-compare-the-results","page":"Mean and variance in a linear 2D system","title":"Step 2: simulate the system and compare the results","text":"","category":"section"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"Now we simulate the system and compare the results. It will work for pretty much any initial condition.","category":"page"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"r_start = mu_an .* 2.0 .* rand(2)","category":"page"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"N.B. the ODE solver determines the actual time step this is just the timestep used for saving the result","category":"page"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"dt_save = 0.01\nTtot = 200.0\nt,_,ei=S.run_network_noise(ntw,r_start,Ttot;stepsize=dt_save)","category":"page"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"I will consider the first half as warmup","category":"page"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"idx_t_keep =findfirst(>(Ttot/2),t) : length(t)\nei_less = view(ei,:,idx_t_keep)\nsigma_num = cov(ei_less;dims=2)\nmu_num = mean(ei_less;dims=2)[:]","category":"page"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"Plot rate Vs analytic","category":"page"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"theplot = let plt=plot()\n  hline!(plt,[mu_an[1]],label=L\"analytic $\\mu_{\\mathrm{exc}}$\",\n    color=colorant\"dark blue\",linestyle=:dash)\n  plot!(plt,t,ei[1,:],label=L\"numeric $r_{\\mathrm{exc}}(t)$\",color=:blue)\n  ylims!(plt,10,60)\n  xlims!(plt,0,10)\n  hline!(plt,[mu_an[2]],label=L\"analytic $\\mu_{\\mathrm{inh}}$\",\n    color=colorant\"dark red\",linestyle=:dash)\n  plot!(plt,t,ei[2,:],label=L\"numeric $r_{\\mathrm{inh}}(t)$\",color=:red)\nend","category":"page"},{"location":"linear_mean_and_variance/#Single-unit-distribution","page":"Mean and variance in a linear 2D system","title":"Single unit distribution","text":"","category":"section"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"Plot distribution Vs analytic for excitatory units","category":"page"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"theplot = let plt=plot()\n  histogram!(plt,ei_less[1,:];\n    label=L\"numeric $r_{\\mathrm{exc}}(t)$\",color=:blue,\n    alpha=0.5,nbins=100,normalize=:pdf)\n  mu_e_an = mu_an[1]\n  sigma_e_an = sqrt(sigma_an[1,1])\n  xs = range(mu_e_an - 4.0*sigma_e_an,stop=mu_e_an + 4.0*sigma_e_an,length=100)\n  ys = pdf.(Normal(mu_e_an,sigma_e_an),xs)\n  plot!(plt,xs,ys;\n    label=L\"analytic $r_{\\mathrm{exc}}(t)$\",\n    color=colorant\"dark blue\",linestyle=:dash,\n    linewidth=3)\n  xlabel!(plt,\"rate (Hz)\")\n  ylabel!(plt,\"probability density\")\n  plt\nend","category":"page"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"same for inhibitory units","category":"page"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"theplot = let plt=plot()\n  histogram!(plt,ei_less[2,:];\n    label=L\"numeric $r_{\\mathrm{inh}}(t)$\",color=:red,\n    alpha=0.5,nbins=100,normalize=:pdf)\n  mu_i_an = mu_an[2] # change to the mean of inhibitory units\n  sigma_i_an = sqrt(sigma_an[2,2]) # change to the standard deviation of inhibitory units\n  xs = range(mu_i_an - 4.0*sigma_i_an,stop=mu_i_an + 4.0*sigma_i_an,length=100)\n  ys = pdf.(Normal(mu_i_an,sigma_i_an),xs)\n  plot!(plt,xs,ys;\n    label=L\"analytic $r_{\\mathrm{inh}}(t)$\",\n    color=colorant\"dark red\",linestyle=:dash,\n    linewidth=3)\n  xlabel!(plt,\"rate (Hz)\")\n  ylabel!(plt,\"probability density\")\n  plt\nend","category":"page"},{"location":"linear_mean_and_variance/#Bivariate-distribution","page":"Mean and variance in a linear 2D system","title":"Bivariate distribution","text":"","category":"section"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"Now, I consider the correlation between excitatory and inhibitory units.","category":"page"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"theplot = let plt=plot()\n  k=10\n  scatter(ei_less[1,1:k:end],ei_less[2,1:k:end];\n    label=L\"numeric $r_{\\mathrm{exc}}(t)$ vs $r_{\\mathrm{inh}}(t)$\",\n    color=:black,alpha=0.5)\nend\n\n\n#= To compare this with analytic, we can print the analytic bivariate distribution\nas a series of ellypses. Code inspired to a blog post by \"David Gold\" with Python Code\nreinterpreted by chatGPT4 and debugged by me.\n=#\n\nfunction plot_bivariate_quantile_ellipse(mean::Vector{Float64}, covariance::Matrix{Float64},\n    plot_quantile::Float64 ; npoints::Int64=100)\n  # Validate input dimensions\n  @assert length(mean) == 2 \"Mean must be a 2-element vector\"\n  @assert size(covariance) == (2, 2) \"Covariance must be a 2x2 matrix\"\n  @assert plot_quantile > 0 && plot_quantile < 1 \"Quantile must be between 0 and 1\"\n\n  # Calculate quantile of chi-squared distribution with 2 degrees of freedom\n  chisq_val = quantile(Chisq(2), plot_quantile)\n  # Eigen decomposition\n  (Eval, Evec) = eigen(covariance)\n  # Calculate radii of the ellipse\n  x_radius = sqrt(chisq_val * Eval[1])\n  y_radius = sqrt(chisq_val * Eval[2])\n  x_vec= [1 ; 0] # vector along x-axis\n  # Rotation angle from the eigenvector\n  cosrotation = -dot(x_vec,Evec[:,2])/(norm(x_vec)*norm(Evec[:,2]))\n  rotation =pi/2-acos(cosrotation)\n  # Rotation matrix\n  R = [cos(rotation) -sin(rotation); sin(rotation) cos(rotation)]\n  # Parametric equations of the ellipse before rotation\n  theta = range(0, 2*pi, length=npoints)\n  x_ellipse = x_radius * cos.(theta)\n  y_ellipse = y_radius * sin.(theta)\n  # Rotate and translate the ellipse\n  rotated_coords = R * [x_ellipse y_ellipse]'\n  x_plot = rotated_coords[1, :] .+ mean[1]\n  y_plot = rotated_coords[2, :] .+ mean[2]\n  return x_plot, y_plot\nend\n\n\n\ntheplot = let plt=plot()\n  k=10\n  scatter!(plt,ei_less[1,1:k:end],ei_less[2,1:k:end];\n    label=L\"numeric $r_{\\mathrm{exc}}(t)$ vs $r_{\\mathrm{inh}}(t)$\",\n    color=:black,alpha=0.5)\n  # 50% quantile\n  (xell,yell) = plot_bivariate_quantile_ellipse(mu_an,sigma_an,0.5)\n  plot!(plt,xell,yell;\n    label=\"analytic 50% quantile\",\n    color=colorant\"forest green\",\n    linewidth=3, alpha=0.5)\n  # 95% quantile\n  (xell,yell) = plot_bivariate_quantile_ellipse(mu_an,sigma_an,0.95)\n  plot!(plt,xell,yell;\n    label=\"analytic 95% quantile\",\n    color=colorant\"forest green\",\n    linewidth=3,alpha=0.5)\n  xlabel!(plt,\"rate exc (Hz)\")\n  ylabel!(plt,\"rate inh (Hz)\")\n  # mean as a big point\n  scatter!(plt,[mu_an[1]],[mu_an[2]],label=\"\",\n    color=colorant\"green\",marker=:circle,markersize=8)\nend","category":"page"},{"location":"linear_mean_and_variance/#Covariance-density-in-time","page":"Mean and variance in a linear 2D system","title":"Covariance density in time","text":"","category":"section"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"Finally, I show the covariance density in time, once again comparing analytic and numeric.","category":"page"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"function compute_covariance_density(X::AbstractVector{R},Y::AbstractVector{R},nsteps::Integer) where R\n  @assert length(X) == length(Y) \"vector must have same length\"\n  # forward shift\n  ret = Vector{Float64}(undef,2*nsteps-1)\n  mean_x = mean(X)\n  mean_y = mean(Y)\n  ndt_tot = length(X)\n  binned_sh = similar(X)\n  # 0 and forward\n  @simd for k in 0:nsteps-1\n    circshift!(binned_sh,Y,-k)\n    ret[nsteps-1+k+1] = dot(X,binned_sh)\n  end\n  # backward shift\n  @simd for k in 1:nsteps-1\n    circshift!(binned_sh,Y,k)\n    ret[nsteps-k] = dot(X,binned_sh)\n  end\n  @. ret = (ret /ndt_tot) - mean_x*mean_y\nend\n\n\ntheplot = let ncov = 500\n  _times = range(start=-dt_save*ncov,stop=dt_save*ncov,length=2*ncov-1)\n  cov_ei = compute_covariance_density(ei_less[1,:],ei_less[2,:],500)\n  cov_ei ./= dt_save^2\n  plt=plot()\n  plot!(plt,_times,cov_ei;\n    label=\"numeric\",\n    color=:black,linewidth=3,alpha=0.5,\n    xlabel=\"time (s)\",ylabel=\"covariance exc vs inh\")\nend","category":"page"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"","category":"page"},{"location":"linear_mean_and_variance/","page":"Mean and variance in a linear 2D system","title":"Mean and variance in a linear 2D system","text":"This page was generated using Literate.jl.","category":"page"}]
}
