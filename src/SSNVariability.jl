module SSNVariability
using LinearAlgebra, Statistics
using Distributions

using OrdinaryDiffEq, StochasticDiffEq
import DiffEqNoiseProcess: CorrelatedWienerProcess


"""
    random_covariance_matrix(dims::Integer,diag_val::Real,k_dims::Integer=5)

Returns a random covariance matrix that is positive definite
and has off-diagonal elements.
# Arguments
- `d`: dimensions
- `diag_val`: scaling of the diagonal
- `k-dims`: to regulate off-diagonal elements
"""
function random_covariance_matrix(dims::Integer,diag_val::Real,k_dims::Integer=5)
  W = randn(dims,k_dims)
  S = W*W'+ Diagonal(rand(dims))
  temp_diag = Diagonal(inv.(sqrt.(diag(S))))
  S = temp_diag * S * temp_diag
  S .*= diag_val
  ret = Symmetric(S)
  return ret
end


abstract type  IOFunction{R} end
Base.Broadcast.broadcastable(x::IOFunction)=Ref(x)

function iofun!(dest::Vector{R},source::Vector{R},
    io::IOFunction{R}) where R
  @inbounds @simd for i in eachindex(dest)
    dest[i] = io(source[i])
  end
  return dest
end

function iofun(source::Vector{R},io::IOFunction{R}) where R
  return iofun!(similar(source),source,io)
end

struct IOIdentity{R} <: IOFunction{R}
  function IOIdentity()
    return new{Float64}()
  end
end
function (::IOIdentity{R})(x::R) where R
  return x
end
ioinv(x::R,::IOIdentity{R}) where R<:Real = x 
ioprime(x::R,::IOIdentity{R}) where R<:Real = 1.0


struct ReLU{R} <: IOFunction{R}
  α::R
end

function (g::ReLU)(x::Real)
  re = x < 0 ? zero(x) : g.α*x
  return re
end
ioinv(x::R,g::ReLU{R}) where R<:Real = x < zero(R) ? zero(R) : x/g.α
ioprime(x::R,g::ReLU{R}) where R<:Real = x < zero(R) ? zero(R) : g.α

struct ReQuad{R} <:IOFunction{R}
  α::R
end

function (g::ReQuad{R})(x::R) where R
  ret = x < zero(R) ? zero(R) : g.α*x*x
  return ret
end
ioinv(x::R,g::ReQuad{R}) where R =  x < zero(R) ? zero(R) : sqrt(x/g.α)
ioprime(x::R,g::ReQuad{R}) where R = x < zero(R) ? zero(R) : 2.0*g.α*x

struct RecurrentNeuralNetwork{R}
  iofun::IOFunction{R}
  weight_matrix::Matrix{R}
  time_membrane::Vector{R}
  input::Vector{R}
  sigma_noise::Symmetric{R,Matrix{R}}
end
Base.Broadcast.broadcastable(x::RecurrentNeuralNetwork)=Ref(x)
function n_neurons(rnn::RecurrentNeuralNetwork)
  return size(rnn.weight_matrix,1)
end
function Base.ndims(rnn::RecurrentNeuralNetwork)
  return size(rnn.weight_matrix,1)
end

function hasnoise(rn::RecurrentNeuralNetwork{R}) where R
  return tr(rn.sigma_noise) > 100.0*eps(R)
end

# generate weight matrix
"""
    diagtozero!(M::AbstractMatrix{T}) where T
Replaces the matrix diagonal with zeros
"""
function diagtozero!(M::AbstractMatrix{T}) where T
    ms = min(size(M)...)
    zz = zero(T)
    @inbounds @simd for i in 1:ms
        M[i,i] = zz
    end
    return nothing
end

"""
    norm_sum_rows!(mat)
Rescales the matrix by row so that the sum of each row is 1.0
"""
function norm_sum_rows!(mat)
    normf=abs.(inv.(sum(mat,dims=2)))
    return broadcast!(*,mat,mat,normf)
end

const wmat2D_default=[ 2.5  -1.3
                 2.4   -1.  ]

input_base_default(n) =fill(7.,n)
time_membrane_default(ne,ni)=vcat(fill(0.02,ne),fill(0.01,ni))

 # two populations means Dale!
function make_wmat(ne::I,ni::I,w2D::Matrix{<:Real} ; noautapses=true) where I<: Integer
    @assert all(size(w2D) .== 2)
    if ne+ni==2 #nothing to do
        return w2D
    end
    # else, expand the 2D
    Wμ = w2D
    d = Exponential(1.0)
    W=rand(d,ne+ni,ne+ni)
    noautapses && diagtozero!(W)
    # now normalize by block
    pee = 1:ne
    pii = (1:ni) .+ ne
    let wee =  view(W,pee,pee)
        norm_sum_rows!(wee)
        wee .*= Wμ[1,1]
    end
    let wii =  view(W,pii,pii)
        norm_sum_rows!(wii)
        wii .*= Wμ[2,2]
    end
    let wei =  view(W,pee,pii) # inhibitory to excitatory
        norm_sum_rows!(wei)
        wei .*= Wμ[1,2]
    end
    let wie =  view(W,pii,pee) # excitatory to inhibitory
        norm_sum_rows!(wie)
        wie .*= Wμ[2,1]
    end
    return W
end

function make_wmat(ne::I,ni::I; noautapses=true) where I<: Integer
    return make_wmat(ne,ni,wmat2D_default;noautapses=noautapses)
end


# constructor with default values
function RecurrentNeuralNetwork(ne::Integer,ni::Integer;
      sigma_noise::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
      iofunction::IOFunction= ReQuad(0.02))
    wmat = make_wmat(ne,ni)
    ntot=ne+ni
    input=input_base_default(ntot)
    taus=time_membrane_default(ne,ni)
    sigma_noise=something(sigma_noise,Symmetric(zeros(ntot,ntot)))
    return RecurrentNeuralNetwork(iofunction,wmat,taus,input,sigma_noise)
end


function scalebytime!(v::Vector{R},rnn::RecurrentNeuralNetwork{R}) where R
  v ./= rnn.time_membrane
  return v
end
function iofun!(dest::Vector{R},source::Vector{R},
    ntw::RecurrentNeuralNetwork{R}) where R
  return iofun!(dest,source,ntw.iofun)
end
function iofun(x::Real,ntw::RecurrentNeuralNetwork)
  return ntw.iofun(x)
end
function ioinv!(dest::Vector{R},source::Vector{R},
    ntw::RecurrentNeuralNetwork{R}) where R
  for i in eachindex(dest)
    @inbounds dest[i] = ioinv(source[i],ntw.iofun)
  end
  return dest
end
function ioinv(x::Real,ntw::RecurrentNeuralNetwork)
  return ioinv(x,ntw.iofun)
end
function ioprime(x::Real,ntw::RecurrentNeuralNetwork)
  return ioprime(x,ntw.iofun)
end

# Jacobian!

struct JGradPars{R}
    weights::Matrix{R}
    u::Matrix{R}
    ddgu_alloc::Vector{R}
    function JGradPars(ntw::RecurrentNeuralNetwork{R}) where R
        w = similar(ntw.weight_matrix)
        u = similar(w)
        ddgu_alloc = similar(ntw.time_membrane)
        new{R}(w,u,ddgu_alloc)
    end
end

function jacobian!(J,gradpars::Union{Nothing,JGradPars},
            u::Vector{R},dgu::Vector{R},ntw::RecurrentNeuralNetwork{R}) where R
    broadcast!(*,J, ntw.weight_matrix, transpose(dgu)) # multiply columnwise
    J -=I
    #normalize by taus, rowwise
    broadcast!(/,J,J,ntw.time_membrane)
    isnothing(gradpars) && return J
    @error "This part is not ready, yet!"
    # GRADIENTS !  W first
    #gradpars.weights .= gradpars.inv_taus * transpose(dgu)
    #broadcast!(/,gradpars.weights,transpose(dgu),ntw.membrane_taus)
    # now u
    #ddg!(gradpars.ddgu_alloc,u,ntw.gain_function)
    #broadcast!(*,gradpars.u,gradpars.inv_taus,
    #    ntw.weights, transpose(gradpars.ddgu_alloc))
    return J
end

"""
        jacobian(u,rn::RecurrentNetwork)
Jacobian matrix of the network dynamics at point u
"""
@inline function jacobian(u::Vector{R},rn::RecurrentNeuralNetwork{R}) where R
    J = similar(rn.weight_matrix)
    dgu = ioprime.(u,rn)
    return jacobian!(J,nothing,u,dgu,rn::RecurrentNeuralNetwork)
end

function spectral_abscissa(u::Vector{R},rn::RecurrentNeuralNetwork{R}) where R
    J=jacobian(u,rn)
    return maximum(real.(eigvals(J)))
end

### Dynamics


function du_nonoise(u::V,ntw::RecurrentNeuralNetwork{R}) where {R<:Real,V<:Vector{R}}
  return du_nonoise!(similar(u),u,ntw)
end
function du_nonoise!(dest::V,u::V,
        ntw::RecurrentNeuralNetwork{R}) where {R<:Real,V<:Vector{R}}
  r = ntw.iofun.(u)
  dest .= ntw.input .- u
  BLAS.gemv!('N',1.,ntw.weight_matrix,r,1.,dest)
  return scalebytime!(dest,ntw)
end

"""
  function run_network_nonoise(ntw::RecurrentNeuralNetwork{R},r_start::Vector{R},
      t_end::Real; verbose::Bool=false,stepsize=0.05) where R<:Real

Runs the network simulation without any external noise.
# Arguments
- `ntw::RecurrentNeuralNetwork`
- `r_start::Vector` : initial conditions
- `t_end::Real`: the maximum time considered
# Optional arguments
- `verbose::Bool = false` : ODE solver is verbose
- `stepsize::Real = 0.05` : stepsize for saving the solution (not solver stepsize)
# Outputs
- `times::Vector{R}` : the times at which the solution is saved
- `u_end`::Matrix{R} : the final state
- `r_end`::Matrix{R} : the final state as rate
"""
function run_network_nonoise(ntw::RecurrentNeuralNetwork{R},r_start::Vector{R},
    t_end::Real; verbose::Bool=false,stepsize=0.05) where R<:Real
  u0=ioinv.(r_start,ntw)
  f(du,u,p,t) = du_nonoise!(du,u,ntw)
  prob = ODEProblem(f,u0,(0.,t_end))
  solv = solve(prob,Tsit5();verbose=verbose,saveat=stepsize)
  ret_u = hcat(solv.u...)
  ret_r = iofun.(ret_u,ntw)
  return solv.t,ret_u,ret_r
end

function run_network_noise_simple(ntw::RecurrentNeuralNetwork{R},
    r_start::Vector{R},noiselevel::Real,t_end::Real;
    verbose::Bool=false,stepsize=0.05) where R<:Real
  u0=ioinv.(r_start,ntw)
  f(du,u,p,t) = du_nonoise!(du,u,ntw)
  σ_f(du,u,p,t) = fill!(du,noiselevel)
  prob = SDEProblem(f,σ_f,u0,(0.,t_end))
  solv =  solve(prob,EM();verbose=verbose,dt=stepsize)
  ret_u = hcat(solv.u...)
  ret_r = iofun.(ret_u,ntw)
  return solv.t,ret_u,ret_r
end

function run_network_noise(ntw::RecurrentNeuralNetwork{R},r_start::Vector{R},t_end::Real;
    stepsize=0.05) where R<:Real
  u0=ioinv.(r_start,ntw)
  chol = let c=cholesky(ntw.sigma_noise)
    Matrix(c.L)
  end
  f(du,u,p,t) = du_nonoise!(du,u,ntw)
  g(du,u,p,t) = (@. du = chol)
  prob = SDEProblem(f,g,u0,(0.,t_end);noise_rate_prototype=similar(chol))
  solv = solve(prob,EM();dt=stepsize)
  ret_u = hcat(solv.u...)
  ret_r = iofun.(ret_u,ntw)
  return solv.t,ret_u,ret_r
end


"""
  run_network_to_convergence(u0, rn::RecurrentNeuralNetwork ;
                t_end=80. , veltol=1E-4)
Runs the network as described in [`run_network_nonoise`](@ref), but stops as soon as
`norm(v) / n < veltol` where `v` is the velocity at time `t`.
If this condition is not satisfied (no convergence to attractor), it runs until `t_end` and prints a warning.
# Arguments
- `rn::RecurrentNeuralNetwork`
- `r_start::Vector` : initial conditions
- `t_end::Real`: the maximum time considered
- `veltol::Real` : the norm (divided by num. dimensions) for velocity at convergence
# Outputs
- `u_end`::Vector : the final state at convergence
- `r_end`::Vector : the final state at convergence as rate
"""
function run_network_to_convergence(ntw::RecurrentNeuralNetwork{R},
     r_start::Vector{R};t_end::Real=50. , veltol::Real=1E-1) where R
    function  condition(u,t,integrator)
        v = get_du(integrator)
        return norm(v) / length(v) < veltol
    end
    function affect!(integrator)
        savevalues!(integrator)
        return terminate!(integrator)
    end
    u0=ioinv.(r_start,ntw)
    cb=DiscreteCallback(condition,affect!)
    ode_solver = Tsit5()
    gu_alloc = similar(u0)
    f(du,u,p,t) = du_nonoise!(du,u,ntw)
    prob = ODEProblem(f,u0,(0.,t_end))
    out = solve(prob,Tsit5();verbose=false,callback=cb)
    u_out = out.u[end]
    t_out = out.t[end]
    if isapprox(t_out,t_end; atol=0.05)
        vel = du_nonoise(u_out,ntw)
        @warn "no convergence after max time $t_end"
        @info "the norm (divided by n) of the velocity is $(norm(vel)/length(vel)) "
    end
    return u_out,ntw.iofun.(u_out)
end

################################################
# Here is the algorithm adapted from Hennequin & Lengyel 2016
##################################################

function nu_fun1(mu::Vector{<:Real},sigma_diag::Vector{<:Real},α::Real)
  nr=Normal()
  return  @. α*(
    mu*cdf(nr,mu/sqrt(sigma_diag)) + 
    sqrt(sigma_diag)*pdf(nr,mu/sqrt(sigma_diag)))
end

function nu_fun2(mu::Vector{<:Real},sigma_diag::Vector{<:Real},α::Real)
  #nu1=nu_fun1(mu,sigma_diag,α)
  #return @. mu * nu1 + α*sigma_diag*cdf(Normal(),mu/sqrt(sigma_diag))
  nr=Normal()
  return  @. α*(
    mu^2*cdf(nr,mu/sqrt(sigma_diag)) + 
    mu*sqrt(sigma_diag)*pdf(nr,mu/sqrt(sigma_diag)) +
    α*sigma_diag*cdf(nr,mu/sqrt(sigma_diag))
    )
end

function nu_fun(mu::Vector{<:Real},sigma_diag::Vector{<:Real},io::ReLU)
  return nu_fun1(mu,sigma_diag,io.α)
end
function nu_fun(mu::Vector{<:Real},sigma_diag::Vector{<:Real},io::ReQuad)
  return nu_fun2(mu,sigma_diag,io.α)
end


function gamma_fun1(mu::Vector{<:Real},sigma_diag::Vector{<:Real},α::Real)
  return @.  α*cdf(Normal(),mu/sqrt(sigma_diag))
end
function gamma_fun2(mu::Vector{<:Real},sigma_diag::Vector{<:Real},α::Real)
  nr=Normal()
  return @. 2*α*(
    mu*cdf(nr,mu/sqrt(sigma_diag)) + 
    sqrt(sigma_diag)*pdf(nr,mu/sqrt(sigma_diag)))
end
function gamma_fun(mu::Vector{<:Real},sigma_diag::Vector{<:Real},io::ReLU)
  return gamma_fun1(mu,sigma_diag,io.α)
end
function gamma_fun(mu::Vector{<:Real},sigma_diag::Vector{<:Real},io::ReQuad)
  return gamma_fun2(mu,sigma_diag,io.α)
end

function j_thingy_old(mu::Vector{R},sigma_diag::Vector{R},
    ntw::RecurrentNeuralNetwork{R}) where R
  γ = gamma_fun(mu,sigma_diag,ntw.iofun)
  ret = copy(ntw.weight_matrix)
  broadcast!(*,ret,ret,transpose(γ))
  ret -= I
  return broadcast!(/,ret,ret,ntw.time_membrane)
end
function j_thingy(mu::Vector{R},sigma_diag::Vector{R},
    ntw::RecurrentNeuralNetwork{R}) where R
  T = Diagonal(ntw.time_membrane)
  γ = Diagonal(gamma_fun(mu,sigma_diag,ntw.iofun))
    return T\(ntw.weight_matrix*γ - I)
end


function dSigma!(dΣ::M,Σ::M,μ::V,   
    ntw::RecurrentNeuralNetwork{R}) where {R<:Real,M<:Matrix{R},V<:Vector{R}}
  J = j_thingy(μ,diag(Σ),ntw)
  dΣ .= ntw.sigma_noise .+ (J*Σ) .+ (Σ*J') 
  return dΣ
end

function dmu!(ret::V,mu::V,sigma_diag::V,
  ntw::RecurrentNeuralNetwork{R}) where {R<:Real,V<:Vector{R}}
  ν = nu_fun(mu,sigma_diag,ntw.iofun)
  ret .= ntw.input .- mu
  BLAS.gemv!('N',1.,ntw.weight_matrix,ν,1.,ret)
  return scalebytime!(ret,ntw)
end

# Up to here everything works. Integrating outise and using starting conditions.
# Let 's try to integrate internally.

function dmu_step!(ret::V, mu::V,Σ::Matrix{R},δ::Float64,
    ntw::RecurrentNeuralNetwork{R}) where {R<:Real,V<:Vector{R}}
  dmu!(ret,mu,diag(Σ),ntw)
  ret .= ret.*δ .+ mu
  return ret
end

function dSigma_step!(ret::M,Σ::M,μ::V,δ::Float64,ntw::RecurrentNeuralNetwork{R}) where {R<:Real,M<:Matrix{R},V<:Vector{R}}
  dSigma!(ret,Σ,μ,ntw)
  ret .= ret.*δ + Σ
  return ret
end

function dmuSigma_step!(dest_mu::V,dest_sigma::M,μ::V,Σ::M,δ::Real,
  ntw::RecurrentNeuralNetwork{R}) where {R<:Real,V<:Vector{R},M<:Matrix{R}}
  dSigma_step!(dest_sigma,Σ,μ,δ,ntw)
  dmu_step!(dest_mu,μ,Σ,δ,ntw)
  return dest_sigma,dest_mu
end

# Optimize this a bit, prelocate
function dSigma_step_special!(Σ::M, μ::V, δ::Float64, ntw::RecurrentNeuralNetwork{R}) where {R<:Real, M<:Matrix{R}, V<:Vector{R}}
  J = j_thingy(μ, diag(Σ), ntw)
  A = (I + δ * J)
  Σ .= A * Σ * A' + δ * ntw.sigma_noise  # Modify Σ in place
  return Σ
end

function dmuSigma_step_special!(dest_mu::V,μ::V,Σ::M,δ::Real,
  ntw::RecurrentNeuralNetwork{R}) where {R<:Real,V<:Vector{R},M<:Matrix{R}}
  dSigma_step_special!(Σ,μ,δ,ntw)
  dmu_step!(dest_mu,μ,Σ,δ,ntw)
  return Σ,dest_mu
end

#=
function dmuSigma_step!(dest_mu::V,dest_sigma::M,μ::V,Σ::M,δ::Real,
    ntw::RecurrentNeuralNetwork{R}) where {R<:Real,V<:Vector{R},M<:Matrix{R}}
    dSigma!(ret,mu,diag(Σ),ntw)
    ret .= ret.*δ .+ mu
    return ret
end

function dSigma_step_uniform_noise!(ret::M,Σ::M,μ::V,δ::Float64,
    ntw::RecurrentNeuralNetwork{R}) where {R<:Real,V<:Vector{R},M<:Matrix{R}}
  Σdiag=diag(Σ)
  J=j_thingy(μ,Σdiag,ntw)  
  # copy!(ret,ntw.sigma_noise)
  ret = (δ.*J + I)*ret*(J.*δ + I)' + ntw.sigma_noise.*δ
  # return  !('N','T',1.0,J*Σ,J,δ,ret)
  return  ret
end

function dmuSigma_step!(dest_mu::V,dest_sigma::M,μ::V,Σ::M,δ::Real,
    ntw::RecurrentNeuralNetwork{R}) where {R<:Real,V<:Vector{R},M<:Matrix{R}}
  dSigma_step_uniform_noise!(dest_sigma,Σ,μ,δ,ntw)
  dmu_step!(dest_mu,μ,dest_sigma,δ,ntw)
  return dest_sigma,dest_mu
end
=#

## initial conditions

function mustart_sigmastart(ntw::RecurrentNeuralNetwork{R}) where R
  mustart = copy(ntw.input)
  if hasnoise(ntw)
    Tm = diagm(-inv.(ntw.time_membrane ))
    sigma_start = lyap(Tm,Matrix(ntw.sigma_noise))
  else
    ntot = ndims(ntw)
    sigma_start = zeros(ntot,ntot)
  end
  return mustart,sigma_start
end

# noise in a linear system
# if S is noise covariance for dx = A x dt + noise 
# then the covariance of x, called P , satisfies 0 = A P + P A' + S
function cov_linear(A::Matrix{Float64},cov_noise::AbstractMatrix{Float64})
  return lyap(A,cov_noise)
end

# dx/dt = A x + h + noise
# then  E[x(t)] = - A^-1 * h
function stable_point_linear(A::Matrix{Float64},h::Vector{Float64})
  return -A\h
end

function mean_cov_linear_ntw(ntw::RecurrentNeuralNetwork{R}) where R
  @assert typeof(ntw.iofun) <: IOIdentity
  T = Diagonal(ntw.time_membrane)
  A = T\(ntw.weight_matrix-I)
  h = T\ntw.input
  μ = stable_point_linear(A,h)
  Σ = cov_linear(A,Matrix(ntw.sigma_noise))
  return (μ,Σ)
end

function run_integration!(ntw::RecurrentNeuralNetwork{R}, N::Int, atol::Float64,dt::Float64) where R
  mu_start, sigma_start = mustart_sigmastart(ntw)
  # Initializing
  tol = [ones(size(mu_start)), ones(size(sigma_start))]
  μ = mu_start
  Σ = sigma_start
  last_mu = copy(μ)
  last_sigma = copy(Σ)
  ii = 1
  while ii < N
    #dmuSigma_step!(μ, Σ, last_mu, last_sigma, dt, ntw)
    dmuSigma_step_special!(μ,last_mu,Σ,dt,ntw)
    if norm(tol, 2) < atol
      @info "Semi-analytical integrator has converged at iteration $ii"
      break
    end
    if !isposdef(round.(Σ, digits=4))  # Round Σ to 4 decimals and check if it's not positive definite
      @warn "Σ is not positive definite at iteration $ii"
      break
    end
    # Euler integrator does not preserve positive definiteness
    tol[1] = μ - last_mu
    tol[2] = Σ - last_sigma
    ii += 1
    last_mu = copy(μ)
    last_sigma = copy(Σ)
  end
  return μ,Σ
end

## Rates covariance calculation
# Attention to the signs of the coefficients mu/sqrt(Sigma)
# A coefficient

function a_fun02(α::Real,μ::Vector{<:Real},Σ::Matrix{<:Real})
  νn = nu_fun2(μ,diag(Σ),α)
  return [α*νn[j] for i in 1:size(Σ, 1), j in 1:size(Σ, 1)]
end

function a_fun01(α::Real,μ::Vector{<:Real},Σ::Matrix{<:Real})  #Used when the network is nonlinear
  ν1 = nu_fun1(μ,diag(Σ),α)
  return  [α*ν1[j] for i in 1:size(Σ, 1), j in 1:size(Σ, 1)]
end

function a_fun11(α::Real,μ::Vector{<:Real},Σ::Matrix{<:Real})
  νn = nu_fun1(μ,diag(Σ),α)
  γn = gamma_fun1(μ,diag(Σ),α)
  #=
  gamman = [Σ[i, j] * γn[j] for i in 1:size(Σ, 1), j in 1:size(Σ, 1)]
  gamma_j_j_elements = diag(gamman)
  gamma_j_j = vcat(fill(gamma_j_j_elements', size(gamman,1))...)
  mat_sqrt_Σ = [sqrt(Σ[i, i] / Σ[j, j]) for i in 1:size(Σ, 1), j in 1:size(Σ, 1)]
  =#
  return [α*μ[i]*νn[j] + α*sqrt(Σ[i, i] / Σ[j, j])*Σ[j, j] * γn[j] for i in 1:size(Σ, 1), j in 1:size(Σ, 1)]
end

function a_fun12(α::Real,μ::Vector{<:Real},Σ::Matrix{<:Real})
  νn = nu_fun2(μ,diag(Σ),α)
  γn = gamma_fun2(μ,diag(Σ),α)
  #=
  gamman = [Σ[i, j] * γn[j] for i in 1:size(Σ, 1), j in 1:size(Σ, 1)]
  gamma_j_j_elements = diag(gamman)
  gamma_j_j = vcat(fill(gamma_j_j_elements', size(gamman,1))...)
  mat_sqrt_Σ = [sqrt(Σ[i, i] / Σ[j, j]) for i in 1:size(Σ, 1), j in 1:size(Σ, 1)]
  =#
  return  [α*μ[i]*νn[j] + α*sqrt(Σ[i, i] / Σ[j, j])*Σ[j, j] * γn[j] for i in 1:size(Σ, 1), j in 1:size(Σ, 1)]
end

function a_fun22(α::Real,μ::Vector{<:Real},Σ::Matrix{<:Real})
  #=
  Σ_i_i_values = diag(Σ)
  Σ_i_i_matrix = vcat(fill(Σ_i_i_values', size(Σ,1))...)' 
  mat_sqrt_2_Σ = [sqrt(Σ[i, i] * Σ[j, j]) for i in 1:size(Σ, 1), j in 1:size(Σ, 1)]
  =#
  return [μ[i]*a_fun12(α,μ,Σ)[i,j] + Σ[i,i] * a_fun02(α,μ,Σ)[i,j] + 2 *sqrt(Σ[i, i] * Σ[j, j])*a_fun11(α,μ,Σ)[i,j] for i in 1:size(Σ, 1), j in 1:size(Σ, 1)]
end

function a_fun(μ::Vector{<:Real},Σ::Matrix{<:Real},io::ReLU)
  return a_fun11(io.α,μ,Σ)
end

function a_fun(μ::Vector{<:Real},Σ::Matrix{<:Real},io::ReQuad)
  return a_fun22(io.α,μ,Σ)
end

# B coefficient

function b_fun00(α::Real,μ::Vector{<:Real},Σ::Matrix{<:Real})
  nr=Normal()
  return  α^2 .* [cdf(nr,μ[i] / sqrt(Σ[i, i])) + cdf(nr,μ[j] / sqrt(Σ[j, j])) - 1 for i in 1:size(Σ,1), j in 1:size(Σ,1)]
end

function b_fun01(α::Real,μ::Vector{<:Real},Σ::Matrix{<:Real})  
  nr=Normal()
  #=
  Σ_j_j_elements = diag(Σ) # gamma is square
  sqrt_Σ_j_j = vcat(fill(sqrt.(Σ_j_j_elements'), size(Σ,1))...)      
  =#
  return [μ[j]*b_fun00(α,μ,Σ)[i,j] + α^2 * sqrt(Σ[j,j]) *(pdf(nr,μ[j] / sqrt(Σ[j, j])) - pdf(nr,μ[i] / sqrt(Σ[i, i]))) for i in 1:size(Σ,1), j in 1:size(Σ,1)]
end

function b_fun02(α::Real,μ::Vector{<:Real},Σ::Matrix{<:Real})  #Used when the network is nonlinear
  nr=Normal()
  #=
  Σ_j_j_elements = diag(Σ) # gamma is square
  sqrt_Σ_j_j = vcat(fill(sqrt.(Σ_j_j_elements'), size(Σ,1))...)  
  =#
  R_mat = [μ[j] + μ[i] * sqrt(Σ[j, j] / Σ[i, i]) for i in 1:size(Σ, 1), j in 1:size(Σ, 1)]
  return [μ[j]*b_fun01(α,μ,Σ)[i,j] - sqrt(Σ[j,j]) * (α^2 * pdf(nr,μ[i] / sqrt(Σ[i, i])) * R_mat[i,j] - sqrt(Σ[j, j]) * b_fun00(α,μ,Σ)[i,j])   for i in 1:size(Σ,1), j in 1:size(Σ,1)]
end

function b_fun10(α::Real,μ::Vector{<:Real},Σ::Matrix{<:Real})  #Used when the network is nonlinear
  nr=Normal()
  #=
  Σ_i_i_values = diag(Σ)
  Σ_i_i_matrix = (hcat([sqrt.(Σ_i_i_values) for _ in 1:size(Σ,1)]...))
  =#
  return [μ[i]*b_fun00(α,μ,Σ)[i,j] + sqrt(Σ[i,i]) * (α^2 * (pdf(nr,μ[i] / sqrt(Σ[i, i])) - pdf(nr,μ[j] / sqrt(Σ[j, j])))) for i in 1:size(Σ,1), j in 1:size(Σ,1)]
end

function b_fun11(α::Real,μ::Vector{<:Real},Σ::Matrix{<:Real})  #Used when the network is nonlinear
  nr=Normal()
  #=
  Σ_i_i_values = diag(Σ)
  Σ_i_i_matrix = (hcat([sqrt.(Σ_i_i_values) for _ in 1:size(Σ,1)]...))
  =#
  R_mat = [μ[j] + μ[i] * sqrt(Σ[j, j] / Σ[i, i]) for i in 1:size(Σ, 1), j in 1:size(Σ, 1)]
  return  [μ[i]*b_fun01(α,μ,Σ)[i,j] + sqrt(Σ[i,i]) * (α^2 * pdf(nr,μ[i] / sqrt(Σ[i, i])) * R_mat[i,j] - sqrt(Σ[j,j]*b_fun00(α,μ,Σ)[i,j])) for i in 1:size(Σ,1), j in 1:size(Σ,1)]
end

function b_fun12(α::Real,μ::Vector{<:Real},Σ::Matrix{<:Real})  #Used when the network is nonlinear
  nr=Normal()
  #=
  Σ_i_i_values = diag(Σ)
  Σ_i_i_matrix = (hcat([sqrt.(Σ_i_i_values) for _ in 1:size(Σ,1)]...))
  =#
  R_mat = [μ[j] + μ[i] * sqrt(Σ[j, j] / Σ[i, i]) for i in 1:size(Σ, 1), j in 1:size(Σ, 1)]
  return  [μ[i]*b_fun02(α,μ,Σ)[i,j] + sqrt(Σ[i,i]) * (α^2 .* pdf(nr,μ[i] / sqrt(Σ[i, i])) * (R_mat[i,j])^2 - 2*sqrt(Σ[j,j]*b_fun01(α,μ,Σ)[i,j])) for i in 1:size(Σ,1), j in 1:size(Σ,1)]
end

function b_fun(μ::Vector{<:Real},Σ::Matrix{<:Real},io::ReLU)
  #=
  Σ_i_i_values = diag(Σ)
  Σ_i_i_matrix = (hcat([sqrt.(Σ_i_i_values) for _ in 1:size(Σ,1)]...))
  =#
  return [μ[i]*b_fun01(io.α,μ,Σ)[i,j] - sqrt(Σ[i,i]*Σ[j,j]) * b_fun00(io.α,μ,Σ)[i,j] for i in 1:size(Σ,1), j in 1:size(Σ,1)]
end

function b_fun(μ::Vector{<:Real},Σ::Matrix{<:Real},io::ReQuad)
  #=
  Σ_i_i_values = diag(Σ)
  Σ_i_i_matrix = (hcat([sqrt.(Σ_i_i_values) for _ in 1:size(Σ,1)]...))
  =#
  return [μ[i]*b_fun12(io.α,μ,Σ)[i,j] + Σ[i,i]*b_fun02(io.α,μ,Σ)[i,j] - 2 * sqrt(Σ[i,i]*Σ[j,j]) * b_fun11(io.α,μ,Σ)[i,j] for i in 1:size(Σ,1), j in 1:size(Σ,1)]
end

# Build the rate covariance

function coef_ansatz(μ::Vector{<:Real},Σ::Matrix{<:Real},ntw::RecurrentNeuralNetwork{<:Real})
  #=
  Σ_i_i_values = diag(Σ)
  Σ_i_i_matrix = (hcat([sqrt.(Σ_i_i_values) for _ in 1:size(Σ,1)]...))
  =#
  B = b_fun(μ,Σ,ntw.iofun)
  A = a_fun(μ,Σ,ntw.iofun)
  γ = gamma_fun(μ,diag(Σ),ntw.iofun)
  ν = nu_fun(μ,diag(Σ),ntw.iofun)
  Λ_pos = [- ν[i]* ν[j] + A[i,j] for i in 1:size(Σ,1), j in 1:size(Σ,1)]
  Λ_neg = [- ν[i]* ν[j] + B[i,j] for i in 1:size(Σ,1), j in 1:size(Σ,1)]
  #=
  Λ_pos = - ν .* ν' + A
  Λ_neg = - ν .* ν' + B
  α_mat_1 = (γ.*Σ_i_i_matrix).*(γ'.*Σ_i_i_matrix')
    =#
  α_mat_1 = [(γ[i]* sqrt(Σ[i,i])) * (γ[j]* sqrt(Σ[j,j])) for i in 1:size(Σ,1), j in 1:size(Σ,1)]
  α_mat_2 = (Λ_pos + Λ_neg)./2
  α_mat_3 = (Λ_pos - Λ_neg)./2 - α_mat_1
  return α_mat_1,α_mat_2,α_mat_3
end

function rate_variance(μ::Vector{<:Real},Σ::Matrix{<:Real},ntw::RecurrentNeuralNetwork{<:Real})
  α_mat_1,α_mat_2,α_mat_3 = coef_ansatz(μ,Σ,ntw)
  C = [Σ[i, j] / sqrt(Σ[i, i] * Σ[j, j]) for i in 1:size(Σ,1), j in 1:size(Σ,1)]
  return α_mat_1.*C + α_mat_2.*(C.^2) + α_mat_3.*(C.^3)
end

function run_full_integration!(ntw::RecurrentNeuralNetwork{R}, N::Int, atol::Float64,dt::Float64) where R
  mu_start, sigma_start = mustart_sigmastart(ntw)
  # Initializing
  tol = [ones(size(mu_start)), ones(size(sigma_start))]
  μ = mu_start
  Σ = sigma_start
  last_mu = copy(μ)
  last_sigma = copy(Σ)
  ii = 1
  while ii < N
    #dmuSigma_step!(μ, Σ, last_mu, last_sigma, dt, ntw)
    dmuSigma_step_special!(μ,last_mu,Σ,dt,ntw)
    if norm(tol, 2) < atol
      @info "Semi-analytical integrator has converged at iteration $ii"
      break
    end
    if !isposdef(round.(Σ, digits=4))  # Round Σ to 4 decimals and check if it's not positive definite
      @warn "Σ is not positive definite at iteration $ii"
      break
    end
    # Euler integrator does not preserve positive definiteness
    tol[1] = μ - last_mu
    tol[2] = Σ - last_sigma
    ii += 1
    last_mu = copy(μ)
    last_sigma = copy(Σ)
  end
  Λ = rate_variance(μ,Σ,ntw) # Compute it once at stationary state
  return μ,Σ,Λ
end


end # module
