

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

function nu_fun(mu::Vector{<:Real},sigma_diag::Vector{<:Real},io::ReLu)
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
function gamma_fun(mu::Vector{<:Real},sigma_diag::Vector{<:Real},io::ReLu)
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


function dmu!(ret::V,mu::V,sigma_diag::V,
    ntw::RecurrentNeuralNetwork{R}) where {R<:Real,V<:Vector{R}}
  ν = nu_fun(mu,sigma_diag,ntw.iofun)
  ret .= ntw.base_input .- mu
  BLAS.gemv!('N',1.,ntw.weight_matrix,ν,1.,ret)
  return scalebytime!(ret,ntw)
end

function dSigma!(dΣ::M,Σ::M,μ::V,
    ntw::RecurrentNeuralNetwork{R}) where {R<:Real,M<:Matrix{R},V<:Vector{R}}
  J=j_thingy(μ,diag(Σ),ntw)
  dΣ .= sqrt.(ntw.sigma_noise) .+ (J*Σ) .+ (Σ*J') 
  return dΣ
end

function dmu_step!(ret::V, mu::V,Σ::Matrix{R},δ::Real,
    ntw::RecurrentNeuralNetwork{R}) where {R<:Real,V<:Vector{R}}
  dmu!(ret,mu,diag(Σ),ntw)
  ret .= ret.*δ .+ mu
  return ret
end

function dSigma_step_uniform_noise!(ret::M,Σ::M,μ::V,δ::Real,
    ntw::RecurrentNeuralNetwork{R}) where {R<:Real,V<:Vector{R},M<:Matrix{R}}
  Σdiag=diag(Σ)
  J=j_thingy(μ,Σdiag,ntw)
  lmul!(δ,J)
  J+=I
  copy!(ret,ntw.sigma_noise)
  return BLAS.gemm!('N','T',1.0,J*Σ,J,δ,ret)
end
function dmuSigma_step!(dest_mu::V,dest_sigma::M,μ::V,Σ::M,δ::Real,
    ntw::RecurrentNeuralNetwork{R}) where {R<:Real,V<:Vector{R},M<:Matrix{R}}
  dSigma_step!(dest_sigma,Σ,μ,δ,ntw)
  dmu_step!(dest_mu,μ,dest_sigma,δ,ntw)
  return nothing
end

## initial conditions

function mustart_sigmastart(ntw::RecurrentNeuralNetwork{R}) where R
  mustart = copy(ntw.base_input)
  if hasnoise(ntw)
    Tm = diagm(-inv.(ntw.time_membrane ))
    sigma_start = lyap(Tm,Matrix(ntw.sigma_noise))
  else
    ntot = ndims(ntw)
    sigma_start = zeros(ntot,ntot)
  end
  return mustart,sigma_start
end
