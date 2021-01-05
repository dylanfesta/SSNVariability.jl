module SSNVariability
using LinearAlgebra
using DifferentialEquations

abstract type  IOFunction{R} end

function iofun!(dest::Vector{R},source::Vector{R},
    io::IOFunction{R}) where R
  for i in eachindex(dest)
    @inbounds dest[i] = io(source[i])
  end
  return dest
end

function iofun(source::Vector{R},io::IOFunction{R}) where R
  return iofun!(similar(source),source,io)
end

struct ReLu{R} <: IOFunction{R}
  α::R
end

function (g::ReLu)(x::Real)
  re = x < 0 ? zero(x) : g.α * x
  return re
end
ioinv(x::Real,g::ReLu) = x > 0 ? x : zero(x)

struct ReQuad{R} <:IOFunction{R}
  α::R
end

function (g::ReQuad)(x::Real)
  ret = x < 0 ? zero(x) : g.α * x
  return ret*ret
end
ioinv(x::Real,g::ReQuad) =  x < 0 ? zero(x) : sqrt(x)

struct RecurrentNeuralNetwork{R}
  iofun::IOFunction{R}
  weight_matrix::Matrix{R}
  time_membrane::Vector{R}
end
Base.Broadcast.broadcastable(x::RecurrentNeuralNetwork)=Ref(x)
function Base.ndims(rnn::RecurrentNeuralNetwork)
  return size(rnn.weight_matrix,1)
end
function scalebytime!(v::Vector{R},rnn::RecurrentNeuralNetwork{R}) where R
  v ./= rnn.time_membrane
end
function iofun!(dest::Vector{R},source::Vector{R},
    ntw::RecurrentNeuralNetwork) where R
  return iofun!(dest,source,ntw.iofun)
end
function iofun(x::Real,ntw::RecurrentNeuralNetwork)
  return ntw.iofun(x)
end
function ioinv!(dest::Vector{R},source::Vector{R},
    ntw::RecurrentNeuralNetwork) where R
  for i in eachindex(dest)
    @inbounds dest[i] = ioinv(source[i],ntw.iofun)
  end
  return dest
end
ioinv(source::Vector{<:Real},ntw::RecurrentNeuralNetwork) = ioinv!(similar(source),source,ntw)

function du_nonoise!(u::V,input_current::V,
    ntw::RecurrentNeuralNetwork{R}) where {R<:Real,V<:Vector{R}}
  r = iofun.(u,ntw)
  u .-= input_current
  BLAS.gemv!('N',1.,W,r,-1.,u)
  return scalebytime!(u,ntw)
end

function run_network_noise(ntw::RecurrentNeuralNetwork{R},r_start::Vector{R},
    noiselevel::Real,t_end::Real; verbose::Bool=false,stepsize=0.05) where R<:Real
  ode_solver = Tsit5()
  u0=ioinv(r_start)
  f(du,u,p,t) = du_nonoise!(u,ntw)
  σ_f(du,u,p,t) = fill!(du,noiselevel)
  prob = SDEProblem(f,σ_f,u0,(0.,t_end))
  solv =  solve(prob,ode_solver;verbose=verbose,saveat=stepsize)
  ret_u = hcat(solv.u...)
  ret_r = iofun.(u,ntw)
  return solv.t,ret_u,ret_r
end

end # module
