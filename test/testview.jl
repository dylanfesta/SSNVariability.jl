using Pkg
pkg"activate ."
using SSNVariability ; const S = SSNVariability
using Test
using LinearAlgebra,Calculus
using Plots,NamedColors ; theme(:dark)

function eiplot(t::V,e::V,i::V) where V<:Vector{<:Real}
  plt=plot()
  plot!(plt,t,hcat(e,i) ; color=[:red :blue] , linewidth=3,leg=false)
end

function eiplot(t::V,e::M,i::M) where {V  <:Vector,M<:Matrix{<:Real}}
  mp(mat)=mapslices(norm,mat;dims=1)[:]
  return eiplot(t,mp(e),mp(i))
end

function plotvs(x::AbstractArray{<:Real},y::AbstractArray{<:Real})
  x,y=x[:],y[:]
  plt=plot()
  scatter!(plt,x,y;leg=false,ratio=1,color=:white)
  lm=xlims()
  plot!(plt,identity,range(lm...;length=3);linestyle=:dash,color=:yellow)
  return plt
end

##

ne,ni = 13,10
ntot = ne+ni
ntw = S.RecurrentNeuralNetwork(ne,ni)

_,r_stable = S.run_network_to_convergence(ntw,rstart)

ustart = rand(ntot) .* 10.0
rstart = ntw.iofun.(ustart)

t,eiu,ei=S.run_network_nonoise(ntw,rstart,1.0;verbose=true,stepsize=0.001 )
_ = let plt = eiplot(t,ei[1,:],ei[14,:])
  xl=xlims()
  e_stab = r_stable[1]
  i_stab = r_stable[14]
  plot!(plt,x->e_stab,range(xl...;length=2);color=:red,linestyle=:dash)
  plot!(plt,x->i_stab,range(xl...;length=2);color=:blue,linestyle=:dash)
end

velnorm,vels = let dat=eiu, n=ndims(ntw)
  vels = mapslices(dat;dims=1) do u
    S.du_nonoise(u,ntw)
  end
  velnorm=mapslices(v->norm(v)/n,vels;dims=1)[:]
  velnorm, vels
end

plot(t,velnorm;leg=false,yscale=:log10)

heatmap(vels)


##
t,_,ei=S.run_network_noise(ntw,rstart,5.0,2.;verbose=true,stepsize=0.001)
eiplot(t,ei[1,:],ei[14,:])

_ = let plt = eiplot(t,ei[1,:],ei[14,:])
  xl=xlims()
  e_stab = r_stable[1]
  i_stab = r_stable[14]
  plot!(plt,x->e_stab,range(xl...;length=2);color=:red,linestyle=:dash)
  plot!(plt,x->i_stab,range(xl...;length=2);color=:blue,linestyle=:dash)
end
