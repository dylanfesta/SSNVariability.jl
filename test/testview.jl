using Pkg
pkg"activate ."
using SSNVariability ; const S = SSNVariability
using Test
using LinearAlgebra,Calculus,Statistics
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
  @info """
  The max differences between the two are $(extrema(x .-y ))
  """
  plt=plot()
  scatter!(plt,x,y;leg=false,ratio=1,color=:white)
  lm=xlims()
  plot!(plt,identity,range(lm...;length=3);linestyle=:dash,color=:yellow)
  return plt
end

##

ne,ni = 50,20
ntot = ne+ni
ntw = S.RecurrentNeuralNetwork(ne,ni)
noise_test=S.random_covariance_matrix(ntot,3.3)
copy!(ntw.sigma_noise,noise_test)

mu_test,sigma_test = let n=ne+ni,
  mucurr=fill(3.,n)
  sigmacurr = fill(0.0,n,n)
  mu=copy(mucurr)
  sigma=copy(sigmacurr)
  for i in 1:100
    S.dmuSigma_step!(mucurr,sigmacurr,mu,sigma,0.001,ntw)
    copy!(mu,mucurr)
    copy!(sigma,sigmacurr)
  end
  mu,sigma
end
##
ustart = rand(ntot) .* 10.0
rstart = ntw.iofun.(ustart)
_,r_stable = S.run_network_to_convergence(ntw,rstart)
t,_,ei=S.run_network_noise(ntw,r_stable,20.;verbose=true,stepsize=0.01)

_ = let  x=mean(ei;dims=2)[:]
  y=ntw.iofun.(mu_test)
  plotvs(x,y)
end

_ = let  x=cov(ei;dims=2)
  y=sigma_test
  plotvs(x,y)
end



##

ne,ni = 13,10
ntot = ne+ni
ntw = S.RecurrentNeuralNetwork(ne,ni)


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
noise_test = 0.8
for i in 1:ndims(ntw)
  ntw.sigma_noise[i,i]=noise_test
end
t,_,ei=S.run_network_noise_simple(ntw,r_stable,noise_test,1.;verbose=true,stepsize=0.001)
eiplot(t,ei[1,:],ei[14,:])

_ = let plt = eiplot(t,ei[1,:],ei[14,:])
  xl=xlims()
  e_stab = r_stable[1]
  i_stab = r_stable[14]
  plot!(plt,x->e_stab,range(xl...;length=2);color=:red,linestyle=:dash)
  plot!(plt,x->i_stab,range(xl...;length=2);color=:blue,linestyle=:dash)
end

##
t,_,ei=S.run_network_noise(ntw,r_stable,3.;verbose=true,stepsize=0.01)
eiplot(t,ei[1,:],ei[14,:])

_ = let plt = eiplot(t,ei[1,:],ei[14,:])
  xl=xlims()
  e_stab = r_stable[1]
  i_stab = r_stable[14]
  plot!(plt,x->e_stab,range(xl...;length=2);color=:red,linestyle=:dash)
  plot!(plt,x->i_stab,range(xl...;length=2);color=:blue,linestyle=:dash)
end

std(ei[9,:])
