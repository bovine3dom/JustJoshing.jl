#!/bin/env julia

# Playing with running the pricer on GPU
#
# Will port into main library once I'm happy with it

using BenchmarkTools, CuArrays, Plots
CuArrays.allowscalar(false)
Plots.unicodeplots()

using Statistics: mean, std

present_value(v,r,t,T) = v*CuArrays.CUDAnative.exp(-r*(T-t))

function rand_price(seed=randn();payoff=(S,K,t,T,r)->S,S_0=100,t=0,T=1,K=S_0,r=0.02,σ=0.05) # Doesn't work on GPU - doesn't like function as argument
    S = B(seed;T=T-t,B0=S_0,r=r,d=0,σ=σ)
    
    present_value(payoff(S,K,t,T,r),r,t,T)
end

function mc_pricer(payoff;rng=CURAND.randn,S_0=100,t=0,T=1.0,r=0.02,σ=0.05,trials=10_000)
  a = rng(Float32,trials)
  #a = B.(rng(Float32,trials),0.9) # ;B0=S_0,T=T-t,r=0.02,σ=0.05) # CuArrays doesn't like any extra args?

  # TODO: stuff gets converted to f64 here - which IIRC is much slower on GPUs
  Bbake(x) = S_0*CuArrays.CUDAnative.exp(r*T-0.5*σ^2*T+σ*CuArrays.CUDAnative.pow(T-t,0.5)*x) # sqrt(T-t) doesn't work
  prices = present_value.(payoff(Bbake.(a)),r,t,T)
  (mean = mean(prices), sem = std(prices)/sqrt(trials))
end

function mc_pricer_pathdep(payoff;S_0=100,ts::Array{Float32,1}=[0f0],T=1.0f0,r=0.02f0,σ=0.05f0)
  # Almost the same speed as using CPU
  # TODO: more work : )
  trials = 100_000
  a = Iterators.product(randn(Float32,trials),ts) |> collect |> cu;
  #a = B.(rng(Float32,trials),0.9) # ;B0=S_0,T=T-t,r=0.02,σ=0.05) # CuArrays doesn't like any extra args?
    
    
  Bbake(x::Tuple{Float32,Float32})::Float32 = begin
    p = x[1]
    t = x[2]
    S_0*CuArrays.CUDAnative.exp(r*T-0.5f0*σ^2*T+σ*CuArrays.CUDAnative.pow(T-t,0.5f0)*p) # sqrt(T-t) doesn't work
  end
  # If prices is not a cuarray we get a big slowdown
  # Need to ensure that our function operates on array for max
  prices::CuArray{Float32,2} = payoff(Bbake.(a)) # payoff must act on a price x time matrix
  (mean = mean(prices), sem = std(prices)/sqrt(trials))
end

function mc_pricer_pathdep_cpu(payoff;S_0=100,ts::Array{Float32,1}=[0f0],T=1.0f0,r=0.02f0,σ=0.05f0)
  trials = 100_000
  a = Iterators.product(randn(Float32,trials),ts)
  Bbake(x::Tuple{Float32,Float32})::Float32 = begin
    p = x[1]
    t = x[2]
    S_0*exp(r*T-0.5f0*σ^2*T+σ*sqrt(T-t)*p)
  end
  prices = payoff(Bbake.(a)) # payoff must act on a price x time matrix
  (mean = mean(prices), sem = std(prices)/sqrt(trials))
end
