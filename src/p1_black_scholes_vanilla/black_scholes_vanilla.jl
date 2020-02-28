#!/usr/bin/env julia

# Project 1: Vanilla options in a Black-Scholes World
#
# Page 437+, Joshi's Mathematical Finance


# Best to run this with JULIA_NUM_THREADS=$(nproc) and julia 1.3

# Notation: if f is a function of x, fv is one realisation of that function

using Distributions: Normal, cdf
using KissThreading

export C, P, B, present_value, rand_price, mc_pricer

N(x) = cdf(Normal(),x)

d_1(S,t,T,K,r,σ) = (log(S/K) + (r + (σ^2)/2)*(T-t)) / (σ*√(T-t))
d_2(S,t,T,K,r,σ) = (log(S/K) + (r - (σ^2)/2)*(T-t)) / (σ*√(T-t))


# Price of a call option (i.e. the holder has the right to buy stock at maturity)
# S = current stock (underlying) price
# t = current time (time since option created)
# T = time of maturity
# K = the price at which the stock may be bought at maturity
# r = the continuously compounded risk-free rate
# σ = volatility of the underlying (standard deviation per root unit of time)
# TODO: d = dividend rate
function C(S,t; T=1, K=100, r=0.02, σ=0.5)
    d_1v = d_1(S,t,T,K,r,σ)
    d_2v = d_2(S,t,T,K,r,σ)
    N(d_1v)*S - N(d_2v)*K*exp(-r*(T-t))
end

# Price of a put option (i.e. holder may sell at maturity)
function P(S,t; T=1, K=100, r=0.02, σ=0.5)
    K*exp(-r*(T-t)) - S + C(S,t, T=T, K=K, r=r, σ=σ)
end


##############################
#                            #
# Validation via Monte Carlo #
#                            #
##############################

# Brownian motion
B(T;B0=100,r=0.02,d=0.00,σ=0.5) = B0*exp((r-d)*T-0.5*σ^2*T+σ*√T*rand(Normal()))

# Sketch of pricer: Brownian motion generates a final stock price; payoff of option is calculated and then discounted. Price of option is average of these.

present_value(v,r,t,T) = v*exp(-r*(T-t))

function rand_price(payoff,S_0;t=0,T=1,K=S_0,r=0.02,σ=0.05)
    S = B(T-t;B0=S_0,r=r,d=0,σ=σ)
    present_value(payoff(S,K,t,T,r),r,t,T)
end

# Strictly we should use a thread-safe RNG here
const mc_pricer(
    payoff,S_0;t=0,T=1,K=S_0,r=0.02,σ=0.05, trials=100_000_000
) = tmapreduce(
    x->rand_price(payoff,S_0;t=t,T=T,K=K,r=r,σ=σ),
    +,
    1:trials,
    init=0
) / trials
