#!/usr/bin/env julia

# Project 1: Vanilla options in a Black-Scholes World
#
# Page 437+, Joshi's Mathematical Finance


# Best to run this with JULIA_NUM_THREADS=$(nproc) and julia 1.3

# Notation: if f is a function of x, fv is one realisation of that function

using KissThreading
using SpecialFunctions: erf

export C, P, B, present_value, rand_price, mc_pricer, binary, mc_pricer_pathdep

N(x) = 0.5 * (1 + erf(x/sqrt(2)))

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
    N(d_1v)*S - N(d_2v)*present_value(K,r,t,T)
end

# Price of a put option (i.e. holder may sell at maturity)
function P(S,t; T=1, K=100, r=0.02, σ=0.5)
    present_value(K,r,t,T) - S + C(S,t, T=T, K=K, r=r, σ=σ)
end

# Price of a binary option:
# Payoff of 1 if S(T) > E, 0 otherwise
# 
# Source: http://www.iam.fmph.uniba.sk/institute/stehlikova/fd14en/lectures/05_black_scholes_1.pdf
#
# Can construct a less than binary thusly:
# less_than_binary = present_value(1) - greater_than_binary
# (cf. put call parity)
function binary(S,t; T=1, E=100, r=0.02, σ=0.5)
    present_value(1,r,t,T) * N(d_2(S,t,T,E,r,σ))
end

##############################
#                            #
# Validation via Monte Carlo #
#                            #
##############################

# Brownian motion
B(T;B0=100,r=0.02,d=0.00,σ=0.5) = B0*exp((r-d)*T-0.5*σ^2*T+σ*√T*randn())

# Sketch of pricer: Brownian motion generates a final stock price; payoff of option is calculated and then discounted. Price of option is average of these.

present_value(v,r,t,T) = v*exp(-r*(T-t))

function rand_price(payoff,S_0;t=0,T=1,K=S_0,r=0.02,σ=0.05)
    S = B(T-t;B0=S_0,r=r,d=0,σ=σ)
    present_value(payoff(S,K,t,T,r),r,t,T)
end

# Strictly we should use a thread-safe RNG here
function mc_pricer(
    payoff,S_0;t=0,T=1,K=S_0,r=0.02,σ=0.05, trials=100_000_000
)
    linear, squared = tmapreduce(
        x->begin
            p = rand_price(payoff,S_0;t=t,T=T,K=K,r=r,σ=σ)
            (p,p^2) # Second term is to keep track of standard error
        end,
        (a,b)->a.+b,
        Base.OneTo(trials),
        batch_size=1000,
        init=(0.0, 0.0)
    )

    mean = linear/trials

    (mean = mean, sem = sqrt(squared/trials - mean^2)/sqrt(trials))
end

function mc_pricer_pathdep(
    payoff,S_0;t=[0],T=1,K=S_0,r=0.02,σ=0.05, trials=100_000_000
)
    linear, squared = tmapreduce(
        x->begin
            p = B.(T.-t;B0=S_0,r=r,d=0,σ=σ)
            price = present_value(payoff(p,K,t,T,r),r,t[1],T) # Payoff must operate on time series
            (price,price^2) 
        end,
        (a,b)->a.+b,
        Base.OneTo(trials),
        batch_size=1000,
        init=(0.0, 0.0)
    )

    mean = linear/trials

    (mean = mean, sem = sqrt(squared/trials - mean^2)/sqrt(trials))
end
