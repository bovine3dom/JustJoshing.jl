#!/usr/bin/env julia

# Best to run this with JULIA_NUM_THREADS=$(nproc) and julia 1.3

# Notation: if f is a function of x, fv is one realisation of that function

using Distributions: Normal, cdf

using ForwardDiff: derivative

import Plots
Plots.unicodeplots()

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
function C(S,t; T=1, K=1, r=0.5, σ=0.5)
    d_1v = d_1(S,t,T,K,r,σ)
    d_2v = d_2(S,t,T,K,r,σ)
    N(d_1v)*S - N(d_2v)*K*exp(-r*(T-t))
end

# Price of a put option (i.e. holder may sell at maturity)
function P(S,t; T=1, K=1, r=0.5, σ=0.5)
    PVv = PV(t,T,K,r)
    PVv - S + C(S,t, T=T, K=K, r=r, σ=σ)
end


# Tests
#
# Price of call option monotonically decreases with strike price
@assert all(derivative.(x->C(1,0.5;K=x),rand(100)*10) .< 0)

#
# Call option should be between S and S - Ke(-rT) for all inputs
#
# (if call option is more expensive than stock, it is clearly better
# to just buy the stock; if it is cheaper than the difference between
# the current value of the stock and the discounted strike price, we
# can arbitrage by selling the stock now, buying the option, putting the
# strike price in a risk free bond, and then exercising the option
# (so we end up with the stock back and some profit))
#
let (S,t,K,r,σ) = rand(5), T = t + rand()
    # The following is true
    println(S >= C(S,t;T=T,K=K,r=r,σ=σ))

    # The following is rarely true, so we have a bug
    println(C(S,t;T=T,K=K,r=r,σ=σ) >= S - K*exp(-r*T))
end

# This should look like Figure 5.3, page 121 in Joshi (it does)
p = Plots.plot(); for t in 0:0.2499:1; Plots.plot!(p,x->C(x,t;K=100,σ=0.3,r=0),60:140); end; p

#
# Price of call option should monotonically increase with volatility
