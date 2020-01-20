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
function C(S,t; T=1, K=100, r=0.02, σ=0.5)
    d_1v = d_1(S,t,T,K,r,σ)
    d_2v = d_2(S,t,T,K,r,σ)
    N(d_1v)*S - N(d_2v)*K*exp(-r*(T-t))
end

# Price of a put option (i.e. holder may sell at maturity)
function P(S,t; T=1, K=100, r=0.02, σ=0.5)
    K*exp(-r*(T-t)) - S + C(S,t, T=T, K=K, r=r, σ=σ)
end


# Tests

# Price of call option monotonically decreases with strike price
@assert all(derivative.(x->C(1,0.5;K=x),rand(100_000)*10) .<= 0)

#
# Call option should be between S and S - Ke(-r(T-t)) for all inputs
#
# (if call option is more expensive than stock, it is clearly better
# to just buy the stock; if it is cheaper than the difference between
# the current value of the stock and the discounted strike price, we
# can arbitrage by selling the stock now, buying the option, putting the
# strike price in a risk free bond, and then exercising the option
# (so we end up with the stock back and some profit))
#
# NB: Joshi refers to time-to-maturity, i.e. t_joshi = T-t :)
@assert [
    let (S,t,K,r,σ) = rand(5), T = t + rand()
        S >= C(S,t;T=T,K=K,r=r,σ=σ) >= (S - K*exp(-r*(T-t)))
    end
for _ in 1:100] |> all

# This should look like Figure 5.3, page 121 in Joshi (it does)
# NB: looks like Joshi's graph is mislabelled - should be spot price
p = Plots.plot(); for t in 0:0.2499:1; Plots.plot!(p,x->C(x,t;K=100,σ=0.3,r=0),60:140); end; p

Plots.plot(x->C(1,x),0:0.001:0.9999) # Value of at-the-money approaches zero as time approaches maturity (Fig 2.1, page 35)

Plots.plot(x->C(0.5,0;σ=x),0:0.01:1) # Call options with volatile underlyings are more expensive (Fig 3.8, page 66)


# Price of call option should monotonically increase with volatility
@assert all(derivative.(x->C(1,0.5;σ=x),rand(1000)) .>= 0)


# Validation via Monte Carlo
# Brownian motion
B(T;B0=100,r=0.02,d=-0.06,σ=0.5) = B0*exp((r-d)*T-0.5*σ^2*T+σ*√T*rand(Normal()))

# Simulated stock price
Plots.plot(x->B(x,σ=0.02),0:0.01:1)
