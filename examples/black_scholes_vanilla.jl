#!/usr/bin/env julia

# Best run in the REPL until I work out how to get unicodeplots to print to stdout when in an `include`

using JustJoshing

import Plots
Plots.unicodeplots()

# This should look like Figure 5.3, page 121 in Joshi (it does)
# NB: looks like Joshi's graph is mislabelled - should be spot price
p = Plots.plot(); for t in 0:0.2499:1; Plots.plot!(p,x->C(x,t;K=100,σ=0.3,r=0),60:140); end; p

Plots.plot(x->C(1,x),0:0.001:0.9999) # Value of at-the-money approaches zero as time approaches maturity (Fig 2.1, page 35)

Plots.plot(x->C(0.5,0;σ=x),0:0.01:1) # Call options with volatile underlyings are more expensive (Fig 3.8, page 66)

# Simulated stock price
Plots.plot(x->B(x,σ=0.02),0:0.01:1)


# Monte-Carlo validation of forward contract
import Statistics: mean

# TODO: multithread this
mean([rand_price((S,K,t,T,r)->max(S-K,0),100;t=0.99,T=1,K=100,r=0.02,σ=0.05) for i in 1:100_000_000])
