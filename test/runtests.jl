using JustJoshing
using Test
using ForwardDiff: derivative

const EPS = 0.0001

@testset "Call option consistency" begin
    # I: see code

    # II: Price of call option monotonically decreases with strike price
    @test all(derivative.(x->C(1,0.5;K=x),rand(100_000)*10) .<= EPS)

    # III: Call option should be between S and S - Ke(-r(T-t)) for all inputs
    #
    # (if call option is more expensive than stock, it is clearly better
    # to just buy the stock; if it is cheaper than the difference between
    # the current value of the stock and the discounted strike price, we
    # can arbitrage by selling the stock now, buying the option, putting the
    # strike price in a risk free bond, and then exercising the option
    # (so we end up with the stock back and some profit))
    #
    # NB: Joshi refers to time-to-maturity, i.e. t_joshi = T-t :)
    @test [
        let (S,t,K,r,σ) = rand(5), T = t + rand()
            S + EPS >= C(S,t;T=T,K=K,r=r,σ=σ) >= (S - K*exp(-r*(T-t))) - EPS
        end
    for _ in 1:100] |> all

    # IV: Price of call option should monotonically increase with volatility
    @test all(derivative.(x->C(1,0.5;σ=x),rand(1000)) .>= -EPS)

    # V: if d=0 (always true in our model) then price should increase as we get closer to expiry
    @test all(derivative.(T->C(1,0;T=T),rand(1000)) .>= -EPS)

    # VI: Call option should be a convex function of strike - i.e. gradient is an increasing function of strike
    #   TODO: work out if there's a more ergonomic way of calculating higher order derivatives
    @test all(derivative.(x->derivative.(K->C(1,0;K=K),x),rand(1000)) .>= -EPS)

    # VII: The price of a call-spread should approximate the price of a digital-call option
    # TODO - NOT IMPLEMENTED - 1) find out what call-spread is and price it; 2) find out to calculate price of digital call (it was really simple IIRC); 3) check they are roughly the same

    # VIII: The price of a digital-call option plus a digital-put option is equal to the price of a zero-coupon bond
    # TODO - Not implemented: see VII
end

@testset "Monte-Carlo validation" begin
    # Validation with Monte-Carlo
    # S-K: forward contract
    let S_t=100, t=0.01, K=100, r=0.02, σ=0.05, T=1, trials=100_000_000
        exact = S_t - K*exp(-r*(T-t))

        mc = mc_pricer((S,K,t,T,r)->S-K,S_t;t=t,T=T,K=K,r=r,σ=σ) |> first

        @test mc ≈ exact rtol=1e-3
    end

    # max(S-K,0): call option
    let S_t=100, t=0.01, K=100, r=0.02, σ=0.05, T=1, trials=100_000_000
        bs = C(S_t,t;T=T,K=K,r=r,σ=σ)

        mc = mc_pricer((S,K,t,T,r)->max(S-K,0),S_t;t=t,T=T,K=S_t,r=r,σ=σ) |> first

        @test mc ≈ bs rtol=1e-3
    end

    # Int(S>E): barrier option
    let S_t=100, t=0.01, E=100, r=0.02, σ=0.05, T=1, trials=100_000_000
        exact = barrier(S_t,t;E=E,r=r,σ=σ,T=T)

        mc = mc_pricer((S,E,t,T,r)->Int(S>E),S_t;t=t,T=T,K=E,r=r,σ=σ) |> first

        @test mc ≈ exact rtol=1e-3
    end
end
