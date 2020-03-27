# NB: this test won't run automatically - you will have to include it manually
using JustJoshing
using Test
using CuArrays

using Statistics: mean

@testset "CPU/GPU consistency" begin
    let S_t=100, t=0.01, r=0.02, σ=0.05, T=1.0, trials=100_000_000
        payoff = S->max(S-100,0)
        kwargs = (S_0=S_t,t=t,T=T,r=r,σ=σ,trials=trials)

        cpu_vanilla = mc_pricer(payoff;kwargs...) |> first
        gpu_vanilla = JustJoshing.CUDA.mc_pricer(payoff;kwargs...) |> first

        @test cpu_vanilla ≈ gpu_vanilla rtol=1e-2

        # GPU seems to always be greater than CPU by a small amount
        @test_broken cpu_vanilla ≈ gpu_vanilla rtol=1e-3

        kwargs = (S_0=S_t,t=0:0.1:1,trials=trials)
        cpu_pathdep = mc_pricer_pathdep(S->max(mean(S)-100,0);kwargs...) |> first
        gpu_pathdep = JustJoshing.CUDA.mc_pricer_pathdep(S->max.(mean(S;dims=2).-100,0);kwargs...) |> first

        # These numbers are very different
        @test_broken cpu_pathdep ≈ gpu_pathdep rtol=1e-3

    end
end
