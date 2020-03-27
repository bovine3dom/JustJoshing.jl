module JustJoshing

using Requires

include("p1_black_scholes_vanilla/black_scholes_vanilla.jl")

function __init__()
    @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("cuda/mc_pricers.jl")
end

end
