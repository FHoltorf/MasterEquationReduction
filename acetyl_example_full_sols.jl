cd(@__DIR__)
using Pkg
Pkg.activate(".")

using FileIO

T_range = range(1000, 1500, step=50) 
p_range = 10 .^ range(-2, 0, length=3)
ω_range = 10 .^ range(-2, 9, length=12)

u_periodic(t, ω) = [(1 + sin(ω*t))]
u_exp_decay(t, ω) = [ω * exp(-ω*t)]
u_exp_increase(t, ω) = [(1-exp(-ω*t))]
u_step(t, ω) = 1

control_signals = [u_exp_increase, u_exp_decay]
control_labels = Dict(u_exp_decay => "exp_decay",
                      u_exp_increase => "exp_increase")

function analytical_sol(u,t,ω,W,Λ,Winv,Bin)
    if u == u_periodic
        return W*Diagonal(-1 ./ Λ .* (1 .- exp.(t*Λ)) + (Λ .* sin(ω*t) .+ ω * cos(ω*t) .-  ω * exp.(t*Λ)) ./ (Λ .^ 2 .+ ω^2) )*Winv*Bin*[1] 
    elseif u == u_exp_decay
        return W*Diagonal(ω ./ (Λ .+ ω) .* (exp.(t*Λ) .- exp(-t*ω)))*Winv*Bin*[1] 
    elseif u == u_exp_increase
        return W*Diagonal(-1 ./ Λ .* (1 .- exp.(t*Λ)) - 1 ./ (Λ .+ ω) .* (exp.(t*Λ) .- exp(-t*ω)))*Winv*Bin*[1] 
    elseif u == u_step
        return W*Diagonal(-1 ./ Λ .* (1 .- exp.(t*Λ)))*Winv*Bin*[1] 
    else
        error("No analytical solution available")
    end
end

function analytical_prod(u,t,ω,W,Λ,Winv,Bin,F)
    c = analytical_sol(u,t,ω,W,Λ,Winv,Bin)
    if u == u_periodic
        fed_material = t + 1/ω*(1 - cos(ω*t))
    elseif u == u_exp_decay
        fed_material = 1 - exp(-ω*t) 
    elseif u == u_exp_increase
        fed_material = t - 1/ω * (1 - exp(-ω*t))
    else
        error("No analytical solution available")
    end
    converted_material =  - c + Bin*[fed_material]
    p = - F * W * Diagonal( 1 ./ Λ ) * Winv * converted_material
    return p
end

sols = readdir("full_model_results/")
for T in T_range, p in p_range
    ME, W, Λ, Winv, Bin, S = problem_data[T,p]  
    T_name = round(T, digits = 4)
    P_name = round(p, digits=4)
    filename = "res_$(P_name)_$(T_name).jld2"
    if !(filename in sols)
        full_sol = Dict()
        full_prod = Dict()  
        for u in control_signals
            for ω in ω_range
                full_sol[control_labels[u],ω] = [analytical_sol(u, t, ω, W, Λ, Winv, Bin) for t in t_range]
                full_prod[control_labels[u],ω] = [Float64.(evaluate_prod(BigFloat.(full_sol[control_labels[u],ω][i]), u, 
                                                                         BigFloat(t_range[i]), BigFloat(ω), 
                                                                         W, Λ, Winv, Bin, ME.F)) for i in eachindex(t_range)]
            end
        end
        println("$filename done.")
        save(joinpath("full_model_results", filename), "concentration", full_sol, "product", full_prod)
    end
end