cd(@__DIR__)
using Pkg
Pkg.activate("../../.")

using DelimitedFiles, DifferentialEquations, Sundials, CairoMakie

path = "/Users/holtorf/.julia/dev/MasterEquation/toy_problem/MEdata/data_pieces/"

# 400/0.01 is hard apparently

T_range = [400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300]#, 1400, 1500]
p_range = [0.01, 0.1, 1.0]#, 10.0]
temperature = 1300
pressure = 0.01

M = readdlm(string(path, "M_", pressure, "_", temperature, ".csv"), ',', Float64, header=false)
F = readdlm(string(path, "K_", pressure, "_", temperature, ".csv"), ',', Float64, header=false)
B = readdlm(string(path, "B_", pressure, "_", temperature, ".csv"), ',', Float64, header=false)
T = readdlm(string(path, "T_", pressure, "_", temperature, ".csv"), ',', Float64, header=false)
idcs = readdlm(string(path, "M_", pressure, "_", temperature, "idx.csv"), ',', Float64, header=false)

iso_labels = [:acetylperoxy, :hydroperoxylvinoxy]
product_labels = [:acetyl, :ketene]

include("MasterEquation.jl")
include("NaiveTruncation.jl")
include("PetrovGalerkinROM.jl")
include("CSEROM.jl")
include("CSCROM.jl")
include("BalancingROM.jl")
include("product_eval.jl")

scale = 1.0e3
ME, Bin = MasterEquation(M, idcs, B, F, T, iso_labels, product_labels)
Bin *= scale
W, Λ, Winv = compute_diagonalization(ME)
@show Λ[1:3]

# cse models
cse_g = CSEModel_Geo(ME, Bin, stationary_correction=false)
cse = CSEModel(ME, Bin, stationary_correction=true)

### other modal reductions
n_modes = 10

## most controllable subspace truncation
function ControllabilityGramian(ME, Bin)
    W, Λ, Winv = compute_diagonalization(ME)
    n = length(Λ)
    b = Winv*Bin
    C = -[b[i]*b[j]/(Λ[i]+Λ[j]) for i in 1:n, j in 1:n]
    return W, C, Winv
end

function ControllabilityGramian(ME, Bin, t)
    W, Λ, Winv = compute_diagonalization(ME)
    n = length(Λ)
    b = Winv*Bin
    C = [b[i]*b[j]/(Λ[i]+Λ[j])*(exp(t*(Λ[i]+Λ[j])) - 1) for i in 1:n, j in 1:n]
    return W, C
end

W_obs, C = ControllabilityGramian(ME, Bin)
L, U = eigen(C,sortby=-)
T = W_obs*U
V = T[:, 1:n_modes]
N = nullspace(V')

pg_rom = build_PetrovGalerkinROM(V, N, W*Diagonal(Λ)*Winv, Bin; F = ME.F, stationary_correction=true)

## balanced truncation
t_int = 10 .^ range(-12, 3, length= 1000)
pushfirst!(t_int, 0.0)
S = zeros(length(ME.specs), size(ME.M,2))
for i in eachindex(ME.specs)
     S[i,ME.spec_idcs[i]] .= 1
end
C_factor = reduce(hcat, sqrt(t_int[i]-t_int[i-1])*W*Diagonal(exp.(t_int[i]*Λ))*Winv*Bin for i in 2:length(t_int))
O_factor = reduce(hcat, sqrt(t_int[i]-t_int[i-1])*Winv'*Diagonal(exp.(t_int[i]*Λ))*W'*S' for i in 2:length(t_int))
T, Tinv, Σ = factor_balancing(C_factor, O_factor; atol = 1e-12)
T_aug = nullspace(T)
T = vcat(T, T_aug')
Tinv = inv(T)
b_rom = BalancedROM(T, Tinv, W*Diagonal(Λ)*Winv, Bin, n_modes; F = ME.F, stationary_correction=true)

## non-negative physical_reduction
C = max.(0,C_factor*C_factor')
V, H, N = non_negative_reduction(C, n_modes)
nn_rom = build_PetrovGalerkinROM(V, N, W*Diagonal(Λ)*Winv, Bin; F = ME.F, stationary_correction=true)

## dominant subspace truncation rom
dom_rom = PetrovGalerkinROM(W[:,1:n_modes], 
                            W[:,n_modes+1:end],
                            Diagonal(Λ[1:n_modes]),
                            Winv[1:n_modes,:]*Bin,
                            ME.F*W[:,1:n_modes], 
                            -ME.F*W[:,n_modes+1:end]*Diagonal(1 ./ Λ[n_modes+1:end])*Winv[n_modes+1:end,:]*Bin,
                            LiftingMap(W[:,1:n_modes], - W[:,n_modes+1:end]*Diagonal(1 ./ Λ[n_modes+1:end])*Winv[n_modes+1:end,:]*Bin))

# models to be run
models = [cse, cse_g, b_rom, dom_rom]
#nn_rom, pg_rom, b_rom, dom_rom]

u_periodic(t, ω) = [(1 + sin(ω*t))]
u_exp_decay(t, ω) = [ω * exp(-ω*t)]
u_exp_increase(t, ω) = [(1-exp(-ω*t))]
u_step(t, ω) = 1

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

function evaluate_prod(c,u,t,ω,W,Λ,Winv,Bin,F)
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

control_signals = [u_exp_increase, u_exp_decay]#, u_periodic]
ω_range = 10 .^ range(-2, 6, length=10)

N_periods = 10
min_horizon = 1e-3
n_horizon = 100

## Reduced solutions
struct ReducedSol
    sol
    idx_set
    t
end

function (rs::ReducedSol)(t)
    return rs.sol(t)[rs.idx_set]
end

solutions = Dict()
for model in models
    solutions[model] = Dict()
    for u in control_signals
        for ω in ω_range
            Tf = max(2π*N_periods/ω, min_horizon)
            checkpoints = vcat(0.0, collect(10 .^ range(-9, log10(Tf), n_horizon)))
            n = size(model.A,1)
            m = size(ME.F,1)
            A_extended = [model.A zeros(n,m);
                          model.C zeros(m,m)]
            B_extended = [model.B;
                          model.D]
            prob = ODEProblem(master_equation!, zeros(n+m), (0, Tf), (A_extended, B_extended, t -> u(t,ω)))
            full_sol = solve(prob, FBDF(), maxiters=typemax(Int32), dtmin=1e-36, abstol=1e-13, reltol=1e-13, saveat = checkpoints) 
            println(full_sol.retcode)
            sol = ReducedSol(full_sol, 1:n, full_sol.t)
            prod = ReducedSol(full_sol, n+1:n+m, full_sol.t)
            solutions[model][u, ω] = (sol, prod) 
        end
    end
end

## Full solution
reference_solution = Dict()
t_scale = 1
for u in control_signals
    for ω in ω_range
        println(ω)
        Tf = max(2π*N_periods/ω, min_horizon)
        checkpoints = vcat(0.0, collect(10 .^ range(-9, log10(Tf), n_horizon)))
        n = size(ME.M,1)
        m = size(ME.F,1)
        M = W * Diagonal(Λ) * Winv
        A_extended = [M zeros(n,m);
                      ME.F zeros(m,m)]
        B_extended = [Bin;
                      zeros(m,size(Bin,2))]
        prob = ODEProblem(master_equation!, zeros(n+m), (0, Tf), (A_extended, B_extended, t -> u(t,ω)))
        full_sol = solve(prob, FBDF(), abstol = 1e-13, reltol=1e-13, saveat = checkpoints) 
        sol = ReducedSol(full_sol, 1:n, full_sol.t)
        prod = ReducedSol(full_sol, n+1:n+m, full_sol.t)
        reference_solution[u, ω] = (sol, prod)  
    end
end


## Plotting
colors = Dict(cse_g => :black,
              cse => :red,
              pg_rom => :green, 
              b_rom => :blue, 
              nn_rom => :orange,
              dom_rom => :purple
              )
u = u_exp_increase
equilibration_time = -1/Λ[2]

# MSE in "chemical" space
S = zeros(length(ME.specs), size(ME.M,2))
for i in eachindex(ME.specs)
    S[i,ME.spec_idcs[i]] .= 1
end

fig_chem = Figure(fontsize=24, resolution = (1000, 500))
ax = Axis(fig_chem[1,1], xlabel = L"\text{time}", ylabel = L"\text{relative error}", 
                         yscale = log10, xscale=log10,
                         yticks = (10.0 .^ range(-12,1), [latexstring("10 ^ {$e}") for e in range(-12,1)]),
                         yminorticksvisible = true, yminorgridvisible = true,
                         yminorticks = IntervalsBetween(8),
                         xticks = (10.0 .^ range(-9, ceil(Int, log10(max(2π*N_periods/ω_range[1], min_horizon)))),
                                   [latexstring("10 ^ {$e}") for e in range(-9, ceil(Int, log10(max(2π*N_periods/ω_range[1], min_horizon))))]),
                         xminorticksvisible = true, xminorgridvisible = true,
                         xminorticks = IntervalsBetween(8))
for ω in ω_range
    trange = reference_solution[u,ω][1].t[2:end]*t_scale
    full_model = [S*analytical_sol(u, t, ω, W, Λ, Winv, Bin) for t in trange]
    for model in models
        error = [norm(full_model[i] - S*model.lift(solutions[model][u,ω][1](t), u(t,ω))) / 
                 norm(full_model[i]) for (i,t) in enumerate(trange)]
        lines!(ax, trange, [e > 1e-12 ? e : missing for e in error], 
                   color = colors[model],
                   label = string(ω)) 
    end
end 
display(fig_chem)

# full state
fig_full = Figure(fontsize=24, resolution = (1000, 500))
ax = Axis(fig_full[1,1], xlabel = L"\text{time}", ylabel = L"\text{relative error}", 
                         yscale = log10, xscale=log10,
                         yticks = (10.0 .^ range(-12,1), [latexstring("10 ^ {$e}") for e in range(-12,1)]),
                         yminorticksvisible = true, yminorgridvisible = true,
                         yminorticks = IntervalsBetween(8),
                         xticks = (10.0 .^ range(-9, ceil(Int, log10(max(2π*N_periods/ω_range[1], min_horizon)))),
                                   [latexstring("10 ^ {$e}") for e in range(-9, ceil(Int, log10(max(2π*N_periods/ω_range[1], min_horizon))))]),
                         xminorticksvisible = true, xminorgridvisible = true,
                         xminorticks = IntervalsBetween(8))
for ω in ω_range
    trange = reference_solution[u,ω][1].t[2:end]*t_scale
    full_model = [analytical_sol(u, t, ω, W, Λ, Winv, Bin) for t in trange]
    for model in models
        error = [norm(full_model[i] - model.lift(solutions[model][u,ω][1](t), u(t,ω))) / 
                 norm(full_model[i]) for (i,t) in enumerate(trange)]
        lines!(ax, trange, [e > 1e-12 ? e : missing for e in error], 
                   color = colors[model],
                   label = string(ω)) 
    end
end
display(fig_full)

# MSE in product space -> alternative computation
fig_prod_mb = Figure(fontsize=24, resolution=(1000,500))
ax = Axis(fig_prod_mb[1,1], xlabel = L"\text{time}", ylabel = L"\text{relative error}", 
                            yscale = log10, xscale=log10,
                            yticks = (10.0 .^ range(-12,1), [latexstring("10 ^ {$e}") for e in range(-12,1)]),
                            yminorticksvisible = true, yminorgridvisible = true,
                            yminorticks = IntervalsBetween(8),
                            xticks = (10.0 .^ range(-9, ceil(Int, log10(max(2π*N_periods/ω_range[1], min_horizon)))),
                                    [latexstring("10 ^ {$e}") for e in range(-9, ceil(Int, log10(max(2π*N_periods/ω_range[1], min_horizon))))]),
                            xminorticksvisible = true, xminorgridvisible = true,
                            xminorticks = IntervalsBetween(8))
for ω in ω_range
    trange = reference_solution[u,ω][1].t[2:end]*t_scale
    true_prod = [analytical_prod(u, t, ω, W, Λ, Winv, Bin, ME.F) for t in trange]
    for model in models
        error = [norm(true_prod[i] - evaluate_prod(model.lift(solutions[model][u,ω][1](t), u(t,ω)), u, t, ω, W, Λ, Winv, Bin, ME.F))
                      for (i,t) in enumerate(trange)]
        lines!(ax, trange, [e > 1e-12 ? e : missing for e in error], color = colors[model], label = string(ω))    
    end
end
display(fig_prod_mb)

# decommissioned because less accurate
# MSE in full space
fig_full = Figure(fontsize=24)
ax = Axis(fig_full[1,1], xlabel = "time", ylabel = "relative error", 
                    yscale = log10, xscale=log10)
for model in models
    for ω in ω_range
        trange = reference_solution[u,ω][1].t[2:end]*t_scale
        normalized_trange = trange
        error = [norm(analytical_sol(u, t, ω, W, Λ, Winv, Bin) - model.lift(solutions[model][u,ω][1](t), u(t,ω))) / 
                 norm(analytical_sol(u, t, ω, W, Λ, Winv, Bin)) for t in trange]
        lines!(ax, normalized_trange, [e > 1e-12 ? e : missing for e in error], 
                   color = colors[model],
                   label = string(ω)) 
    end
end
display(fig_full)

# MSE in product space
fig_prod = Figure(fontsize=24)
ax = Axis(fig_prod[1,1], xlabel = "time", 
                    ylabel = "relative error", 
                    yscale = log10, xscale=log10)
for model in models 
    for ω in ω_range
        trange = solutions[model][u,ω][1].t 
        normalized_trange = trange 
        error = [norm(analytical_prod(u, t, ω, W, Λ, Winv, Bin, ME.F) - solutions[model][u,ω][2](t)) / 
                 (1 + norm(analytical_prod(u, t, ω, W, Λ, Winv, Bin, ME.F))) for (i,t) in enumerate(trange)]
        lines!(ax, normalized_trange, [e > 1e-12 ? e : missing for e in error], color = colors[model], label = string(ω))    
    end
end
display(fig_prod)