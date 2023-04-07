cd(@__DIR__)
using Pkg
Pkg.activate(".")

using DelimitedFiles, DifferentialEquations, Sundials, CairoMakie, Statistics
include("models/MasterEquation.jl")
include("models/PetrovGalerkinROM.jl")
include("models/CSEROM.jl")
include("models/BalancingROM.jl")

path = "MEdata/"

T_range = range(1000, 1500, step=50) 
p_range = 10 .^ range(-2, 0, length=3) 

u_periodic(t, ω) = [(1 + sin(ω*t))]
u_exp_decay(t, ω) = [ω * exp(-ω*t)]
u_exp_increase(t, ω) = [(1-exp(-ω*t))]
u_step(t, ω) = 1

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

# computational experiments
control_signals = [u_exp_decay]

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

ω_range = 10 .^ range(0, 11, length=12)
n_mode_range = [10]
horizon = 1e3
n_horizon = 300
min_time_exp = -11
scale = 1.0e3

t_range = vcat(0.0, collect(10 .^ range(min_time_exp, log10(horizon), n_horizon)))

## Reduced solutions
struct ReducedSol
    sol
    idx_set
    t
end

function (rs::ReducedSol)(t)
    return rs.sol(t)[rs.idx_set]
end

results = Dict()
problem_data = Dict()
for Temp in T_range, Pres in p_range
    temperature = Temp
    pressure = Pres
    T_name = round(Temp,digits = 4)
    P_name = round(Pres, digits=4)
    println("Temperature: $(Temp)K -- Pressure: $(Pres)bar")
    M = readdlm(string(path, "M_", P_name, "_", T_name, ".csv"), ',', Float64, header=false)
    F = readdlm(string(path, "K_", P_name, "_", T_name, ".csv"), ',', Float64, header=false)
    B = readdlm(string(path, "B_", P_name, "_", T_name, ".csv"), ',', Float64, header=false)
    T = readdlm(string(path, "T_", P_name, "_", T_name, ".csv"), ',', Float64, header=false)
    idcs = readdlm(string(path, "M_", P_name, "_", T_name, "idx.csv"), ',', Float64, header=false)

    iso_labels = [:acetylperoxy, :hydroperoxylvinoxy]
    product_labels = [:ketene, :lactone]

    ME, Bin = MasterEquation(M, idcs, B, F, T, iso_labels, product_labels)
    Bin *= scale
    W, Λ, Winv = compute_diagonalization(ME)
    
    S = zeros(length(ME.specs), size(ME.M,2))
    for i in eachindex(ME.specs)
        S[i,ME.spec_idcs[i]] .= 1
    end
    problem_data[temperature,pressure] = (ME,W,Λ,Winv,Bin,S)

    # cse georgievskii model
    cse_rom = CSEModel_Geo(ME, Bin, stationary_correction=false)

    # dst model
    dst_roms = Dict()
    for n_modes in n_mode_range 
        rom = PetrovGalerkinROM(W[:,1:n_modes], W[:,n_modes+1:end], 
                                Diagonal(Λ[1:n_modes]), 
                                Winv[1:n_modes,:]*Bin,
                                ME.F*W[:,1:n_modes], 
                                -ME.F*W[:,n_modes+1:end]*Diagonal(1 ./ Λ[n_modes+1:end])*Winv[n_modes+1:end,:]*Bin,
                                LiftingMap(W[:,1:n_modes], -W[:,n_modes+1:end]*Diagonal(1 ./ Λ[n_modes+1:end])*Winv[n_modes+1:end,:]*Bin))
        dst_roms[Symbol("dst_$n_modes")] = rom
    end

    # bt model
    t_int = 10 .^ range(-12, 3, length = 300)
    pushfirst!(t_int, 0.0)
    S = zeros(length(ME.specs), size(ME.M,2))
    for i in eachindex(ME.specs)
        S[i,ME.spec_idcs[i]] .= 1
    end
    C_factor = reduce(hcat, sqrt(t_int[i]-t_int[i-1])*W*Diagonal(exp.(t_int[i]*Λ))*Winv*Bin for i in 2:length(t_int))
    O_factor = reduce(hcat, sqrt(t_int[i]-t_int[i-1])*Winv'*Diagonal(exp.(t_int[i]*Λ))*W'*S' for i in 2:length(t_int))
    T, Tinv, Σ = factor_balancing(C_factor, O_factor; rtol = 1e-40, atol = 1e-14)
    
    bt_roms = Dict()
    for n_modes in n_mode_range
        bt_rom = BalancedROM(T, Tinv, W*Diagonal(Λ)*Winv, Bin, n_modes; F = ME.F, stationary_correction=false)
        bt_roms[Symbol("bt_$n_modes")] = bt_rom
    end

    # models to consider
    models = [cse_rom => :cse,
              [val => key for (key, val) in dst_roms]...,
              [val => key for (key, val) in bt_roms]...]
    
    solutions = Dict()
    for (model, label) in models
        solutions[label] = Dict()
        print(label)
        print(" ")
        for u in control_signals
            for ω in ω_range
                n = size(model.A,1)
                m = size(ME.F,1)
                A_extended = [model.A zeros(n,m);
                              model.C zeros(m,m)]
                B_extended = [model.B;
                              model.D]
                prob = ODEProblem(master_equation!, zeros(n+m), (0, horizon), (A_extended, B_extended, t -> u(t,ω)))
                red_sol = solve(prob, FBDF(), dtmin=1e-80, maxiters=1e12, reltol=1e-12, abstol=1e-12, saveat = t_range) 
                sol = ReducedSol(red_sol, 1:n, red_sol.t)
                prod = ReducedSol(red_sol, n+1:n+m, red_sol.t)
                solutions[label][u, ω] = (sol, prod, model) 
            end
        end
    end
    results[temperature,pressure] = solutions
    print("\n")
end

# visualization
models = vcat(:cse,
              [Symbol("dst_$n_mode") for n_mode in n_mode_range],
              [Symbol("bt_$n_mode") for n_mode in n_mode_range])

labels = Dict(:cse => "CSE", 
              [Symbol("dst_$n_mode") => "DST $n_mode" for n_mode in n_mode_range]..., 
              [Symbol("bt_$n_mode") => "BT $n_mode" for n_mode in n_mode_range]...)

# error vs decay rate for exponential decay
rel_error = Dict()
tested_models = [:cse, :dst_10, :bt_10]
u = u_exp_decay
for p in p_range
    for T in T_range, ω in ω_range
        # product 
        t_test = 1/ω
        # analytical solution
        ME, W, Λ, Winv, Bin, S = problem_data[T,p] 
        p_true = analytical_prod(u, t_test, ω, W, Λ, Winv, Bin, ME.F)
        for label in tested_models
            sol, prod, model = results[T,p][label][u,ω]
            p_red = evaluate_prod(model.lift(sol(t_test),u(t_test,ω)),u,t_test,ω,W,Λ,Winv,Bin,ME.F)
            rel_error[T,p,ω,label] = norm(p_red - p_true)/norm(p_true)
        end
    end
end
min_err = 1e-9 
max_err = 1.0  

colorticks = ([-9,-6,-3,0], vcat(latexstring("\\leq 10^{-9}"),
                                 [latexstring("10^{$e}") for e in [-6,-3]],
                                 latexstring("\\geq 10^{0}")))
for p in p_range
    fig = Figure(fontsize=24, resolution = (1500, 500))
    axs = [Axis(fig[1,k], xlabel = L"\text{decay time scale } \tau \text{ [s]}", 
                        ylabel = model == :cse ? L"\text{Temperature [K]}" : "", 
                        yticks = model == :cse ? T_range : ([], []),
                        xscale = log10,
                        xticks = (10.0 .^ range(-11, 0, step = 2),[latexstring("10 ^ {$e}") for e in range(-11, 0, step=2)]),
                        title = labels[model])
            for (k,model) in enumerate(tested_models)]           
    T_grid = [T for T in T_range]
    ω_grid = vcat(0.999e-9,[1/ω for ω in reverse(ω_range)])
    val_grid = [rel_error[T,p,ω,:cse] for ω in reverse(ω_range), T in T_range]
    for (k,model) in enumerate(tested_models)
        val_grid = [rel_error[T,p,ω,model] for ω in reverse(ω_range), T in T_range]
        global hm = heatmap!(axs[k], ω_grid, 
                                    T_grid, 
                                    log10.(val_grid), 
                                    colormap = :RdBu_9, colorrange = log10.((min_err, max_err)))
    end
    cb = Colorbar(fig[1, end+1], hm, 
                ticks = colorticks,
                label = L"\text{rel. error } \frac{\Vert p(\tau) - \hat{p}(\tau) \Vert}{\Vert p(\tau) \Vert}"
                )
    P_name = round(p, digits = 4)
    save("figures/exp_decay_short_timescale_p_$P_name.pdf", fig)
    display(fig)
end 

