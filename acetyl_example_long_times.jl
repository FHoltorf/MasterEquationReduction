cd(@__DIR__)
using Pkg
Pkg.activate(".")

using DelimitedFiles, DifferentialEquations, Sundials, CairoMakie, Statistics, FileIO
include("models/MasterEquation.jl")
include("models/PetrovGalerkinROM.jl")
include("models/CSEROM.jl")
include("models/BalancingROM.jl")

path = "MEdata/data_pieces_no_reverse/"

T_range = range(1000, 1500, step=50) #[1000, 1100, 1200, 1300]
p_range = 10 .^ range(-2, 0, length=3) #[0.01, 0.1, 1.0]

u_periodic(t, ω) = [(1 + sin(ω*t))]
u_exp_decay(t, ω) = [ω * exp(-ω*t)]
u_exp_increase(t, ω) = [(1-exp(-ω*t))]
u_step(t, ω) = 1

# computational experiments
control_signals = [u_exp_increase] #u_exp_decay]
control_labels = Dict(u_exp_decay => "exp_decay",
                      u_exp_increase => "exp_increase")

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

ω_range = 10 .^ range(-2, 9, length=12)
n_mode_range = [2, 4, 6, 8, 10]
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
    product_labels = [:acetyl, :ketene]

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
    # dst_rom = CSEModel(ME, Bin, stationary_correction=true)
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
                if label == :cse 
                    rel_tol = 1e-14
                    abs_tol = 1e-14
                else
                    rel_tol = 1e-14
                    abs_tol = 1e-14
                end
                red_sol = solve(prob, CVODE_BDF(), dtmin=1e-24, reltol=rel_tol, abstol=abs_tol, saveat = t_range) 
                # tried RadauIIA5, QNDF, FBDF and CVODE_BDF
                sol = ReducedSol(red_sol, 1:n, red_sol.t)
                prod = ReducedSol(red_sol, n+1:n+m, red_sol.t)
                solutions[label][u, ω] = (sol, prod, model) 
            end
        end
    end
    results[temperature,pressure] = solutions
    print("\n")
end

# full/analytical solution
function compare(full_sol, red_sol; singularity = 0.0)
    abs_error = [norm(full_sol[i] - red_sol[i]) for i in eachindex(full_sol)]
    rel_error = abs_error ./ (singularity .+ norm.(full_sol))
    return abs_error, rel_error
end

# visualization
models = [:cse,
          [Symbol("dst_$n_mode") for n_mode in n_mode_range]...,
          [Symbol("bt_$n_mode") for n_mode in n_mode_range]...]

labels = Dict(:cse => "CSE", 
              [Symbol("dst_$n_mode") => "DST $n_mode" for n_mode in n_mode_range]..., 
              [Symbol("bt_$n_mode") => "BT $n_mode" for n_mode in n_mode_range]...)

markers = [:circle, :dtriangle, :rect, :diamond, :hexagon, :xcross, :star8]
color = Dict(:cse => (:black, nothing), 
             [Symbol("dst_$n_mode") => (:red, markers[i]) for (i,n_mode) in enumerate(n_mode_range)]...,
             [Symbol("bt_$n_mode") => (:blue, markers[i]) for (i,n_mode) in enumerate(n_mode_range)]...)

min_error = -9
max_error = 10
# interesting observables/comparisons
singularity = 0.0 # => regularization for numerical accuracy

# for now compare: CSE, DST, BT

# rel. error in \hat{c} over time 
# 3 plots in a row, one for each observable, trace for each model
# -> avg (T,p) 
# -> max/min (T,p) 
# -> distribution (T,p)

rel_macro_err = Dict()
abs_macro_err = Dict()
for label in models
    println("evaluating: $label")
    for u in control_signals
        println("    control signal $(control_labels[u])")
        rel_macro_err[label,u] = Dict()
        abs_macro_err[label,u] = Dict()
        for T in T_range, p in p_range
            T_name = round(T, digits = 4)
            P_name = round(p, digits = 4)
            filename = "res_$(P_name)_$(T_name).jld2"
            full_sol = load(joinpath("full_model_results", filename), "concentration")
            for ω in ω_range
                ME, W, Λ, Winv, Bin, S = problem_data[T,p]
                sol, prod, model = results[T,p][label][u,ω]
                if sol.sol.retcode == :Success
                    if label == :cse 
                        abs_err, rel_err = compare([S*c for c in full_sol[control_labels[u], ω][2:end]], 
                                                   [sol(t) for t in t_range[2:end]]; singularity=singularity)
                    else
                        abs_err, rel_err = compare([S*c for c in full_sol[control_labels[u], ω][2:end]], 
                                                   [S*model.lift(sol(t), u(t,ω)) for t in t_range[2:end]]; singularity=singularity)
                    end
                    abs_macro_err[label, u][T,p,ω] = abs_err
                    rel_macro_err[label, u][T,p,ω] = rel_err
                else
                    println("Model $label failed at (T,p) = ($T,$p) due to $(sol.sol.retcode)")
                end
            end
        end
    end
end

# T = T_range[end]
# p = p_range[end]
# T_name = round(T, digits = 4)
# P_name = round(p, digits = 4)
# filename = "res_$(P_name)_$(T_name).jld2"
# full_sol = load(joinpath("full_model_results", filename), "concentration")
# u = u_exp_decay
# ω = ω[end]
# test = [norm(c) for c in full_sol[control_labels[u],ω][2:end]]
# relative error
for u in control_signals
    fig = Figure(fontsize=24, resolution = (1200, 500))
    ax = Axis(fig[1,1], xlabel = L"\text{time } t", 
                        ylabel = L"\text{relative error }  \frac{\Vert Sc(t) - S\hat{c}(t) \Vert}{\Vert S c(t) \Vert}", 
                        yscale = log10, xscale=log10,
                        yticks = (10.0 .^ range(min_error,max_error), [latexstring("10 ^ {$e}") for e in range(min_error,max_error)]),
                        yminorticksvisible = true, yminorgridvisible = true,
                        yminorticks = IntervalsBetween(8),
                        xticks = (10.0 .^ range(min_time_exp, ceil(Int, log10(horizon))),
                                [latexstring("10 ^ {$e}") for e in range(min_time_exp, ceil(Int, log10(horizon)))]),
                        xminorticksvisible = true, xminorgridvisible = true,
                        xminorticks = IntervalsBetween(8))
    #ylims!(ax, 1e-9, 1e2)
    xlims!(ax, 10.0 .^ min_time_exp, 10.0 .^ ceil(Int, log10(horizon)))
    plots = []
    plot_labels = String[]
    for label in models
        means = mean(rel_macro_err[label, u][key] for key in keys(rel_macro_err[label, u]))
        if !isnothing(color[label][2])
            p = scatterlines!(ax, t_range[2:end], [m > 10.0 ^ min_error ? m : missing for m in means], 
                                  color = color[label][1], 
                                  marker = color[label][2],
                                  markersize = [mod(i-1, 25) == 0 ? 15 : 0 for i in 1:length(t_range)-1],
                                  linewidth=2)
            push!(plots, p)
        else
            p = lines!(ax, t_range[2:end], [m > 10.0 ^ min_error ? m : missing for m in means], 
                           color = color[label][1], 
                           linewidth=2)
            push!(plots, p)
        end
        #lines!(ax, t_range[2:end], [m > 1e-9 ? m : missing for m in test])
        push!(plot_labels, labels[label])
    end
    Legend(fig[1,2], plots, plot_labels)
    save(joinpath(@__DIR__,"figures",control_labels[u]*"_rel_macro_comparison.pdf"), fig)
end

# abs error
for u in control_signals
    fig = Figure(fontsize=24, resolution = (1200, 500))
    ax = Axis(fig[1,1], xlabel = L"\text{time } t", 
                        ylabel = L"\text{absolute error }  \Vert Sc(t) - S\hat{c}(t) \Vert", 
                        yscale = log10, xscale=log10,
                        yticks = (10.0 .^ range(min_error,max_error), [latexstring("10 ^ {$e}") for e in range(min_error,max_error)]),
                        yminorticksvisible = true, yminorgridvisible = true,
                        yminorticks = IntervalsBetween(8),
                        xticks = (10.0 .^ range(min_time_exp, ceil(Int, log10(horizon))),
                                [latexstring("10 ^ {$e}") for e in range(min_time_exp, ceil(Int, log10(horizon)))]),
                        xminorticksvisible = true, xminorgridvisible = true,
                        xminorticks = IntervalsBetween(8))
    xlims!(ax, 10.0 .^ min_time_exp, 10.0 .^ ceil(Int, log10(horizon)))    
    plots = []
    plot_labels = String[]
    for label in models
        means = mean(abs_macro_err[label, u][key] for key in keys(abs_macro_err[label, u]))
        if !isnothing(color[label][2])
            p = scatterlines!(ax, t_range[2:end], [m > 10.0 ^ min_error ? m : missing for m in means], 
                                  color = color[label][1], 
                                  marker = color[label][2],
                                  markersize = [mod(i-1, 25) == 0 ? 15 : 0 for i in 1:length(t_range)-1],
                                  linewidth=2)
            push!(plots, p)
        else
            p = lines!(ax, t_range[2:end], [m > 10.0 ^ min_error ? m : missing for m in means], 
                           color = color[label][1], 
                           linewidth=2)
            push!(plots, p)
        end
        push!(plot_labels, labels[label])
    end
    Legend(fig[1,2], plots, plot_labels)
    save(joinpath(@__DIR__,"figures",control_labels[u]*"_abs_macro_comparison.pdf"), fig)
end



# rel. error in c over time 
# 3 plots in a row, one for each observable, trace for each model
# -> avg over (T,p,ω) (or any other statistic)
rel_micro_err = Dict()
abs_micro_err = Dict()
for label in models
    println("evaluating: $label")
    for u in control_signals
        println("    control signal $(control_labels[u])")
        rel_micro_err[label,u] = Dict()
        abs_micro_err[label,u] = Dict()
        for T in T_range, p in p_range
            T_name = round(T, digits = 4)
            P_name = round(p, digits = 4)
            filename = "res_$(P_name)_$(T_name).jld2"
            full_sol = load(joinpath("full_model_results", filename), "concentration")
            for ω in ω_range
                ME, W, Λ, Winv, Bin, S = problem_data[T,p]
                sol, prod, model = results[T,p][label][u,ω]
                if sol.sol.retcode == :Success
                    abs_err, rel_err = compare([c for c in full_sol[control_labels[u], ω][2:end]], 
                                               [model.lift(sol(t), u(t,ω)) for t in t_range[2:end]]; singularity=singularity)
                    abs_micro_err[label, u][T,p,ω] = abs_err
                    rel_micro_err[label, u][T,p,ω] = rel_err
                else
                    println("Model $label failed at (T,p) = ($T,$p) due to $(sol.sol.retcode)")
                end
            end
        end
    end
end

# rel error
for u in control_signals
    fig = Figure(fontsize=24, resolution = (1200, 500))
    ax = Axis(fig[1,1], xlabel = L"\text{time } t", 
                        ylabel = L"\text{relative error }  \frac{\Vert c(t) - \hat{c}(t) \Vert}{ \Vert c(t)\Vert}", 
                        yscale = log10, xscale=log10,
                        yticks = (10.0 .^ range(min_error,max_error), [latexstring("10 ^ {$e}") for e in range(min_error,max_error)]),
                        yminorticksvisible = true, yminorgridvisible = true,
                        yminorticks = IntervalsBetween(8),
                        xticks = (10.0 .^ range(min_time_exp, ceil(Int, log10(horizon))),
                                [latexstring("10 ^ {$e}") for e in range(min_time_exp, ceil(Int, log10(horizon)))]),
                        xminorticksvisible = true, xminorgridvisible = true,
                        xminorticks = IntervalsBetween(8))
    xlims!(ax, 10.0 .^ min_time_exp, 10.0 .^ ceil(Int, log10(horizon)))
    plots = []
    plot_labels = String[]
    for label in models
        means = mean(rel_micro_err[label, u][key] for key in keys(rel_micro_err[label, u]))
        if !isnothing(color[label][2])
            p = scatterlines!(ax, t_range[2:end], [m > 10.0 ^ min_error ? m : missing for m in means], 
                                  color = color[label][1], 
                                  marker = color[label][2],
                                  markersize = [mod(i-1, 25) == 0 ? 15 : 0 for i in 1:length(t_range)-1],
                                  linewidth=2)
            push!(plots, p)
        else
            p = lines!(ax, t_range[2:end], [m > 10.0 ^ min_error ? m : missing for m in means], 
                           color = color[label][1], 
                           linewidth=2)
            push!(plots, p)
        end
        push!(plot_labels, labels[label])
    end
    Legend(fig[1,2], plots, plot_labels)
    save(joinpath(@__DIR__,"figures",control_labels[u]*"_rel_micro_comparison.pdf"), fig)
end

# abs error
for u in control_signals
    fig = Figure(fontsize=24, resolution = (1200, 500))
    ax = Axis(fig[1,1], xlabel = L"\text{time } t", 
                        ylabel = L"\text{absolute error }  \Vert c(t) - \hat{c}(t) \Vert", 
                        yscale = log10, xscale=log10,
                        yticks = (10.0 .^ range(min_error,max_error), [latexstring("10 ^ {$e}") for e in range(min_error,max_error)]),
                        yminorticksvisible = true, yminorgridvisible = true,
                        yminorticks = IntervalsBetween(8),
                        xticks = (10.0 .^ range(min_time_exp, ceil(Int, log10(horizon))),
                                [latexstring("10 ^ {$e}") for e in range(min_time_exp, ceil(Int, log10(horizon)))]),
                        xminorticksvisible = true, xminorgridvisible = true,
                        xminorticks = IntervalsBetween(8))
    xlims!(ax, 10.0 .^ min_time_exp, 10.0 .^ ceil(Int, log10(horizon)))
    plots = []
    plot_labels = String[]
    for label in models
        means = mean(abs_micro_err[label, u][key] for key in keys(abs_micro_err[label, u]))
        if !isnothing(color[label][2])
            p = scatterlines!(ax, t_range[2:end], [m > 10.0 ^ min_error ? m : missing for m in means], 
                                  color = color[label][1], 
                                  marker = color[label][2],
                                  markersize = [mod(i-1, 25) == 0 ? 15 : 0 for i in 1:length(t_range)-1],
                                  linewidth=2)
            push!(plots, p)
        else
            p = lines!(ax, t_range[2:end], [m > 10.0 ^ min_error ? m : missing for m in means], 
                           color = color[label][1], 
                           linewidth=2)
            push!(plots, p)
        end
        push!(plot_labels, labels[label])
    end
    Legend(fig[1,2], plots, plot_labels)
    save(joinpath(@__DIR__,"figures",control_labels[u]*"_abs_micro_comparison.pdf"), fig)
end

# error in p over time 
# 3 plots in a row, one for each observable, trace for each model
# -> mean over (T,p,ω) (or any other statstic?)

abs_prod_err = Dict()
rel_prod_err = Dict()
min_abs_error_exp = -9
max_abs_error_exp = -10
for label in models
    println("evaluating: $label")
    for u in control_signals
        println("    control signal $(control_labels[u])")
        abs_prod_err[label,u] = Dict()
        rel_prod_err[label,u] = Dict()
        for T in T_range, p in p_range
            T_name = round(T, digits = 4)
            P_name = round(p, digits = 4)
            filename = "res_$(P_name)_$(T_name).jld2"
            full_prod = load(joinpath("full_model_results", filename), "product")
            for ω in ω_range
                ME, W, Λ, Winv, Bin, S = problem_data[T,p]
                sol, prod, model = results[T,p][label][u,ω]
                if sol.sol.retcode == :Success
                    if label == :cse
                        red_prod = [prod(t) for t in t_range[2:end]]
                    else
                        red_prod = [Float64.(evaluate_prod(BigFloat.(model.lift(sol(t),u(t,ω))),u,BigFloat(t),BigFloat(ω),W,Λ,Winv,Bin,ME.F)) for t in t_range[2:end]]
                    end
                    abs_err, rel_err = compare([c for c in full_prod[control_labels[u], ω][2:end]], 
                                               red_prod; singularity=singularity)
                    abs_prod_err[label, u][T,p,ω] = abs_err
                    rel_prod_err[label, u][T,p,ω] = rel_err
                    max_abs_error_exp = max(ceil(Int,log10(maximum(abs_prod_err[label, u][T,p,ω]))), max_abs_error_exp)
                    #min_abs_error_exp = min(ceil(Int,log10(minimum(abs_prod_err[label, u][T,p,ω]))), min_abs_error_exp)
                else
                    println("Model $label failed at (T,p) = ($T,$p) due to $(sol.sol.retcode)")
                end
            end
        end
    end
end

# absolute error
for u in control_signals
    fig = Figure(fontsize=24, resolution = (1200, 500))
    ax = Axis(fig[1,1], xlabel = L"\text{time } t", 
                        ylabel = L"\text{absolute error }  \Vert p(t) - \hat{p}(t) \Vert", 
                        yscale = log10, xscale=log10,
                        yticks = (10.0 .^ range(min_abs_error_exp,max_abs_error_exp), [latexstring("10 ^ {$e}") for e in range(min_abs_error_exp,max_abs_error_exp)]),
                        yminorticksvisible = true, yminorgridvisible = true,
                        yminorticks = IntervalsBetween(8),
                        xticks = (10.0 .^ range(min_time_exp, ceil(Int, log10(horizon))),
                                  [latexstring("10 ^ {$e}") for e in range(min_time_exp, ceil(Int, log10(horizon)))]),
                        xminorticksvisible = true, xminorgridvisible = true,
                        xminorticks = IntervalsBetween(8))
    xlims!(ax, 10.0 .^ min_time_exp, 10.0 .^ ceil(Int, log10(horizon)))
    plots = []
    plot_labels = String[]
    for label in models
        means = mean(abs_prod_err[label, u][key] for key in keys(abs_prod_err[label, u]))
        if !isnothing(color[label][2])
            p = scatterlines!(ax, t_range[2:end], [m > 10.0 ^ min_abs_error_exp ? m : missing for m in means], 
                                  color = color[label][1], 
                                  marker = color[label][2],
                                  markersize = [mod(i-1, 25) == 0 ? 15 : 0 for i in 1:length(t_range)-1],
                                  linewidth=2)
            push!(plots, p)
        else
            p = lines!(ax, t_range[2:end], [m > 10.0 ^ min_abs_error_exp ? m : missing for m in means], 
                           color = color[label][1], 
                           linewidth=2)
            push!(plots, p)
        end
        push!(plot_labels, labels[label])
    end
    Legend(fig[1,2], plots, plot_labels)
    save(joinpath(@__DIR__,"figures",control_labels[u]*"_abs_prod_comparison.pdf"), fig)
end

# relative error
for u in control_signals
    fig = Figure(fontsize=24, resolution = (1200, 500))
    ax = Axis(fig[1,1], xlabel = L"\text{time } t", 
                        ylabel = L"\text{relative error }  \frac{\Vert p(t) - \hat{p}(t) \Vert}{\Vert p(t) \Vert  }", 
                        yscale = log10, xscale=log10,
                        yticks = (10.0 .^ range(min_error,max_error), [latexstring("10 ^ {$e}") for e in range(min_error,max_error)]),
                        yminorticksvisible = true, yminorgridvisible = true,
                        yminorticks = IntervalsBetween(8),
                        xticks = (10.0 .^ range(min_time_exp, ceil(Int, log10(horizon))),
                                [latexstring("10 ^ {$e}") for e in range(min_time_exp, ceil(Int, log10(horizon)))]),
                        xminorticksvisible = true, xminorgridvisible = true,
                        xminorticks = IntervalsBetween(8))
    xlims!(ax, 10.0 .^ min_time_exp, 10.0 .^ ceil(Int, log10(horizon)))
    plots = []
    plot_labels = String[]
    for label in models
        means = mean(rel_prod_err[label, u][key] for key in keys(rel_prod_err[label, u]))
        if !isnothing(color[label][2])
            p = scatterlines!(ax, t_range[2:end], [m > 10.0 ^ min_error ? m : missing for m in means], 
                                  color = color[label][1], 
                                  marker = color[label][2],
                                  markersize = [mod(i-1, 25) == 0 ? 15 : 0 for i in 1:length(t_range)-1],
                                  linewidth=2)
            push!(plots, p)
        else
            p = lines!(ax, t_range[2:end], [m > 10.0 ^ min_error ? m : missing for m in means], 
                           color = color[label][1], 
                           linewidth=2)
            push!(plots, p)
        end
        push!(plot_labels, labels[label])
    end
    Legend(fig[1,2], plots, plot_labels)
    save(joinpath(@__DIR__,"figures",control_labels[u]*"_rel_prod_comparison.pdf"), fig)
end

#= test
fig = Figure()
ax = Axis(fig[1,1],xscale=log10, yscale=log10)
for T in T_range, p in p_range, ω in ω_range[1:4]
    T_name = round(T, digits = 4)
    P_name = round(p, digits = 4)
    filename = "res_$(P_name)_$(T_name).jld2"
    full_prod = load(joinpath("full_model_results", filename), "product")
    lines!(ax, t_range[2:end], [norm(c) for c in full_prod[control_labels[u_exp_increase], ω][2:end]])
end
fig

fig = Figure()
ax = Axis(fig[1,1],xscale=log10, yscale=log10)
for T in T_range, p in p_range, ω in ω_range[1:4]
    T_name = round(T, digits = 4)
    P_name = round(p, digits = 4)
    filename = "res_$(P_name)_$(T_name).jld2"
    full_prod = load(joinpath("full_model_results", filename), "concentration")
    lines!(ax, t_range[2:end], [norm(c) for c in full_prod[control_labels[u_exp_increase], ω][2:end]])
end
fig

fig = Figure()
ax = Axis(fig[1,1],xscale=log10, yscale=log10)
for T in T_range, p in p_range, ω in ω_range[1:4]
    T_name = round(T, digits = 4)
    P_name = round(p, digits = 4)
    filename = "res_$(P_name)_$(T_name).jld2"
    full_conc = load(joinpath("full_model_results", filename), "concentration")[control_labels[u_exp_increase], ω]
    ME, W, Λ, Winv, Bin, S = problem_data[T,p]  
    full_prod = [evaluate_prod(Dec128.(full_conc[i]), u_exp_increase, Dec128(t_range[i]), Dec128(ω), W, Λ, Winv, Bin, ME.F) for i in 2:length(t_range)]
    lines!(ax, t_range[2:end], [norm(c) > 1e-10 ? norm(c) : missing for c in full_prod])
end
fig
=#