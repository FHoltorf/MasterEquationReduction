cd(@__DIR__)
using Pkg
Pkg.activate(".")

using DelimitedFiles, DifferentialEquations, Sundials, CairoMakie, Statistics, FileIO
include("models/MasterEquation.jl")
include("models/PetrovGalerkinROM.jl")
include("models/CSEROM.jl")
include("models/BalancingROM.jl")

data_path = "MEdata/"

T_range = range(1000, 1500, step=50) 
p_range = 10 .^ range(-2, 0, length=3) 

u_exp_decay(t, ω) = [ω * exp(-ω*t)]
u_exp_increase(t, ω) = [(1-exp(-ω*t))]

# computational experiments
control_signals = [u_exp_increase] 
control_labels = Dict(u_exp_decay => "exp_decay",
                      u_exp_increase => "exp_increase")

function evaluate_prod(c,u,t,ω,W,Λ,Winv,Bin,F)
    if u == u_exp_decay
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
    T_name = round(Temp, digits=4)
    P_name = round(Pres, digits=4)
    println("Temperature: $(Temp)K -- Pressure: $(Pres)bar")
    M = readdlm(string(data_path, "M_", P_name, "_", T_name, ".csv"), ',', Float64, header=false)
    F = readdlm(string(data_path, "K_", P_name, "_", T_name, ".csv"), ',', Float64, header=false)
    B = readdlm(string(data_path, "B_", P_name, "_", T_name, ".csv"), ',', Float64, header=false)
    T = readdlm(string(data_path, "T_", P_name, "_", T_name, ".csv"), ',', Float64, header=false)
    idcs = readdlm(string(data_path, "M_", P_name, "_", T_name, "idx.csv"), ',', Float64, header=false)

    iso_labels = [:acetylperoxy, :hydroperoxylvinoxy]
    product_labels = [:acetyl, :ketene]

    ME, Bin = MasterEquation(M, idcs, B, F, T, iso_labels, product_labels)
    Bin *= scale
    W, Λ, Winv = compute_diagonalization(ME)
    println("Largest time scale $(-1/Λ[1])")

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
    dst_pheno_rom = CSEModel(ME, Bin, stationary_correction=true)

    # bt model
    t_int = 10 .^ range(-12, 3, length = 500)
    pushfirst!(t_int, 0.0)
    S = zeros(length(ME.specs), size(ME.M,2))
    for i in eachindex(ME.specs)
        S[i,ME.spec_idcs[i]] .= 1
    end
    C_factor = reduce(hcat, sqrt(t_int[i]-t_int[i-1])*W*Diagonal(exp.(t_int[i]*Λ))*Winv*Bin for i in 2:length(t_int))
    O_factor = reduce(hcat, sqrt(t_int[i]-t_int[i-1])*Winv'*Diagonal(exp.(t_int[i]*Λ))*W'*S' for i in 2:length(t_int))
    T, Tinv, Σ = factor_balancing(C_factor, O_factor; rtol = 1e-40, atol = 1e-16)
    
    bt_roms = Dict()
    for n_modes in n_mode_range
        bt_rom = BalancedROM(T, Tinv, W*Diagonal(Λ)*Winv, Bin, n_modes; F = ME.F, stationary_correction=false)
        bt_roms[Symbol("bt_$n_modes")] = bt_rom
    end
    bt_pheno_rom = BalancedROM(T, Tinv, W*Diagonal(Λ)*Winv, Bin, 2; F = ME.F, stationary_correction=false)
    L = S*Tinv[:, 1:2] # c = L*z => dc/dt = L*dz/dt = L*A*L^-1 z + L*B*u
    bt_pheno_rom.C = ME.F * W * Diagonal(1 ./ Λ) * Winv * Tinv[:,1:2]*bt_pheno_rom.A*inv(L)
    bt_pheno_rom.D = ME.F * W * Diagonal(1 ./ Λ) * Winv * (Tinv[:,1:2]*T[1:2,:] - I)*Bin
    bt_pheno_rom.A = L*bt_pheno_rom.A*inv(L)
    bt_pheno_rom.B = L*bt_pheno_rom.B
    bt_pheno_rom.lift.A .= bt_pheno_rom.lift.A*inv(L)
    
    # models to consider
    models = [cse_rom => :cse,
              dst_pheno_rom => :dst_pheno,
              bt_pheno_rom => :bt_pheno,
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
                red_sol = solve(prob, CVODE_BDF(), dtmin=1e-24, maxiters=Int(1e6),#maxiters=Int(1e7), 
                                      reltol = (label == :bt_pheno ? 1e-10 : 1e-14), abstol=1e-14,
                                      saveat = t_range) 
                sol = ReducedSol(red_sol, 1:n, red_sol.t)
                prod = ReducedSol(red_sol, n+1:n+m, red_sol.t)
                solutions[label][u, ω] = (sol, prod, model) 
                if red_sol.retcode != :Success
                    println("$label did not coverge within tolerance.")
                end
            end
        end
    end
    results[temperature,pressure] = solutions
    print("\n")
end

include("acetyl_example_full_sols.jl")

# full/analytical solution
function compare(full_sol, red_sol; singularity = 0.0)
    abs_error = [norm(full_sol[i] - red_sol[i]) for i in eachindex(full_sol)]
    rel_error = abs_error ./ (singularity .+ norm.(full_sol))
    return abs_error, rel_error
end

# visualization
models = [:cse,
          :dst_pheno,
          :bt_pheno,
          [Symbol("dst_$n_mode") for n_mode in n_mode_range]...,
          [Symbol("bt_$n_mode") for n_mode in n_mode_range]...]

labels = Dict(:cse => "CSE", 
              :bt_pheno => "BT 2(P)",
              :dst_pheno => "DST 2(P)",
              [Symbol("dst_$n_mode") => "DST $n_mode" for n_mode in n_mode_range]..., 
              [Symbol("bt_$n_mode") => "BT $n_mode" for n_mode in n_mode_range]...)

markers = [:circle, :dtriangle, :rect, :diamond, :hexagon, :xcross, :star8]
color = Dict(:cse => (:black, nothing, nothing, 3), 
             :dst_pheno => (:red, nothing, :dash, 3),
             :bt_pheno => (:blue, nothing, :dash, 3),
             #:cse_corr => (:black, nothing, :dash),
             [Symbol("dst_$n_mode") => (:red, markers[i], nothing, 2) for (i,n_mode) in enumerate(n_mode_range)]...,
             [Symbol("bt_$n_mode") => (:blue, markers[i], nothing, 2) for (i,n_mode) in enumerate(n_mode_range)]...)

min_error = -9
max_error = 10
# interesting observables/comparisons
singularity = 0.0 # => regularization for numerical accuracy

function plot_means(t_range, error, u;
                    xlabel = L"\text{error}", 
                    ylabel = L"\text{time}",
                    min_error = min_error,
                    color = color,
                    models = models,
                    control_signals=control_signals,
                    resolution = (1200,500),
                    fontsize = 24,
                    yticks = (10.0 .^ range(min_error,max_error), [latexstring("10 ^ {$e}") for e in range(min_error,max_error)]),
                    xticks = (10.0 .^ range(min_time_exp, ceil(Int, log10(horizon))), [latexstring("10 ^ {$e}") for e in range(min_time_exp, ceil(Int, log10(horizon)))]))

    fig = Figure(fontsize=fontsize, resolution = resolution)
    ax = Axis(fig[1,1], xlabel = xlabel, 
                        ylabel = ylabel, 
                        yscale = log10, xscale=log10,
                        yticks = yticks,
                        yminorticksvisible = true, yminorgridvisible = true,
                        yminorticks = IntervalsBetween(8),
                        xticks = xticks,
                        xminorticksvisible = true, xminorgridvisible = true,
                        xminorticks = IntervalsBetween(8))
    xlims!(ax, xticks[1][1], xticks[1][end]) #10.0 .^ min_time_exp, 10.0 .^ ceil(Int, log10(horizon)))
    plots = []
    plot_labels = String[]
    for label in models
        means = mean(error[label, u][key] for key in keys(error[label, u]))
        if !isnothing(color[label][2])
            p = scatterlines!(ax, t_range[2:end], [m > 10.0 ^ min_error ? m : missing for m in means], 
                                    color = color[label][1], 
                                    marker = color[label][2],
                                    markersize = [mod(i-1, 25) == 0 ? 15 : 0 for i in 1:length(t_range)-1],
                                    linewidth=color[label][4])
            push!(plots, p)
        else
            p = lines!(ax, t_range[2:end], [m > 10.0 ^ min_error ? m : missing for m in means], 
                            color = color[label][1], linestyle=color[label][3],
                            linewidth=color[label][4])
            push!(plots, p)
        end
        push!(plot_labels, labels[label])
    end
    Legend(fig[1,2], plots, plot_labels)
    return fig
end

function plot_means_stds(t_range, error, u;
                    xlabel = L"\text{error}", 
                    ylabel = L"\text{time}",
                    min_error = min_error,
                    color = color,
                    models = models,
                    control_signals=control_signals,
                    resolution = (1200,500),
                    fontsize = 24,
                    yticks = (10.0 .^ range(min_error,max_error), [latexstring("10 ^ {$e}") for e in range(min_error,max_error)]),
                    xticks = (10.0 .^ range(min_time_exp, ceil(Int, log10(horizon))), [latexstring("10 ^ {$e}") for e in range(min_time_exp, ceil(Int, log10(horizon)))]))

    fig = Figure(fontsize=fontsize, resolution = resolution)
    ax_mean = Axis(fig[1,1], title = "means",
                        xlabel = xlabel, 
                        ylabel = ylabel, 
                        yscale = log10, xscale=log10,
                        yticks = yticks,
                        yminorticksvisible = true, yminorgridvisible = true,
                        yminorticks = IntervalsBetween(8),
                        xticks = xticks,
                        xminorticksvisible = true, xminorgridvisible = true,
                        xminorticks = IntervalsBetween(8))
    xlims!(ax_mean, xticks[1][1], xticks[1][end]) 

    ax_std = Axis(fig[1,2], title = "standard deviations",
                            xlabel = xlabel, 
                            yscale = log10, xscale=log10,
                            yticks = (yticks[1], ["" for i in eachindex(yticks[1])]),
                            yminorticksvisible = true, yminorgridvisible = true,
                            yminorticks = IntervalsBetween(8),
                            xticks = xticks,
                            xminorticksvisible = true, xminorgridvisible = true,
                            xminorticks = IntervalsBetween(8))
    xlims!(ax_std, xticks[1][1], xticks[1][end]) 

    plots = []
    plot_labels = String[]
    for label in models
        means = mean(error[label, u][key] for key in keys(error[label, u]))
        stds = [std(error[label, u][key][i] for key in keys(error[label, u])) for i in 1:length(t_range)-1]
        if !isnothing(color[label][2])
            p = scatterlines!(ax_mean, t_range[2:end], [m > 10.0 ^ min_error ? m : missing for m in means], 
                                    color = color[label][1], 
                                    marker = color[label][2],
                                    markersize = [mod(i-1, 25) == 0 ? 15 : 0 for i in 1:length(t_range)-1],
                                    linewidth=color[label][4])
            scatterlines!(ax_std, t_range[2:end], [s > 10.0 ^ min_error ? s : missing for s in stds], 
                                    color = color[label][1], 
                                    marker = color[label][2],
                                    markersize = [mod(i-1, 25) == 0 ? 15 : 0 for i in 1:length(t_range)-1],
                                    linewidth=color[label][4])
            push!(plots, p)
        else
            p = lines!(ax_mean, t_range[2:end], [m > 10.0 ^ min_error ? m : missing for m in means], 
                            color = color[label][1], linestyle=color[label][3],
                            linewidth=color[label][4])
            lines!(ax_std, t_range[2:end], [s > 10.0 ^ min_error ? s : missing for s in stds], 
                            color = color[label][1], linestyle=color[label][3],
                            linewidth=color[label][4])
            push!(plots, p)
        end
        push!(plot_labels, labels[label])
    end
    Legend(fig[1,3], plots, plot_labels)
    return fig
end

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
                    if label in [:cse, :dst_pheno, :bt_pheno] 
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

# relative error
u = control_signals[1]
fig_rel_macro_err = plot_means(t_range, rel_macro_err, u,
                            xlabel = L"\text{time } t", 
                            ylabel = L"\text{relative error }  \frac{\Vert Sc(t) - S\hat{c}(t) \Vert}{\Vert S c(t) \Vert}")
save(joinpath(@__DIR__,"figures",control_labels[u]*"_rel_macro_comparison.pdf"), fig_rel_macro_err)

fig_abs_macro_err = plot_means(t_range, abs_macro_err, u,
                                xlabel = L"\text{time } t", 
                                ylabel = L"\text{absolute error }  \Vert Sc(t) - S\hat{c}(t) \Vert")
save(joinpath(@__DIR__,"figures",control_labels[u]*"_abs_macro_comparison.pdf"), fig_abs_macro_err)

xticks = (10.0 .^ range(min_time_exp, ceil(Int, log10(horizon))), [iseven(e) ? latexstring("10 ^ {$e}") : "" for e in range(min_time_exp, ceil(Int, log10(horizon)))])
fig_rel_macro_err_vars = plot_means_stds(t_range, rel_macro_err, control_signals[1],
                                resolution=(1200,500),
                                xticks=xticks, 
                                xlabel = L"\text{time } t", 
                                ylabel = L"\text{relative error }  \frac{\Vert Sc(t) - S\hat{c}(t) \Vert}{\Vert S c(t) \Vert}")
save(joinpath(@__DIR__,"figures",control_labels[u]*"_rel_macro_comparison_vars.pdf"), fig_rel_macro_err_vars)


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

u = control_signals[1]
fig_rel_micro_err = plot_means(t_range, rel_micro_err, u,
                            xlabel = L"\text{time } t", 
                            ylabel = L"\text{relative error }  \frac{\Vert c(t) - \hat{c}(t) \Vert}{\Vert c(t) \Vert}")
save(joinpath(@__DIR__,"figures",control_labels[u]*"_rel_micro_comparison.pdf"), fig_rel_micro_err)

fig_abs_micro_err = plot_means(t_range, abs_micro_err, u,
                                xlabel = L"\text{time } t", 
                                ylabel = L"\text{absolute error }  \Vert c(t) - \hat{c}(t) \Vert")
save(joinpath(@__DIR__,"figures",control_labels[u]*"_abs_micro_comparison.pdf"), fig_abs_micro_err)

xticks = (10.0 .^ range(min_time_exp, ceil(Int, log10(horizon))), [iseven(e) ? latexstring("10 ^ {$e}") : "" for e in range(min_time_exp, ceil(Int, log10(horizon)))])
fig_rel_micro_err_vars = plot_means_stds(t_range, rel_micro_err, control_signals[1],
                                resolution=(1200,500),
                                xticks=xticks, 
                                xlabel = L"\text{time } t", 
                                ylabel = L"\text{relative error }  \frac{\Vert c(t) - \hat{c}(t) \Vert}{\Vert c(t) \Vert}")
save(joinpath(@__DIR__,"figures",control_labels[u]*"_rel_micro_comparison_vars.pdf"), fig_rel_micro_err_vars)

# error in p over time 
# 3 plots in a row, one for each observable, trace for each model
# -> mean over (T,p,ω) (or any other statstic?)
abs_prod_err = Dict()
rel_prod_err = Dict()
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
                    if label in [:cse, :bt_pheno_rom, :dst_pheno_rom]
                        red_prod = [prod(t) for t in t_range[2:end]]
                    else
                        red_prod = [Float64.(evaluate_prod(BigFloat.(model.lift(sol(t),u(t,ω))),u,BigFloat(t),BigFloat(ω),W,Λ,Winv,Bin,ME.F)) for t in t_range[2:end]]
                    end
                    abs_err, rel_err = compare([c for c in full_prod[control_labels[u], ω][2:end]], 
                                               red_prod; singularity=singularity)
                    abs_prod_err[label, u][T,p,ω] = abs_err
                    rel_prod_err[label, u][T,p,ω] = rel_err
                    max_abs_error_exp = max(ceil(Int,log10(maximum(abs_prod_err[label, u][T,p,ω]))), max_abs_error_exp)
                else
                    println("Model $label failed at (T,p) = ($T,$p) due to $(sol.sol.retcode)")
                end
            end
        end
    end
end

u = control_signals[1]
fig_rel_prod_err = plot_means(t_range, rel_prod_err, u,
xlabel = L"\text{time } t", 
ylabel = L"\text{relative error }  \frac{\Vert p(t) - \hat{p}(t) \Vert}{\Vert p(t) \Vert}")
save(joinpath(@__DIR__,"figures",control_labels[u]*"_rel_prod_comparison.pdf"), fig_rel_prod_err)

fig_abs_prod_err = plot_means(t_range, abs_prod_err, u,
xlabel = L"\text{time } t", 
ylabel = L"\text{absolute error }  \Vert p(t) - \hat{p}(t) \Vert")
save(joinpath(@__DIR__,"figures",control_labels[u]*"_abs_prod_comparison.pdf"), fig_abs_prod_err)

xticks = (10.0 .^ range(min_time_exp, ceil(Int, log10(horizon))), [iseven(e) ? latexstring("10 ^ {$e}") : "" for e in range(min_time_exp, ceil(Int, log10(horizon)))])
fig_rel_prod_err_vars = plot_means_stds(t_range, rel_prod_err, control_signals[1],
                                resolution=(1200,500),
                                xticks=xticks, 
                                xlabel = L"\text{time } t", 
                                ylabel = L"\text{relative error }  \frac{\Vert p(t) - \hat{p}(t) \Vert}{\Vert p(t) \Vert}")
save(joinpath(@__DIR__,"figures",control_labels[u]*"_rel_prod_comparison_vars.pdf"), fig_rel_prod_err_vars)
