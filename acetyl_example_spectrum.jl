cd(@__DIR__)
using Pkg
Pkg.activate(".")

using DelimitedFiles, DifferentialEquations, Sundials, CairoMakie, Statistics
include("models/MasterEquation.jl")
include("models/NaiveTruncation.jl")
include("models/PetrovGalerkinROM.jl")
include("models/CSEROM.jl")
include("models/CSCROM.jl")
include("models/BalancingROM.jl")
include("models/product_eval.jl")

path = "MEdata/"

# easy conditions
Temp = 700.0
pres = 1.0
scale = 1e3

M = readdlm(string(path, "M_", pres, "_", Temp, ".csv"), ',', Float64, header=false)
F = readdlm(string(path, "K_", pres, "_", Temp, ".csv"), ',', Float64, header=false)
B = readdlm(string(path, "B_", pres, "_", Temp, ".csv"), ',', Float64, header=false)
T = readdlm(string(path, "T_", pres, "_", Temp, ".csv"), ',', Float64, header=false)
idcs = readdlm(string(path, "M_", pres, "_", Temp, "idx.csv"), ',', Float64, header=false)

iso_labels = [:acetylperoxy, :hydroperoxylvinoxy]
product_labels = [:acetyl, :ketene]

ME, Bin = MasterEquation(M, idcs, B, F, T, iso_labels, product_labels)
Bin *= scale
W, Λ, Winv = compute_diagonalization(ME)

S = zeros(length(ME.specs), size(ME.M,2))
for i in eachindex(ME.specs)
    S[i,ME.spec_idcs[i]] .= 1
end

# bt model
t_int = 10 .^ range(-12, 3, length = length(Λ))
pushfirst!(t_int, 0.0)
C_factor = reduce(hcat, sqrt(t_int[i]-t_int[i-1])*W*Diagonal(exp.(t_int[i]*Λ))*Winv*Bin for i in 2:length(t_int))
O_factor = reduce(hcat, sqrt(t_int[i]-t_int[i-1])*Winv'*Diagonal(exp.(t_int[i]*Λ))*W'*S' for i in 2:length(t_int))
T, Tinv, Σ = factor_balancing(C_factor, O_factor; rtol = 1e-40, atol = 1e-14)

n = length(Λ)
time_scales = (Λ[1] ./ Λ[1:n])
fig = Figure(fontsize=28)
ax = Axis(fig[1,1], ylabel = "magnitude", 
                    xlabel = L"i",
                    yscale = log10, 
                    yticks = (10.0 .^ range(-12, 0), [latexstring("10 ^ {$e}") for e in range(-12,0)]),
                    yminorticksvisible = true, yminorgridvisible = true,
                    yminorticks = IntervalsBetween(8),)
ylims!(ax, 0.5e-12, 5.0)
scatter!(ax, 1:n, time_scales , color = :red, label = L"\text{scaled eigenvalues } \frac{\lambda_{1}}{\lambda_i}")
scatter!(ax, 1:n, Σ[1:n] ./ Σ[1,1], color = :blue, label = L"\text{scaled Hankel singular values } \frac{\sigma_{i}}{\sigma_1}")
#axislegend(ax, position = :rb)
fig
save("figures//spectrum_$(Temp)_$(pres).pdf", fig)

# hard conditions
Temp = 1400.0
pres = 1.0
scale = 1e3

M = readdlm(string(path, "M_", pres, "_", Temp, ".csv"), ',', Float64, header=false)
F = readdlm(string(path, "K_", pres, "_", Temp, ".csv"), ',', Float64, header=false)
B = readdlm(string(path, "B_", pres, "_", Temp, ".csv"), ',', Float64, header=false)
T = readdlm(string(path, "T_", pres, "_", Temp, ".csv"), ',', Float64, header=false)
idcs = readdlm(string(path, "M_", pres, "_", Temp, "idx.csv"), ',', Float64, header=false)

iso_labels = [:acetylperoxy, :hydroperoxylvinoxy]
product_labels = [:acetyl, :ketene]

ME, Bin = MasterEquation(M, idcs, B, F, T, iso_labels, product_labels)
Bin *= scale
W, Λ, Winv = compute_diagonalization(ME)

S = zeros(length(ME.specs), size(ME.M,2))
for i in eachindex(ME.specs)
    S[i,ME.spec_idcs[i]] .= 1
end

# bt model
t_int = 10 .^ range(-12, 3, length = length(Λ))
pushfirst!(t_int, 0.0)
C_factor = reduce(hcat, sqrt(t_int[i]-t_int[i-1])*W*Diagonal(exp.(t_int[i]*Λ))*Winv*Bin for i in 2:length(t_int))
O_factor = reduce(hcat, sqrt(t_int[i]-t_int[i-1])*Winv'*Diagonal(exp.(t_int[i]*Λ))*W'*S' for i in 2:length(t_int))
T, Tinv, Σ = factor_balancing(C_factor, O_factor; rtol = 1e-40, atol = 1e-14)

n = length(Λ)
time_scales = (Λ[1] ./ Λ[1:n])
fig = Figure(fontsize=28)
ax = Axis(fig[1,1], ylabel = "magnitude", 
                    yscale = log10, 
                    xlabel = L"i",
                    yticks = (10.0 .^ range(-12, 0), [latexstring("10 ^ {$e}") for e in range(-12,0)]),
                    yminorticksvisible = true, yminorgridvisible = true,
                    yminorticks = IntervalsBetween(8),)
ylims!(ax, 0.5e-12, 5.0)
scatter!(ax, 1:n, time_scales, color = :red, label = L"\text{scaled eigenvalues } \frac{\lambda_{1}}{\lambda_i}")
scatter!(ax, 1:n, Σ[1:n] ./ Σ[1,1], color = :blue, label = L"\text{scaled Hankel singular values } \frac{\sigma_{i}}{\sigma_1}")
axislegend(ax, position = :rb)
fig
save("figures//spectrum_$(Temp)_$(pres).pdf", fig)
