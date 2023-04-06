cd(@__DIR__)
using Pkg
cd(@__DIR__)
Pkg.activate(".")

using DelimitedFiles, DifferentialEquations, FileIO, PrettyTables
include("models/MasterEquation.jl")
include("models/CSEROM.jl")
include("models/BalancingROM.jl")
path = "MEdata//data_pieces_no_reverse/"

# easy conditions
io = open("phenomenological_coeffs.txt","w")
io_tex = open("phenomenological_coeffs.tex","w")
T_range = 1000.0:100.0:1500.0
for Temp in T_range
    pres = 1.0
    scale = 1e3

    M = readdlm(string(path, "M_", pres, "_", Temp, ".csv"), ',', Float64, header=false)
    F = readdlm(string(path, "K_", pres, "_", Temp, ".csv"), ',', Float64, header=false)
    B = readdlm(string(path, "B_", pres, "_", Temp, ".csv"), ',', Float64, header=false)
    T = readdlm(string(path, "T_", pres, "_", Temp, ".csv"), ',', Float64, header=false)
    idcs = readdlm(string(path, "M_", pres, "_", Temp, "idx.csv"), ',', Float64, header=false)

    iso_labels = [:acetylperoxy, :hydroperoxylvinoxy]
    product_labels = [:acetyl, :ketene]
    coeff_labels = []
    for i in 1:2, j in 1:2
        if i != j 
            push!(coeff_labels, String(iso_labels[j]) => String(iso_labels[i]))
        end
    end
    for i in 1:2
        push!(coeff_labels, String(iso_labels[i]) => String(iso_labels[i]))
    end
    for i in 1:2, j in 1:2
        push!(coeff_labels, String(iso_labels[j]) => String(product_labels[i]))
    end
    for i in 1:2
        push!(coeff_labels, "O2+acetyl" => String(iso_labels[i]))
    end
    for i in 1:2
        push!(coeff_labels, String(iso_labels[i]) => "")
    end
    
    function extract_coeffs(model)
        data = []
        for i in 1:2, j in 1:2 
            if i != j
                push!(data, model.A[i,j])
            end
        end
        for i in 1:2, j in 1:2 
            push!(data, model.C[i,j])
        end
        for i in 1:2
            push!(data, model.B[i,1])
        end
        for i in 1:2
            push!(data, model.D[i,1])
        end
        for i in 1:2
            push!(data, model.A[i,i])
        end
        return data
    end

    ME, Bin = MasterEquation(M, idcs, B, F, T, iso_labels, product_labels)
    Bin *= scale
    W, Λ, Winv = compute_diagonalization(ME)

    # cse georgievskii model
    cse_rom = CSEModel_Geo(ME, Bin, stationary_correction=false)

    # dst model
    dst_rom = CSEModel(ME, Bin, stationary_correction=true)

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
    
    bt_rom = BalancedROM(T, Tinv, W*Diagonal(Λ)*Winv, Bin, 2; F = ME.F, stationary_correction=false)
    # apply similarity transform
    L = S*Tinv[:, 1:2] # c = L*z => dc/dt = L*dz/dt = L*A*L^-1 z + L*B*u
    bt_rom.C = ME.F * W * Diagonal(1 ./ Λ) * Winv * Tinv[:,1:2]*bt_rom.A*inv(L)
    bt_rom.D = ME.F * W * Diagonal(1 ./ Λ) * Winv * (Tinv[:,1:2]*T[1:2,:] - I)*Bin
    bt_rom.A = L*bt_rom.A*inv(L)
    bt_rom.B = L*bt_rom.B

    data = hcat([c[1] for c in coeff_labels], [c[2] for c in coeff_labels], reduce(hcat, extract_coeffs(model) for model in [cse_rom, dst_rom, bt_rom]))
    pretty_table(io, data, title = "T = $(Temp)K, p = $(pres)bar",  header = ["reactant", "product", "CSE", "DST 2(P)", "BT 2(P)"])#, backend = Val(:latex))
    println(io, "\n\n\n")

    si_formatter = LatexHighlighter((data,i,j) -> i >= 1 && j > 2, ["niceformat"])
    println(io_tex,"\\begin{table}\n\\centering\n\\caption{Phenomenological rate coefficients predicted at \$ \\SI{$Temp}{\\kelvin}\$ and \$\\SI{$pres}{\\bar}\$.}")
    pretty_table(io_tex, data, title = "T = $(Temp)K, p = $(pres)bar",  
                           header = (["reactants", "product", "CSE", "DST 2(P)", "BT 2(P)"],
                                     ["", "", "[1/s]", "[1/s]", "[1/s]"]),
                           backend = Val(:latex),
                           hlines = [0,1,size(data,1)-1,size(data,1)+1],
                           highlighters = (si_formatter), 
                           tf=tf_latex_modern)
    println(io_tex,"\\end{table}")
    println(io_tex,"\n\n\n")
end
close(io)
close(io_tex)
