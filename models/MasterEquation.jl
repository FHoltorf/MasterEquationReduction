using LinearAlgebra, BlockDiagonals, LaTeXStrings

mutable struct MasterEquation
    M
    Msym
    Z
    Zmat 
    K
    Kmat
    F 
    Fsym
    T
    b
    ρ
    E
    spec_idcs
    specs
    prods
    network
end

function symmetrize(x, ME::MasterEquation)
    return ME.T*x 
end

function unsymmetrize(z, ME::MasterEquation)
    return inv(ME.T)*z
end

function boltzmann(ρ, E)
    dist = @. ρ * exp(-E)
    return dist
end

function collision_operator(P,ω,b)
    Z = zeros(size(P))
    for i in 1:n, j in 1:i-1
        Z[i,j] = P[i,j]
        Z[j,i] = Z[i,j] * b[j]/b[i] # detailed balance
    end
    for i in 1:n
        Z[i,i] = -sum(Z[:,i])
    end
    return ω*Z
end

function kinetic_operator(k,specs,ρ,n)
    K = Dict((s_source, s_dest) => Diagonal(zeros(n)) for s_source in specs, s_dest in specs)
    for rxn in keys(k)
        s_source, s_dest = rxn
        K[s_dest, s_source].diag .= k[rxn]
        @. K[s_source, s_dest].diag = K[s_dest, s_source].diag * (ρ[s_source] / ρ[s_dest])  
    end
    # diagonals for conservation of material
    for spec in specs
        K[spec, spec].diag .= -sum(K[s,spec].diag for s in specs)
    end
    return K 
end

# without products
function RandomizedMasterEquation(specs, network, ρ, E, ω)
    n = length(E)
    b = Dict(spec => boltzmann(ρ[spec], E) for spec in specs)
    
    spec_idcs = [(k-1)*n+1:k*n for k in eachindex(specs)]
    
    # Collision operators 
    Z = Dict{eltype(specs), Matrix{Float64}}()
    for spec in specs
        Z[spec] = collision_operator(rand(n,n), ω, b[spec])
    end
    
    # Kinetic operator
    k = Dict(rxn => rand() * exp.(E) for rxn in network)
    K = kinetic_operator(k, specs, ρ, n)

    # Master Equation coefficient matrix
    Zmat = BlockDiagonal([Z[spec] for spec in specs])
    Kmat = zeros(0, length(specs)*n)
    for s_dest in specs
        row = reduce(hcat, [K[s_dest, s_source] for s_source in specs])
        Kmat = vcat(Kmat, row)
    end

    M = Zmat + Kmat

    # Symmetrize via
    # similarity transform z = Tx (x natural state, z transformed/symmetrized state)
    T = Diagonal(reduce(vcat, [1 ./sqrt.(b[spec]) for spec in specs])) 
    
    # return Master Equation Object
    return MasterEquation(M, Symmetric(T*M*inv(T)), Z, Zmat, K, Kmat, missing, missing, T, b, ρ, E, spec_idcs, specs, missing, network)
end

# with products
function RandomizedMasterEquation(specs, network, ρ, E, ω, products)
    ME = RandomizedMasterEquation(specs, network, ρ, E, ω)
    prods = []
    n = length(E)
    F = zeros(0,length(specs)*n)
    for spec in keys(products)
        i = findfirst(x->x==spec,specs)
        spec_range = n*(i-1)+1:n*i
        for prod in products[spec]
            Fspec = rand()#*exp.(E)    
            k = findfirst(x -> x == prod, prods)
            if isnothing(k)
                F = vcat(F, zeros(1, length(specs)*n))
                push!(prods, prod)
                k = size(F,1)
            end
            F[k, spec_range] .= Fspec
        end
    end
    # conservation:
    D = Diagonal(sum(F, dims=1)[1,:])
    ME.Msym -= D
    ME.M -= D

    ME.F = F
    ME.Fsym = F * inv(ME.T)
    ME.prods = prods

    return ME
end

function RandomizedBimolecularForcing(ME::MasterEquation, specs::AbstractVector, bA, bB)
    n = length(ME.E)
    B = zeros(n*length(ME.specs))
    for spec in specs
        i = findfirst(x->x == spec, ME.specs)
        spec_range = (i-1)*n+1:n*i
        B[spec_range] .= rand() * bA .* bB .* exp.(ME.E)
    end
    return B
end


function RandomizedBimolecularForcing(ME, spec, bA, bB)
    return RandomizedBimolecularForcing(ME, [spec], bA, bB)
end

function RandomizedBoltzmannDistribution(degeneracy_range, E)
    n = length(E)
    b = rand(degeneracy_range, n) .* exp.(E)
    return b ./ sum(b)
end

function test_symmetry(ME)
    test_mat = ME.T*ME.M*inv(ME.T)
    error = -Inf
    for i in 1:size(test_mat,1), j in i+1:size(test_mat,2)
        error = max(error, abs(test_mat[i,j]-test_mat[j,i]))
    end
    return error
end 

function plot_spectrum(ME)
    fig = Figure(fontsize = 24)
    ax = Axis(fig[1,1], ylabel = L"|\lambda|", yscale = log10)
    L = abs.(eigvals(ME.M, sortby=-))
    lines!(ax, 1:length(L), L, color = :black, linewidth= 2)
    display(fig)
    return fig
end

function MasterEquation(M, idcs, B, F, T, iso_labels = missing, prod_labels = missing)
    n_iso = size(idcs, 1)
    n_prod = size(F, 1)
    
    # reorder things properly
    iso_idcs = [Int.(idcs[k,findall(x -> x >= 0, idcs[k,:])] .+ 1) for k in axes(idcs,1)]  
    spec_idcs = [1:length(iso_idcs[1])]
    offset = length(iso_idcs[1])
    for k in 2:length(iso_idcs)
        push!(spec_idcs, offset+1:offset+length(iso_idcs[k]))
        offset += length(iso_idcs[k])
    end
    iso_order = reduce(vcat, iso_idcs)
    M_reordered = M[iso_order, iso_order]
    T_reordered = T[iso_order]
    B_reordered = B[iso_order, :]
    F_reordered = F[:,iso_order]
    M_reordered .-= Diagonal(vec(sum(F_reordered,dims=1)))

    M_sym = similar(M_reordered)
    F_sym = similar(F_reordered)
    for i in axes(M_reordered,1)
        for j in axes(M_reordered,2)
            scaling_factor = T_reordered[j]/T_reordered[i]
            M_sym[i,j] = scaling_factor*M_reordered[i,j]
        end
        F_sym[:,i] = F_reordered[:,i] * T_reordered[i]
    end
    
    if ismissing(iso_labels)
        b = Dict(k => sqrt.(T[iso]) for (k,iso) in enumerate(iso_idcs))
        isos = 1:n_iso
    else
        b = Dict(iso_labels[k] => sqrt.(T[iso]) for (k,iso) in enumerate(iso_idcs))
        isos = [iso_labels[k] for k in 1:n_iso]
    end

    if ismissing(prod_labels)
        prods = n_iso+1:n_iso+n_prod
    else
        prods = [prod_labels[k] for k in 1:n_prod]
    end

    network = []
    # isomerization reactions
    for k in 1:n_iso
        for j in k+1:n_iso
            if sum(M[iso_idcs[k], iso_idcs[j]]) > 0 
                push!(network, (isos[k],isos[j]))
            end
        end
    end

    # product formation
    for k in 1:n_iso
        for j in axes(F,1)
            if sum(F[j, iso_idcs[k]]) > 0 
                push!(network, isos[k] => prods[j])
            end
        end
    end


    return MasterEquation(M_reordered, 
                          M_sym, 
                          missing,
                          missing,
                          missing,
                          missing,
                          F_reordered,
                          F_sym,
                          inv(Diagonal(T_reordered)),
                          b,
                          missing,
                          missing, 
                          spec_idcs,
                          isos,
                          prods,
                          network), B_reordered
end


# general utilities
function master_equation!(dx, x, p, t)
    A, B, u = p 
    dx .= A*x + B*u(t)
end

struct LiftingMap{aType, bType}
    A::aType
    B::bType
end

function (lm::LiftingMap)(c)
    return lm.A*c
end

function (lm::LiftingMap)(c,u)
    return lm.A*c + lm.B*u
end
