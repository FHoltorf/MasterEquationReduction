using NMF, CairoMakie

mutable struct CSCROM
    V
    N
    A
    B
    C
    D
    q
end

function normalize_modes(V,H)
    scale = sum(V, dims = 1)
    return V ./ scale, scale' .* H
end

function CSCROM(model::PetrovGalerkinROM)
    return CSCROM(model.V,model.N,model.A,model.B,model.C,model.D,model.q)
end

function non_negative_reduction(data, n_modes; maxiter = 1000)
    res = nnmf(data, n_modes, maxiter = maxiter)
    println("Relative error = $(norm(data - res.W*res.H)/norm(data))")
    V, H = normalize_modes(res.W, res.H)
    N = nullspace(V')
    return V, H, N
end

function visualize_CSCs(V,n,m; specs = nothing)
    fig = Figure(fontsize=24)
    axs = [Axis(fig[i,j], title = "CSE $(i+(j-1)*m)",
                xminorgridvisible=false,
                xgridvisible=false,
                ygridvisible=false,
                yminorgridvisible=false) for i in 1:n, j in 1:m]

    if !isnothing(specs)
        n = length(specs)
        step = 1.0/n
        label_xpos = range(step/2, 1-step/2, length = n)
        vbar_xpos = size(V,1)/n
    end
    for i in 1:n, j in 1:m
        barplot!(axs[i,j], V[:,i+(j-1)*n], color= :dodgerblue)
        if !isnothing(specs)
            for (k,spec) in enumerate(specs)
                vlines!(axs[i,j], [k*vbar_xpos], linestyle=:dash, color = :black)   
                text!(axs[i,j], string(spec), space=:relative, align=(:center, :center), position=Point2f(label_xpos[k], 0.9))     
            end
        end
    end
    display(fig)
    return fig
end

# function compute_overlap(supports)
#     n = size(supports,2)
#     overlap = Dict()
#     for i in 1:n
#         for j in i+1:n
#             overlap[i,j] = [supports[k,i] > 0 && supports[k,j] > 0 for k in axes(supports, 1)]
#         end
#         overlap[j,i] = overlap[i,j]
#     end
#     return overlap
# end

function physical_reduction(V, ME; tol = 1e-8)
    weighted_support = (1 ./ sum(V, dims = 2)) .* V 
    support = V .> tol

    K_red = weighted_support'*ME.Kmat*V
    Z_red = support'*ME.Zmat*V
    F_red = ME.F*V
    D = Diagonal(sum(F_red, dims = 1)[1,:])
    M_red = K_red + Z_red - D
    return CSCROM(V, nothing, M_red, nothing, F_red, nothing, nothing)
end

function physical_reduction(V, ME, B; tol = 1e-8)
    weighted_support = (1 ./ sum(V, dims = 2)) .* V 
    support = V .> tol

    K_red = weighted_support'*(ME.Kmat - Diagonal(ME.Kmat))'*V
    B_red = weighted_support'*B
    Z_red = support'*(ME.Zmat - Diagonal(ME.Zmat))*V
    F_red = ME.F*V
    D = Diagonal(sum(F_red, dims = 1)[1,:] + sum(K_red + Z_red, dims = 1)[1,:])
    M_red = K_red + Z_red - D
    return CSCROM(V, nothing, M_red, B_red, F_red, zeros(size(F_red,1), size(B,2)), nothing)
end