using BlockDiagonals

mutable struct BlockTruncationROM
    ME
    blockED
    Λ
    V
    Vinv
    importance_ranking    
    A
    B
    C
    D
end

function build_BlockTruncationROM(ME::MasterEquation, N_trunc)
    blockED = Dict(spec => eigen(ME.Z[spec], sortby=-) for spec in specs)
    V_full = BlockDiagonal([blockED[spec].vectors for spec in specs])
    V_full_inv = BlockDiagonal([inv(blockED[spec].vectors) for spec in specs])
    Λ = reduce(vcat, blockED[spec].values for spec in specs)
    importance_ranking = sortperm(Λ, by = -)

    Λ_ranked = Λ[importance_ranking]
    dominant_modes = importance_ranking[1:N_trunc]
    truncated_modes = importance_ranking[N_trunc+1:end] 
    V = Matrix(V_full)[:,importance_ranking[dominant_modes]]
    Vinv = Matrix(V_full_inv)[importance_ranking[truncated_modes], :]

    K = copy(ME.Kmat)
    D = Diagonal(sum(ME.F, dims=1)[1,:])
    K -= D

    A = Diagonal(Λ_ranked[1:N_trunc]) + Vinv*K*V  
    B = missing
    C = ME.F*V
    D = missing
    return BlockTruncationROM(ME, blockED, Λ, V, Vinv, importance_ranking, A, B, C, D)
end

function build_BlockTruncationROM(ME::MasterEquation, Bin, N_trunc; stationary_correction = true)
    blockED = Dict(spec => eigen(ME.Z[spec], sortby=-) for spec in specs)
    V_full = BlockDiagonal([blockED[spec].vectors for spec in specs])
    V_full_inv = BlockDiagonal([inv(blockED[spec].vectors) for spec in specs])
    Λ = reduce(vcat, blockED[spec].values for spec in specs)
    importance_ranking = sortperm(Λ, by = -)

    Λ_ranked = Λ[importance_ranking]
    dominant_modes = importance_ranking[1:N_trunc]
    truncated_modes = importance_ranking[N_trunc+1:end] 
    Vdom = Matrix(V_full)[:,importance_ranking[dominant_modes]]
    Vdom_inv = Matrix(V_full_inv)[importance_ranking[dominant_modes], :]
    Vtrunc = Matrix(V_full)[:,importance_ranking[truncated_modes]]
    Vtrunc_inv = Matrix(V_full_inv)[importance_ranking[truncated_modes], :]
    Λdom = Diagonal(Λ[1:N_trunc])
    Λtrunc = Diagonal(Λ[N_trunc+1:end])

    K = copy(ME.Kmat)
    D = Diagonal(sum(ME.F, dims=1)[1,:])
    K -= D
    if stationary_correction
        Q = inv(Λtrunc + Vtrunc_inv*K*Vtrunc)
        aux = Vdom_inv*K*Vtrunc*Q
        A = (Λdom + Vdom_inv*K*Vdom - aux*Vtrunc_inv*K*Vdom)
        B = (Vdom_inv - aux*Vtrunc_inv)*Bin
        C = ME.F*(I - Vtrunc*Q*Vtrunc_inv*K)*Vdom
        D = -ME.F*Vtrunc*Q*Vtrunc_inv*Bin
    else
        A = Diagonal(Λ_ranked[1:N_trunc]) + Vdom_inv*K*Vdom  
        B = Vdom_inv*Bin
        C = ME.F*Vdom
        D = zeros(size(ME.F,1), size(Bin, 2))
    end
    return BlockTruncationROM(ME, blockED, Λ, Matrix(V_full), Matrix(V_full_inv), importance_ranking, A, B, C, D)
end