mutable struct BalancedROM
    V
    N
    A
    B
    C
    D
    lift
end

function BalancedROM(T, Tinv, ME::MasterEquation, Bin, n; stationary_correction = true)
    return BalancedROM(T, Tinv, ME.M, Bin, n; F=ME.F, stationary_correction=stationary_correction)
end
function BalancedROM(T, Tinv, M::Matrix, Bin, n; F = zeros(0,size(M,1)), stationary_correction = true)
    T_D = T[1:n, :]
    T_T = T[n+1:end, :]
    Q_D = Tinv[:, 1:n]
    Q_T = Tinv[:, n+1:end]

    if stationary_correction
        A11 = T_D*M*Q_D
        A12 = T_D*M*Q_T
        A21 = T_T*M*Q_D
        A22 = T_T*M*Q_T
        B1 = T_D*Bin
        B2 = T_T*Bin

        aux = A22 \ A21 
        A = A11 - A12*aux 
        B = B1 - A12*(A22 \ B2)
        C = F*(Q_D - Q_T*aux)
        D = -F*Q_T*(A22\B2)
        lift = LiftingMap(Q_D - Q_T*aux, -Q_T*(A22 \ B2))
    else
        A = T_D*M*Q_D 
        B = T_D*Bin
        C = F*Q_D
        D = zeros(size(F,1), size(Bin,2))
        lift = LiftingMap(Q_D, zeros(size(Bin)))
    end
    return BalancedROM(Q_D, Q_T, A, B, C, D, lift)
end

# P must be controllability Gramian like, i.e.,
# If x -> Tz, then Tz -> T^-1 P T^-T
# P := \int exp(tA)B B^T exp(tA^T) dt 
# If dx/dt = Ax + Bu => dz/dt = T^-1 A Tz + T^-1 B u
# Thus,
# Pz := \int T^-1 exp(tA) T * T^-1 B B^T T^-T T^T exp(t A^T) T^-T dt 
#     = T^-1 \int exp(tA) B B^T exp(t A^T) dt T^-T

# Q must be observability Gramian like, i.e.,
# If x -> Tz, then Tz -> T P T^*
# Q := \int exp(tA^T) C^T C exp(tA) dt 
# If dx/dt = Ax + Bu => dz/dt = T^-1 A Tz + T^-1 B u
# Thus,
# Qz = \int T^T exp(tA^T) T^{-T} (C*T)^T C T T^-1 exp(tA) T dt 
#    = T^T Q T

function balancing(P, Q; rtol = 1e-8, atol = 1e-8)
    rP = rank(P)
    rQ = rank(Q)
    if rP == size(P,1)
        U, _ = cholesky(P)
    else
        ΛP, WP = eigen(Symmetric(P))
        rP = findfirst(x -> x < max(rtol*ΛP[1], atol), ΛP) - 1
        U = WP[:,1:rP]*Diagonal(sqrt.(ΛP[1:rP]))
    end
    if rQ == size(Q,1)
        Z, _ = cholesky(Q)
    else
        ΛQ, WQ = eigen(Symmetric(Q))
        rQ = findfirst(x -> x < max(rtol*ΛQ[1], atol), ΛP) -1
        Z = WQ[:,1:rQ]*Diagonal(sqrt.(ΛQ[1:rQ]))
    end
    #ΛP, U = eigen(Symmetric(P))
    #ΛQ, Z = eigen(Symmetric(Q))
    W, Σ, V = svd(U'*Z)
    n_eff = findfirst(x -> x < max(atol, rtol*Σ[1]), Σ) 
    if isnothing(n_eff)
        n_eff = length(Σ)
    else
        n_eff = n_eff - 1
    end
    Σ_pinv = Diagonal(1 ./ Σ[1:n_eff])
    T_bal = sqrt.(Σ_pinv)*V[:,1:n_eff]'*Z'
    T_bal_inv = U*W[:,1:n_eff]*sqrt.(Σ_pinv)
    return T_bal, T_bal_inv, Σ
end

function factor_balancing(U, Z; rtol = 1e-8, atol = 1e-8)
    W, Σ, V = svd(U'*Z)
    n_eff = findfirst(x -> x < max(atol, rtol*Σ[1]), Σ) 
    if isnothing(n_eff)
        n_eff = minimum(size(U))
    else
        n_eff = n_eff - 1
    end
    Σ_pinv = Diagonal(1 ./ Σ[1:n_eff])
    T_bal = sqrt.(Σ_pinv)*V[:,1:n_eff]'*Z'
    T_bal_inv = U*W[:,1:n_eff]*sqrt.(Σ_pinv)
    return T_bal, T_bal_inv, Σ
end

# add analytical equation for observability Gramian?
function ControllabilityGramian(ME, Bin)
    W, Λ, Winv = compute_diagonalization(ME)
    n = length(Λ)
    b = Winv*Bin
    C = -[b[i]*b[j]/(Λ[i]+Λ[j]) for i in 1:n, j in 1:n]
    return W, C, Winv
end

function Gramian(W, Λ, Winv, B)
    n = length(Λ)
    b = Winv*B
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

