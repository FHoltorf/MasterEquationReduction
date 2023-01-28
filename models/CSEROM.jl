mutable struct CSEModel{N,aType,bType,cType,dType,lType}
    ME::MasterEquation
    W::Matrix{N} # orthogonal
    Λ::Vector{N}
    A::aType
    B::bType
    C::cType
    D::dType
    lift::lType
end

function compute_diagonalization(ME)
    Λ, Wsym = eigen(ME.Msym, sortby=-)
    W = inv(ME.T)*Wsym
    Winv = Wsym'*ME.T
    
    return W, Λ, Winv
end

function CSEModel(ME::MasterEquation)#, n_modes = length(ME.specs))
    n_specs = length(ME.specs)
    
    W, Λ, Winv = compute_diagonalization(ME)
    Λ_dom = Diagonal(Λ[1:n_specs])
    W_dom = W[:, 1:n_specs]
    
    S = zeros(n_specs, sum(length.(ME.spec_idcs)))
    for i in 1:n_specs
        S[i, ME.spec_idcs[i]] = ones(length(ME.spec_idcs[i]))
    end
    
    P = S*W_dom
    Pinv = inv(P)    

    A = P * Λ_dom * Pinv
    C = ME.F * W_dom * Pinv 
    return CSEModel(ME, W, Λ, A, missing, C, missing) 
end

# with forcing 
function CSEModel(ME::MasterEquation, Bin; stationary_correction = true)
    n_specs = length(ME.specs)
    
    dom_subspace = 1:n_specs
    trunc_subspace = n_specs+1:size(ME.M,1)

    W, Λ, Winv = compute_diagonalization(ME)
    Λ_dom = Diagonal(Λ[dom_subspace])
    W_dom = W[:, dom_subspace]
    W_dom_inv = Winv[dom_subspace, :]
    Λ_trunc = Diagonal(Λ[trunc_subspace])
    W_trunc = W[:, trunc_subspace]
    W_trunc_inv = Winv[trunc_subspace, :]
    
    S = zeros(n_specs, sum(length.(ME.spec_idcs)))
    for i in 1:n_specs
        S[i, ME.spec_idcs[i]] = ones(length(ME.spec_idcs[i]))
    end
    
    P = S*W_dom
    Q = S*W_trunc
    Pinv = inv(P)
    
    
    B_trunc = W_trunc_inv * Bin
    B_dom = W_dom_inv * Bin

    A = P * Diagonal(Λ_dom) * Pinv
    B = P * B_dom 
    C = ME.F * W_dom * Pinv 
    if stationary_correction
        B += P * Λ_dom * Pinv * Q * inv(Λ_trunc) * B_trunc
        D = ME.F * (W_dom * Pinv * Q - W_trunc) * inv(Λ_trunc) * B_trunc
        lift = LiftingMap(W_dom*Pinv, W_dom*Pinv*Q*inv(Λ_trunc)*B_trunc - W_trunc*inv(Λ_trunc)*B_trunc)
    else
        D = zeros(size(ME.F, 1), size(Bin,2))
        lift = LiftingMap(W_dom*Pinv, zeros(size(Bin)))
    end

    return CSEModel(ME, W, Λ, A, B, C, D, lift) 
end


function CSEModel_consistent(ME::MasterEquation, Bin; stationary_correction = true)
    n_specs = length(ME.specs)
    
    dom_subspace = 1:n_specs
    trunc_subspace = n_specs+1:size(ME.M,1)

    W, Λ, Winv = compute_diagonalization(ME)
    Λ_dom = Diagonal(Λ[dom_subspace])
    W_dom = W[:, dom_subspace]
    W_dom_inv = Winv[dom_subspace, :]
    Λ_trunc = Diagonal(Λ[trunc_subspace])
    W_trunc = W[:, trunc_subspace]
    W_trunc_inv = Winv[trunc_subspace, :]
    
    S = zeros(n_specs, sum(length.(ME.spec_idcs)))
    for i in 1:n_specs
        S[i, ME.spec_idcs[i]] = ones(length(ME.spec_idcs[i]))
    end
    
    P = S*W_dom
    Q = S*W_trunc
    Pinv = inv(P)
    
    B_trunc = W_trunc_inv * Bin
    B_dom = W_dom_inv * Bin

    A = P * Diagonal(Λ_dom) * Pinv
    B = P * B_dom 
    C = ME.F * W_dom * Pinv 
    if stationary_correction
        D = -ME.F * W_trunc * inv(Λ_trunc) * B_trunc
        lift = LiftingMap(W_dom*Pinv, - W_trunc*inv(Λ_trunc)*B_trunc) #W_dom*Pinv*Q*inv(Λ_trunc)*B_trunc 
    else
        D = zeros(size(ME.F, 1), size(Bin,2))
        lift = LiftingMap(W_dom*Pinv, zeros(size(Bin)))
    end

    return CSEModel(ME, W, Λ, A, B, C, D, lift) 
end

function CSEModel_Geo(ME::MasterEquation, Bin; stationary_correction = false)
    n_specs = length(ME.specs)

    dom_subspace = 1:n_specs
    trunc_subspace = n_specs+1:size(ME.M,1)

    W, Λ, Winv = compute_diagonalization(ME)
    Λ_dom = Diagonal(Λ[dom_subspace])
    W_dom = W[:, dom_subspace]
    W_dom_inv = Winv[dom_subspace, :]
    Λ_trunc = Diagonal(Λ[trunc_subspace])
    W_trunc = W[:, trunc_subspace]
    W_trunc_inv = Winv[trunc_subspace, :]
    
    S = zeros(n_specs, sum(length.(ME.spec_idcs)))
    for i in 1:n_specs
        S[i, ME.spec_idcs[i]] = ones(length(ME.spec_idcs[i]))
    end

    P = S*W_dom
    Pinv = inv(P) 

    C = ME.F*W_dom*Pinv
    D = -ME.F*W_trunc*inv(Λ_trunc)*W_trunc_inv*Bin

    B = P*W_dom_inv*Bin
    A = P*Λ_dom*Pinv 
    #loss_terms = [sum(A[:,i]) - A[i,i] + sum(C[:,i]) for i in axes(A,2)]
    #A -= Diagonal(A) + Diagonal(loss_terms)
    loss_terms = [sum(A[:,i]) + sum(C[:,i]) for i in axes(A,2)]
    A -= Diagonal(loss_terms)
    if stationary_correction
        # lift = LiftingMap(W_dom*Pinv, W_dom*Pinv*Q*Diagonal(inv(Λ_trunc))*B_trunc - W_trunc*Diagonal(inv(Λ_trunc))*B_trunc)
        # this appears the most fair representation
        lift = LiftingMap(W_dom*Pinv, - W_trunc*Diagonal(inv(Λ_trunc))*W_trunc_inv*Bin)
    else
        lift = LiftingMap(W_dom*Pinv, zeros(size(Bin)))
    end
    return CSEModel(ME, W, Λ, A, B, C, D, lift) 
end
