# here the idea is to find the reduced order model by fitting directly to data
# the model structure is assumed to satisfy
# x = Dz where D is column/left stochastic (e'*D = 1)
# z follows a dynamical system which ensures that mass is conserved and z >= 0, i.e.,
# dz/dt = A*z + B*u where A is Metzler and 

# versuchen wir erst InfiniteOpt.jl
cd(@__DIR__)
using Pkg
Pkg.activate("../../.")

using LinearAlgebra, NMF, CairoMakie, InfiniteOpt, Ipopt, DifferentialEquations, BlockDiagonals

include("MasterEquation.jl")
include("CSEROM.jl")
include("PetrovGalerkinROM.jl")
include("CSCROM.jl")
specs = [:A, :B, :C]
network = [(:A, :B), (:B, :C)]
products = Dict(:C => [:P])
σ = 0.1
n = 100
ω_coll = 10.0
ρ = Dict(spec => rand(1:4, n) for spec in specs)
ρ_S1 = 1:10
ρ_S2 = 1:10
forcing_strength = 1.0/300

# computed from inputs
ΔE = 2*σ*rand(n-1)
E = [0, [sum(ΔE[1:i]) for i in 1:n-1]...]

# Master Equation
ME = RandomizedMasterEquation(specs, network, ρ, E, ω_coll, products)

# forcing 
bR1 = RandomizedBoltzmannDistribution(ρ_S1, E)
bR2 = RandomizedBoltzmannDistribution(ρ_S2, E)
Bin = forcing_strength*RandomizedBimolecularForcing(ME, :A, bR1, bR2)

function master_equation!(dx, x, p, t)
    A, B, u = p
    dx .= A * x + B * u(t)
end
x0 = zeros(size(ME.M, 1))
tspan = (0.0, 1.0)
scale = 100.0
full_problem = ODEProblem(master_equation!, x0, tspan, (ME.M, Bin, t->scale))
sol = solve(full_problem, saveat = 0.1)

cse_model = CSEModel(ME,Bin, stationary_correction=true)
cse_problem = ODEProblem(master_equation!, zeros(size(cse_model.A,1)), tspan, (cse_model.A, cse_model.B, t->scale))
cse_sol = solve(cse_problem, saveat=sol.t)

n_modes = 10
n_data = size(ME.M,1)
model = InfiniteModel(Ipopt.Optimizer)
@infinite_parameter(model, t in [tspan...], supports = sol.t, derivative_method = OrthogonalCollocation(4))
@parameter_function(model, data[i = 1:n_data] == t -> sol(t)[i])
@variable(model, z[i in 1:n_modes], Infinite(t))
#=
for i in 1:n_modes 
    set_start_value_function(z[i], t->cse_sol(t)[i])
end
=#
@variable(model, D[i in 1:size(ME.M,1), j in 1:n_modes] >= 0)#, start = max(0,cse_model.W[i,j]))
@variable(model, A[i in 1:n_modes, j in 1:n_modes] >= 0)#, start = i == j ? 0 : cse_model.A[i,j])
@variable(model, B[i in 1:n_modes] >= 0)#, start = cse_model.B[i])
@variable(model, C[i in 1:n_modes] <= 0)#, start = cse_model.A[i,i])
@constraint(model, aux_con[i in 1:n_modes], A[i,i] == 0)
#@constraint(model, conservation[i in 1:n_modes], C[i] + sum(A[:,i]) == 0)
@constraint(model, normalization[i in 1:n_modes], 1 == sum(D[:,i]))
@constraint(model, dynamics[i in 1:n_modes], ∂(z[i],t) == A[i,:]'*z + C[i]*z[i] + B[i]*scale) 
@constraint(model, init_condition[i in 1:n_modes], z[i](0) == 0.0)
@objective(model, Min, ∫(sum(abs2, data[i] - D[i,:]'*z for i in 1:n_data), t))
optimize!(model)


A_opt = value.(A) + Diagonal(value.(C))
B_opt = value.(B)
D_opt = value.(D)
red_problem = ODEProblem(master_equation!, zeros(n_modes), tspan, (A_opt, B_opt, t -> scale))
red_sol = solve(red_problem, saveat = 0.001)
fine_sol = solve(full_problem, DifferentialEquations.CVODE_BDF(), saveat = 0.001)
cse_fine_sol = solve(cse_problem, saveat=0.001)

S = BlockDiagonal([ones(1,n), ones(1,n), ones(1,n)])
# untransform
cse_inv = unsymmetrize(cse_model.W[:,1:3], ME)*inv(S*unsymmetrize(cse_model.W[:,1:3], ME))

fig = Figure(fontsize = 24)
ax = Axis(fig[1,1], title = "error trace", xlabel = "time", ylabel = "error")
lines!(ax, red_sol.t, [norm(D_opt*red_sol(t) - sol(t))/norm(sol(t)) for t in red_sol.t], color = :black)
lines!(ax, cse_fine_sol.t, [norm(cse_inv*cse_fine_sol(t) - sol(t))/norm(sol(t)) for t in cse_fine_sol.t], color = :red)
fig

visualize_CSCs(D_opt, 5, 1)

fig = Figure(fontsize = 24)
ax = Axis(fig[1,1], title = "error trace", xlabel = "time", ylabel = "error")
red_trace = S*D*Array(red_sol)
full_trace = S*Array(fine_sol)
cse_trace = Array(cse_fine_sol)
for i in 1:3
    lines!(ax, fine_sol.t, full_trace[i,:])
    lines!(ax, red_sol.t, red_trace[i,:], color = :black, linestyle = :dash)
    lines!(ax, cse_fine_sol.t, cse_trace[i,:], color = :red)
end
fig

s
#=
function objective(x, prob, data, dt, n, n_modes)
    D = reshape(x[1:n*n_modes], n, n_modes)
    A = reshape(x[n*n_modes+1:end], n_modes, n_modes)
    B = x[n*n_modes + n_modes^2+1:end]
    _prob = remake(prob, p = (A,B))
    sol = Array(solve(_prob, saveat=dt))
    return sum(abs2, D*sol - data)
end


function constraints!(nlp, x, val)
    @unpack n, n_modes = nlp
    # D is left stochastic 
    D = reshape(x[1:n*n_modes], n, n_modes)
    A = reshape(x[n*n_modes+1:end], n_modes, n_modes)
    B = x[n*n_modes + n_modes^2+1:end]
    
    # stochasticity
    n_cons = 0
    for i in n_cons + 1 : n_cons + n_modes
        val[i] = sum(D[:,i]) - 1
    end
    n_cons += n_modes

    # non-negativity of D entries
    for i in n_cons + 1 : n_cons + n_modes*n
        val[i] = D[i - n_cons]
    end
    n_cons += n_modes*n

    # metzlerity + conservation of dynamics
    for i in n_cons+1 : n_cons + n_modes^2
        k, j = divrem(i - (n_cons + 1), n_modes) .+ 1
        if k == j
            val[i] = sum(A[k,:])
        else
            val[i] = A[j,k]
        end
    end
    n_cons += n_modes^2

    # non-negativity of B entries
    for i in n_cons+1 : n_cons + n_modes
        val[i] = B[i - (n_cons + 1)]
    end

    return 
end
=#