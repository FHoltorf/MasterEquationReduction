cd(@__DIR__)
using Pkg
Pkg.activate("../../.")

using LinearAlgebra, NMF, CairoMakie

include("MasterEquation.jl")
include("NaiveTruncation.jl")
include("PetrovGalerkinROM.jl")
include("CSEROM.jl")
include("CSCROM.jl")
include("BalancingROM.jl")

 """ 
reaction network 
    S1 + S2 → A ⇋ B ⇋ C → P

energy grains
    *n* uniformly random steps with expected distance *σ*
    the multiplicity ρ of states is drawn randomly from a given collection

kinetic parameters
    normalized to one and weighted by the collision frequency *ω_coll*, i.e., M = *ω_col*Z + K

forcing
    computed as randomized Boltzmann distributions on the energy grains E

"""
# reaction network specifications
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
B = forcing_strength*RandomizedBimolecularForcing(ME, :A, bR1, bR2)

# data collection from balanced truncation
h = 1e-3
h_range = h*1:300 
n_bal = 3
impulse_response = sqrt(h)*reduce(hcat, exp(h*ME.M)*B for h in h_range)
control_gramian = impulse_response * impulse_response'
dual_impulse_response = sqrt(h)*reduce(hcat, exp(h*ME.M')*ME.F' for h in h_range)
obs_gramian = dual_impulse_response*dual_impulse_response'

T_bal, T_bal_inv, Σ = factor_balancing(impulse_response, ME.M') #dual_impulse_response)
balancedROM = BalancedROM(T_bal, T_bal_inv, ME, B, n_bal, stationary_correction = false)


obs = T_bal_inv'*obs_gramian*T_bal_inv
control = T_bal*control_gramian*T_bal'