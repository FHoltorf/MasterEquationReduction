using QuadGK

function evaluate_product(C, D, sol, u, p0 = zeros(size(C,1)))
    products = [quadgk(t -> C*sol(t) + D*u(t), sol.t[i-1], sol.t[i])[1] for i in 2:length(sol.t)]
    pushfirst!(products, p0)
    return cumsum(products)
end

function evaluate_product(C, sol, p0 = zeros(size(C,1)))
    products = [quadgk(t -> C*sol(t), sol.t[i-1], sol.t[i])[1] for i in 2:length(sol.t)]
    pushfirst!(products, p0)
    return cumsum(products)
end

function evaluate_product(ME::MasterEquation, Bin, c::Array, u, t, p0 = zeros(size(ME.F,1)))
#p = F*inv(M)*c + q
    #q = p0 - F*inv(M)*B*(c0 + B*fed_material)
    #fed_material = int_0^t u(τ) dτ
    c0 = c[:,1]
    fed_material = reduce(hcat, cumsum([quadgk(u, t[i-1], t[i])[1] for i in 2:length(t)]))
    converted_material = (c0 .- c[:, 2:end]) + Bin*fed_material
    p = p0 .- ME.F*(ME.M\converted_material)
    return p
end

function evaluate_product(ME::MasterEquation, Bin, c, u, t, p0 = zeros(size(ME.F,1)))
    return evaluate_product(ME, Bin, reduce(hcat, c(τ) for τ in t), u, t, p0=p0)
end