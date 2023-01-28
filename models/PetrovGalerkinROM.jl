#=
dx/dt = M*x + B*u
dp/dt = F*x
with x = V*z + N*q, it follows that
dz/dt = Pv*M*(V*z + N*q) + Pv*B*u
dp/dt = F*V*z + F*N*q
with either
 a) dq/dt = 0 = Pn*M*(V*z + N*q) + Pn*B*u => q = -inv(Pn*M*N)*(Pn*B)*u - inv(Pn*M*N)*Pn*M*V*z
or 
 b) q = 0

Thus,
 a) dz/dt = Pv*M*(I - N*inv(Pn*M*N)*Pn*M)*V*z + Pv*(I - M*N*inv(Pn*M*N)*Pn)*B*u
    dp/dt = F*(I - N*inv(Pn*M*N)*Pn*M)*V*z - F*N*inv(Pn*M*N)*(Pn*B)*u 
or
 b) dz/dt = Pv*M*V*z + Pv*B*u
    dp/dt = F*V*z
=# 

mutable struct PetrovGalerkinROM
    V
    N
    A
    B
    C
    D
    lift
end

function build_PetrovGalerkinROM(V, N, M, Bin; F = zeros(0, size(M,2)), stationary_correction = true)
    if stationary_correction
        Pv = inv(V'*V)*V'
        Pn = inv(N'*N)*N'
        A11 = Pv*M*V
        A12 = Pv*M*N
        A21 = Pn*M*V
        A22 = Pn*M*N
        B1 = Pv*Bin
        B2 = Pn*Bin

        aux = (A22\A21)
        A = A11 - A12*aux
        B = B1 - A12*(A22\B2)
        C = F*V - F*N*aux
        D = - F*N*(A22\B2)
        lift = LiftingMap(V - N*aux, -N*(A22\B2))
    else
        A11 = Pv*M*V
        B1 = Pv*Bin

        A = A11
        B = B1
        C = F*V
        D = zeros(size(F,1), size(Bin,2))
        lift = LiftingMap(V, zeros(size(Bin)))
    end
    return PetrovGalerkinROM(V,N,A,B,C,D,lift)
end
