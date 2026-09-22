"""
alg1(x, F, pdata, proj, tau; maxit=100)

Simple solver from the paper

Inputs: 
x = initial iterate
F(x,pdata) = nonlinear residual; pdata=precomputed data
proj = projection onto feasible set
tau = step size

Output:
sol = solution
ithist = iteration history
"""
function alg1(x, F, pdata, proj, tau; maxit = 100)
    ithist = zeros(maxit + 1)
    for i in 1:maxit
        fx = F(x, pdata)
        ithist[i] = norm(fx, Inf)
        x .= proj.(x - tau * fx)
    end
    fx = F(x, pdata)
    ithist[maxit + 1] = norm(fx, Inf)
    return (sol = x, ithist = ithist)
end

function proj0(x)
    proj0 = max(x, 0.0)
    return (proj0)
end
