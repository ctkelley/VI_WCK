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
        G = proj.(x - tau * fx)
        ithist[i] = norm(x-G, Inf)
        x .= G
    end
    fx = F(x, pdata)
    G = proj.(x - tau * fx)
    ithist[maxit + 1] = norm(x-G, Inf)
    return (sol = x, ithist = ithist)
end

function alg1fp(x, G, gdata; maxit=100)
    ithist = zeros(maxit + 1)
    for i in 1:maxit
        Gx=G(x,gdata)
        ithist[i] = norm(x-Gx, Inf)
        x .= Gx
    end
    Gx = G(x,gdata)
    ithist[maxit + 1] = norm(x-Gx, Inf)
    return (sol = x, ithist = ithist)
end


