#
# Functions for Example 2
#

#
# Residual function F
#
function Fe2(x,pdata)
F = [x[2]*x[3]+2.0*x[1],
     x[1]*x[3]+2.0*x[2],
     x[1]*x[2]+2.0*x[3]]
return F
end

#
# projection
#
function proje2(x)
xp = max(-1.0, min(x, 2.0))
return xp
end

#
# Fixed point map for this problem. Note how
# the projection is part of the map
#
function Ge2(x,gdata)
proj=gdata.proj
tau=gdata.tau
pdata=gdata.pdata
F=gdata.F
fx=F(x,pdata) 
G = proj.(x - tau*fx)
return G
end

#
# Fixed point map for this problem with storage
# for G preallocated. Note how
# the projection is part of the map
#
function Ge2!(G,x,gdata)
proj=gdata.proj
tau=gdata.tau
pdata=gdata.pdata
F=gdata.F
fx=F(x,pdata)
G .= proj.(x - tau*fx)
return G
end


#
# build the data for the problem which we pass
# to the functions
#
function build_ex2(tau)
gdata=(tau=tau, proj=proje2, F=Fe2, pdata=[])
return gdata
end

#
# Alg 1 in fixed point form.
#
function TestE2fp(x0=[1.0,2.0,-1.0]; tau=.1)
gdata = build_ex2(tau)
x=copy(x0)
oute2=alg1fp(x, Ge2, gdata)
oute2
end

#
# Alg 1 with the formulation in the paper.
#
function TestE2(x0=[1.0,2.0,-1.0]; tau=.1)
pdata=[];
x = copy(x0)
oute2=alg1(x, Fe2, pdata, proje2, tau)
oute2
end

#
# Results for Alg 1. We use the fixed point formulation
# for all examples because it is simpler.
#
function Alg1E2(x0=[1.0,2.0,-1.0]; tau=.1)
gdata = build_ex2(tau)
x=copy(x0)
oute2=alg1fp(x, Ge2, gdata)
semilogy(oute2.ithist/oute2.ithist[1],"k-")
xlabel=("Iterations")
ylabel("Relative Residual Norm")
title("Alg 1; Example 2")
end

#
# Results for Alg 1. We use the fixed point formulation
# for all examples because it is simpler.
#
function Alg1E2AA(x0=[1.0,2.0,-1.0]; tau=.1, m=3, maxit=100, proj=proje2)
gdata = build_ex2(tau)
x=copy(x0)
N=length(x)
(m == 0) ? d1=4 : d1=3
Vstore=zeros(N, 3*m+d1)
outaa=paasol(Ge2!, x0, m, Vstore, proj; pdata=gdata, maxit=maxit)
semilogy(outaa.history/outaa.history[1],"k-")
xlabel=("Iterations")
ylabel("Relative Residual Norm")
title("Alg 1; Example 2")
end


