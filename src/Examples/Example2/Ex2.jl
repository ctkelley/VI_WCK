function Fe2(x,pdata)
F = [x[2]*x[3]+2.0*x[1],
     x[1]*x[3]+2.0*x[2],
     x[1]*x[2]+2.0*x[3]]
return F
end

function proje2(x)
xp = max(-1.0, min(x, 2.0))
return xp
end

function build_ex2(tau)
gdata=(tau=tau, proj.proje2, F=Fe2, pdata=[])
return gdaga
end

function Ge2(x,gdata)
proj=gdata.proj
tau=gdata.tau
pdata=gdata.pdata
F=gdata.F
fx=F(x,pdata)
G = proj.(x - tau*fx)
return G
end

TestE2fp(x0=[1.0,2.0,-1.0]; tau=.1)
gdata = build_ex2(tau)
x=copy(x0)
oute2=alg1fp(x, Ge2, gdata)
oute2
end

function TestE2(x0=[1.0,2.0,-1.0]; tau=.1)
pdata=[];
x = copy(x0)
oute2=alg1(x, Fe2, pdata, proje2, tau)
oute2
end
