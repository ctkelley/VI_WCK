function tstA(n; maxit=100, tau=.001)
pdata=ex1_build(n)
x=zeros(n); x .= pdata.uD
ithist=alg1(x, F, pdata, proj0, tau; maxit=maxit)
ithist
end


function alg1(x, F, pdata, proj, tau; maxit=100)
ithist=zeros(maxit+1)
for i=1:maxit
    fx=F(x,pdata)
    ithist[i]=norm(fx,Inf)
    x .= proj.(x - tau*fx)
end
    fx=F(x,pdata)
    ithist[maxit+1]=norm(fx,Inf)
return ithist
end 

function proj0(x)
proj0=max(x,0.0)
return(proj0)
end
