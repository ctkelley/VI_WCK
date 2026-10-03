module VI_WCK

using LinearAlgebra
using LinearAlgebra.BLAS
using LinearAlgebra.LAPACK
using SuiteSparse
using SparseArrays
using SIAMFANLEquations


include("Solvers/alg1.jl")
include("Examples/Example2/Ex2.jl")

export alg1
export Fe2, TestE2, Ge2, TestE2fp

end
