module VI_WCK

using LinearAlgebra
using LinearAlgebra.BLAS
using LinearAlgebra.LAPACK
using SuiteSparse
using SparseArrays
using SIAMFANLEquations
using PythonPlot


include("Solvers/alg1.jl")
include("Examples/Example2/Ex2.jl")

export alg1
export Alg1E2

end
