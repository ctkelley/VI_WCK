module VI_WCK

struct Data_4_Plots
    relreshist::Vector{Float64}
    itdata::Vector{Int}
    fdata::Vector{Int}
    legend::String
end

using LinearAlgebra
using LinearAlgebra.BLAS
using LinearAlgebra.LAPACK
using SuiteSparse
using SparseArrays
using SIAMFANLEquations
using PythonPlot

import SIAMFANLEquations.Anderson_Init
import SIAMFANLEquations.EvalF!
import SIAMFANLEquations.ItStatsA
import SIAMFANLEquations.updateHist!
import SIAMFANLEquations.updateStats!
import SIAMFANLEquations.falpha
import SIAMFANLEquations.Orthogonalize!
import SIAMFANLEquations.AndersonOK
import SIAMFANLEquations.CloseIteration

include("Solvers/alg1.jl")
include("Solvers/paasol.jl")
include("Examples/Example2/Ex2.jl")

export alg1, paasol
export Alg1E2, Alg1E2AA, AAE2Plot
end
