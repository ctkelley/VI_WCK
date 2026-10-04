#
# Plotter for all examples
#
function Plot_VI_Examples(GF!, proj, x0, pdata;
      maxit=100, marray=[0,1,2])
    mmax = 5
    N = length(x0)
    pstr = ["k-", "k-.", "k--", "k-o", "k."]
    HistV = Vector{Vector{Float64}}()
    LegD = Vector{String}()
    plot_hist = Vector{Data_4_Plots}()
    Vstore = zeros(N, 3 * mmax + 3)
    nm = length(marray)
    for i in 1:nm
        m = marray[i]
        push!(LegD, "m=$m")
        outaa = paasol(
            GF!, x0, m, Vstore, proj; pdata = pdata,
            maxit = maxit
        )
        push!(HistV, outaa.history / outaa.history[1])
        itl = length(outaa.history)
        xcord = 0:1:(itl - 1)
        semilogy(xcord, HistV[i], pstr[i])
    end
    xlabel("Iterations")
    ylabel("Relative Residual")
    legend(LegD)
    return HistV
end

