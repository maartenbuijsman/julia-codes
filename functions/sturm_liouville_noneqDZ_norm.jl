"""
    sturm_liouville_noneqDZ_norm(zf::Vector{Float64}, N2::Vector{Float64}, f::Float64, om::Float64, nonhyd::Int)

Solve for n eigenfunctions and eigenvalues for given vector N2[zf]. n = length(zf)-1.
Returns wavenumber k, wavelength L, phase, group, and eigen speeds C, Cg, and Ce, 
and vertical velocity eigenfunctions W2 at faces and horizontal velocty eigenfunction Ueig2 (normalized) 
and non-normalized Ueig1 at centers

# Arguments    
zf: layer faces [m], can either surface to bottom (e.g., 0 to -H) or bottom to surface,

N2: Brunt-Väisälä frequency squared [rad^2/s^2] at layer faces zf,

f: Coriolis frequency [rad/s],

om: internal wave frequency [rad/s],

nonhyd: if -1, solve the non-hydrostatic Sturm-Liouville problem

# info
Maarten Buijsman, Oladeji Siyanbola, USM, 2025-8-8. Based on Ashok & Bhaduria (2009). 
"""
function sturm_liouville_noneqDZ_norm(zf::Vector{Float64}, N2::Vector{Float64}, f::Float64, om::Float64, nonhyd::Int)
    # Check direction of zf (depths): assume more negative = deeper
    flipped = zf[1] > zf[end]  # if true, input is top to bottom (surface to bottom)
    
    if !flipped
        zf = reverse(zf)
        N2 = reverse(N2)
    end

    dz = -diff(zf)
    H = sum(dz)
    N = length(dz)

    # Handle hydrostatic / nonhydrostatic modes
    if nonhyd == -1
        NN = clamp.(N2 .- om^2, 1e-12, Inf)
    else
        NN = N2
    end

    # Construct B matrix
    B = Diagonal(-NN[2:end-1])

    # Construct A matrix
    A = zeros(N - 2, N - 2)
    A[1,1] = -2 / (dz[1] * dz[2])
    A[1,2] = 2 / (dz[2] * (dz[1] + dz[2]))

    for i in 2:N-3
        A[i,i-1] =  2 / (dz[i] * (dz[i] + dz[i+1]))
        A[i,i]   = -2 / (dz[i] * dz[i+1])
        A[i,i+1] =  2 / (dz[i+1] * (dz[i] + dz[i+1]))
    end

    i = N - 2
    A[i,i-1] =  2 / (dz[i] * (dz[i] + dz[i+1]))
    A[i,i]   = -2 / (dz[i] * dz[i+1])

    # Solve eigenvalue problem
    invCe2, W1 = eigen(A, B)
    Ce2 = 1 ./ invCe2
    Ce = sqrt.(Ce2)
    idx = sortperm(Ce, rev=true)
    Ce = Ce[idx]
    W1 = W1[:, idx]

    k = abs.(sqrt(om^2 - f^2) ./ Ce)
    C = om ./ k
    L = 2π ./ k
    Cg = Ce.^2 .* k ./ om

    # Compute vertical structure functions (Weig: faces)
    W2 = vcat(zeros(1, size(W1,2)), W1, zeros(1, size(W1,2)))

    # Compute horizontal eigenfunctions (Ueig: centers)
    dW2 = W2[2:end, :] .- W2[1:end-1, :]
    dzu = repeat(dz, 1, size(W1,2))
    Ueig1 = dW2 ./ dzu

    norm_factor = sqrt.(sum(Ueig1.^2 .* dzu, dims=1) ./ H)
    norm_factor[norm_factor .== 0] .= Inf
    Ueig2 = Ueig1 ./ norm_factor

    # set the correct sign
    for i in 1:size(Ueig2,2)
        if Ueig2[end,i] < 0
            Ueig1[:,i] .*= -1
            Ueig2[:,i] .*= -1            
        end
    end

    # Reverse output structure functions if input was reversed
    if !flipped
        W2    = reverse(W2, dims=1)
        Ueig1 = reverse(Ueig1, dims=1)
        Ueig2 = reverse(Ueig2, dims=1)        
    end

    return k, L, C, Cg, Ce, W2, Ueig1, Ueig2
end