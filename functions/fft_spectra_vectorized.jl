"""
    fft_spectra(t, y; tukeycf=0.5, numwin=1, linfit=true, prewhit=false)

Perform simple spectral analysis similar to the MATLAB's fft_spectra function.

# Arguments
- `t::AbstractVector{<:Real}`: Time vector
- `y::AbstractVector{<:Real}`: Data vector
- `tukeycf::Float64`: Tukey window parameter (1=hanning, 0=boxcar)
- `numwin::Int`: Number of overlapping windows (50% overlap)
- `linfit::Bool`: Remove linear trend
- `prewhit::Bool`: Apply prewhitening

# Returns
- `period::Vector{Float64}`
- `freq::Vector{Float64}`
- `power::Vector{Float64}`

# Example
See below the code of this function

# Info
Maarten Buijsman, USM, 2025-8-4l fft_spectra_vectorized.jl
https://chatgpt.com/c/6890bc81-9f54-800d-910c-bae1637ec9ac
"""
function fft_spectra(t::AbstractVector{<:Real}, y::AbstractVector{<:Real};
    tukeycf::Float64 = 0.5, numwin::Int = 1, linfit::Bool = true, prewhit::Bool = false)

    # Convert to Float64
    t = collect(float.(t))
    y = collect(float.(y))

    # Ensure even length
    if isodd(length(t))
        t = t[1:end-1]
        y = y[1:end-1]
    end

    dt = t[2] - t[1]
    nt1 = length(t)

    # Compute window size and create overlapping windows
    inw = floor(Int, nt1 ÷ (numwin + 1))
    windows = [(i*inw + 1):(i*inw + 2*inw) for i in 0:(numwin-1)]

    # Window length for freq calculation
    nt = length(windows[1])
    df = 1 / (dt * nt)
    freq = (1:nt ÷ 2) ./ (dt * nt)
    period = 1.0 ./ freq

    # Collect all windowed segments into a matrix
    segments = [y[w] for w in windows]

    # example on how to access vectors in vectors .....
    # println("begin, end values = ", segments[1][[1 end]])

    # Remove linear trend and mean if requested
    if linfit
        segments = [begin
    #        println("len segment: ", length(seg))
            x = 1:length(seg)
            X = hcat(x, ones(length(x)))
            β = X \ seg
            seg .- X * β
        end for seg in segments];
    else
        segments = [begin 
    #        println("mean segment: ", mean(seg))        
            seg .- mean(seg) 
        end for seg in segments];
    end

    # Apply window
    # tukey(dims, α; padding=0, zerophase=false)
    # For α == 0, the window is equivalent to a rectangular window. 
    # For α == 1, the window is a Hann window.
    base_window = tukey(nt,tukeycf)
    segments    = [seg .* base_window for seg in segments]

    # Prewhiten if needed
    if prewhit
        segments = [begin
            d = diff(seg) ./ dt
            isodd(length(d)) ? d[1:end-1] : d
        end for seg in segments]
    end

    # Compute FFT and power for all windows
    # for vectors, this FFT is the same as in MATLAB
    # and yields the same results
    power_matrix = map(segments) do seg
        n = length(seg)
        Y = fft(seg) .* dt
        deleteat!(Y, 1)         # remove first value because that is the mean of seg
        P2 = abs2.(Y)           # energy y_unit^2*t_unit^2
        P1 = 2 .* P2[1:n ÷ 2]   # select left side and double the power (= folding the spectrum)
        # println("if parseval hold theorem we get 1: ",sum(seg.^2*dt)/sum(P1*df))        
        prewhit ? (P1 .* df) ./ (freq .^ 2) : P1 .* df  # if true the first argument is returned to power_matrix
    end

    # Convert to matrix and average
    power = reduce(+, power_matrix) ./ numwin

    return period, freq, power
end

#=
    % Parseval's theorem ratio between integrated energy and power (should be 1)
    % note that re-redding is excluded 
    % sum(y.^2*dt)/sum(P1*df)  % this is OK

    % NOTE that including re-reddening Parseval's theorem does not hold anymore ......
    % sum(yd.^2*dt)/sum(P1*df./freq.^2) 
=#

#= Example
T1, T2, T3 = 0.5, 1, 2
t = collect(1:1447)/24
y = 1.0 .*  cos.(2π .* t ./ T1) .+ 0.5 .*  cos.(2π .* t ./ T2) .+  0.25 .* cos.(2π .* t ./ T3)

fig1 = Figure(); ax1 = fig1[1, 1]; lines(ax1, t, y); fig1  

tukeycf=0.0; numwin=3; linfit=true; prewhit=false;
period, freq, power = fft_spectra(t, y; tukeycf, numwin, linfit, prewhit);

fig2 = Figure(); ax = Axis(fig2[1, 1], title = "Power Spectrum", xlabel = "Frequency [1/unit]", ylabel = "Power", yscale = log10)
lines!(ax, freq, power, color = :green, linewidth = 2)
expected_freqs = [1/T1, 1/T2, 1/T3]; expected_power = fill(mean(power), 3); # Mark expected frequencies
scatter!(ax, expected_freqs, expected_power, color = [:red, :blue, :orange], markersize = 14)
text!(ax, expected_freqs, expected_power .* 1.2; text = ["T1=$T1", "T2=$T2", "T3=$T3"], align = (:center, :bottom), fontsize = 14)
xlims!(ax, (0, 2.5)); ylims!(ax, (1e-4, 1e2)); fig2
=#