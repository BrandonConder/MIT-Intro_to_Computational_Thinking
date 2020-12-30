### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 1ca14906-eca1-11ea-23f6-472ed97d75aa
begin
	using Statistics
	using Images
	using FFTW
	using FFTW: fft
	using Plots
	using DSP
	using ImageFiltering
	using PlutoUI
end

# ╔═╡ 42ed52ba-ed34-11ea-26b5-05379824cbc0
md"""
# Convolutions with various kernels
"""

# ╔═╡ 4c13d558-ee15-11ea-2ed9-c5fb90d93881
kernel = Kernel.gaussian((5, 5))

# ╔═╡ 673f7ac0-ee16-11ea-35d0-cf3da430b843
sum(kernel)

# ╔═╡ 20d5eaf6-4adf-11eb-116f-e30605ab85da
K = centered(
	    [ -0.5 -1.0 -0.5
		-1.0 7.0 -1.0
		-0.5 -1.0 -0.5 ])

# ╔═╡ ea850c62-4adf-11eb-126c-7f57185269f6
edge_kernel = Kernel.sobel()[1]

# ╔═╡ a2b94e76-4ae1-11eb-129b-2187e0d884bf
img = mktemp() do fn,f
    download("https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Zebra_Herd_Michael_makalundwa.jpg/1280px-Zebra_Herd_Michael_makalundwa.jpg", fn)
    load(fn)
end

# ╔═╡ a9090e9c-4ae3-11eb-085f-ad8fe0ab15bb
# Not great, let's crop it...

# ╔═╡ cc4d65d4-4ae2-11eb-2c0f-7f7eabb3939b
size(img)

# ╔═╡ d23c2372-4ae2-11eb-145e-b77f7ee4e32c
img[280:end-100, :]

# ╔═╡ 9c90feb8-ec79-11ea-2870-31be5cedff43
md"""
# Function definitions
"""

# ╔═╡ 84e6a57c-edfc-11ea-01a0-157f1df77518
function show_colored_kernel(kernel)
	to_rgb(x) = RGB(max(-x, 0), max(x, 0), 0)
	to_rgb.(kernel) / maximum(abs.(kernel))
end

# ╔═╡ 9424b46a-ee16-11ea-1819-f17ce53e9997
show_colored_kernel(kernel)

# ╔═╡ 69c9c8f6-4adf-11eb-08c8-c1ae833e5507
show_colored_kernel(K)

# ╔═╡ f86065e8-4adf-11eb-1520-79d22cfd11d8
show_colored_kernel(edge_kernel)

# ╔═╡ 68f2afec-eca2-11ea-0758-2f22c7afdd94
function decimate(arr, ratio=5)
	return arr[1:ratio:end, 1:ratio:end]
end

# ╔═╡ aa3b9bd6-ed35-11ea-1bdc-33861bdbd29a
function shrink_image(image, ratio=5)
	(height, width) = size(image)
	new_height = height ÷ ratio - 1
	new_width = width ÷ ratio - 1
	list = [
		mean(image[
			ratio * i:ratio * (i + 1),
			ratio * j:ratio * (j + 1),
		])
		for j in 1:new_width
		for i in 1:new_height
	]
	reshape(list, new_height, new_width)
end

# ╔═╡ 6d39fea8-ed3c-11ea-3d7c-3f62ca91ce23
begin
	large_image = load("cat_in_bowtie.jpg")
	# Since the instructor didn't include an image I've fetched one from https://pixabay.com/photos/cat-pet-pets-bowtie-blue-russian-4122355/
	image = shrink_image(large_image, 2)
end

# ╔═╡ 2f446dcc-ee15-11ea-0e78-931ff507b5e5
size(image)

# ╔═╡ 14d5b144-ee18-11ea-0080-c187f068c168
image

# ╔═╡ 959c4a4c-4ae1-11eb-3dbb-7385aad5e832
image

# ╔═╡ 160eb236-eca1-11ea-1dbe-47ad61cc9397
function rgb_to_float(color)
    return mean([color.r, color.g, color.b])
end

# ╔═╡ fa3c5074-eca0-11ea-2d2d-bb6bcdeb834c
function fourier_spectrum_magnitudes(img)
    grey_values = rgb_to_float.(img)
    spectrum = fftshift(fft(grey_values))
	return abs.(spectrum)
end

# ╔═╡ e40d807e-ed3a-11ea-2340-7f98bd5d04a2
function plot_1d_fourier_spectrum(img, dims=1)
	spectrum = fourier_spectrum_magnitudes(img)
	plot(centered(mean(spectrum, dims=1)[1:end]))
end

# ╔═╡ 98be16e2-4ae1-11eb-2f2e-e9b5133e4c0d
plot_1d_fourier_spectrum(image)

# ╔═╡ 6fa7b91c-4ae2-11eb-12e2-258552010c94
plot_1d_fourier_spectrum(image)

# ╔═╡ f23260d8-4ae2-11eb-1b4e-3726cf59bcfa
plot_1d_fourier_spectrum(img[280:end-100, :])

# ╔═╡ beb6b4b0-eca1-11ea-1ece-e3c9931c9c13
function heatmap_2d_fourier_spectrum(img)
	heatmap(log.(fourier_spectrum_magnitudes(img)))
end

# ╔═╡ 18045956-ee18-11ea-3e34-612133e2e39c
heatmap_2d_fourier_spectrum(image)

# ╔═╡ 57f2d9ac-4ae3-11eb-31a8-7744dcc9e6c8
heatmap_2d_fourier_spectrum(img[280:end-100, :])

# ╔═╡ 58f4754e-ed31-11ea-0464-5bfccf397966
function clamp_at_boundary(M, i, j)
	return M[
		clamp(i, 1, size(M, 1)),
		clamp(j, 1, size(M, 2)),	
	]
end

# ╔═╡ f28af11e-ed31-11ea-2b46-7dff147ccb48
function rolloff_boundary(M, i, j)
	if (1 ≤ i ≤ size(M, 1)) && (1 ≤ j ≤ size(M, 2))
		return M[i, j]
	else
		return 0 * M[1, 1]
	end
end

# ╔═╡ 572cf620-ecb2-11ea-0019-21666a30d9d2
function convolve(M, kernel, M_index_func=clamp_at_boundary)
    height = size(kernel, 1)
    width = size(kernel, 2)
    
    half_height = height ÷ 2
    half_width = width ÷ 2
    
    new_image = similar(M)
	
    # (i, j) loop over the original image
    @inbounds for i in 1:size(M, 1)
        for j in 1:size(M, 2)
            # (k, l) loop over the neighbouring pixels
			new_image[i, j] = sum([
				kernel[k, l] * M_index_func(M, i - k, j - l)
				for k in -half_height:-half_height + height - 1
				for l in -half_width:-half_width + width - 1
			])
        end
    end
    
    return new_image
end

# ╔═╡ 5afed4ea-ee18-11ea-1aa4-abca154b3793
conv_image = convolve(image, kernel)

# ╔═╡ 6340c0f8-ee18-11ea-1765-45f4bc140670
heatmap_2d_fourier_spectrum(conv_image)

# ╔═╡ 6e8046b8-4adf-11eb-3795-f797c8cfd7a9
conv_image_2 = convolve(image, K)

# ╔═╡ 007a9960-4ae0-11eb-0594-8f33bf89fecb
conv_image_3 = Gray.(abs.(convolve(image, edge_kernel)))

# ╔═╡ 66f89ffe-4ae3-11eb-20fa-27d50006b52c
heatmap_2d_fourier_spectrum(convolve(img[280:end-100, :], kernel))

# ╔═╡ fb8a0a8c-4ae2-11eb-0d95-b7ed3d03b619
Gray.(abs.(convolve(convert.(ColorTypes.RGB{Float64}, img[280:end-100, :]), Kernel.sobel()[2])))

# ╔═╡ 587092e4-ecb2-11ea-18fc-ad5e9778fb30
box_blur(n) = centered(ones(n, n) ./ (n^2))

# ╔═╡ 991cb9b8-ecb8-11ea-3f80-5d95b2200259
function gauss_blur(n, sigma=0.25)
	kern = gaussian((n, n), sigma)
	return kern / sum(kern)
end

# ╔═╡ Cell order:
# ╟─42ed52ba-ed34-11ea-26b5-05379824cbc0
# ╠═6d39fea8-ed3c-11ea-3d7c-3f62ca91ce23
# ╠═2f446dcc-ee15-11ea-0e78-931ff507b5e5
# ╠═4c13d558-ee15-11ea-2ed9-c5fb90d93881
# ╠═9424b46a-ee16-11ea-1819-f17ce53e9997
# ╠═673f7ac0-ee16-11ea-35d0-cf3da430b843
# ╠═14d5b144-ee18-11ea-0080-c187f068c168
# ╠═18045956-ee18-11ea-3e34-612133e2e39c
# ╠═5afed4ea-ee18-11ea-1aa4-abca154b3793
# ╠═6340c0f8-ee18-11ea-1765-45f4bc140670
# ╠═20d5eaf6-4adf-11eb-116f-e30605ab85da
# ╠═69c9c8f6-4adf-11eb-08c8-c1ae833e5507
# ╠═6e8046b8-4adf-11eb-3795-f797c8cfd7a9
# ╠═ea850c62-4adf-11eb-126c-7f57185269f6
# ╠═f86065e8-4adf-11eb-1520-79d22cfd11d8
# ╠═007a9960-4ae0-11eb-0594-8f33bf89fecb
# ╠═959c4a4c-4ae1-11eb-3dbb-7385aad5e832
# ╠═98be16e2-4ae1-11eb-2f2e-e9b5133e4c0d
# ╠═a2b94e76-4ae1-11eb-129b-2187e0d884bf
# ╠═6fa7b91c-4ae2-11eb-12e2-258552010c94
# ╠═a9090e9c-4ae3-11eb-085f-ad8fe0ab15bb
# ╠═cc4d65d4-4ae2-11eb-2c0f-7f7eabb3939b
# ╠═d23c2372-4ae2-11eb-145e-b77f7ee4e32c
# ╠═f23260d8-4ae2-11eb-1b4e-3726cf59bcfa
# ╠═57f2d9ac-4ae3-11eb-31a8-7744dcc9e6c8
# ╠═66f89ffe-4ae3-11eb-20fa-27d50006b52c
# ╠═fb8a0a8c-4ae2-11eb-0d95-b7ed3d03b619
# ╠═9c90feb8-ec79-11ea-2870-31be5cedff43
# ╟─1ca14906-eca1-11ea-23f6-472ed97d75aa
# ╟─84e6a57c-edfc-11ea-01a0-157f1df77518
# ╟─68f2afec-eca2-11ea-0758-2f22c7afdd94
# ╟─aa3b9bd6-ed35-11ea-1bdc-33861bdbd29a
# ╟─160eb236-eca1-11ea-1dbe-47ad61cc9397
# ╟─fa3c5074-eca0-11ea-2d2d-bb6bcdeb834c
# ╟─e40d807e-ed3a-11ea-2340-7f98bd5d04a2
# ╟─beb6b4b0-eca1-11ea-1ece-e3c9931c9c13
# ╟─58f4754e-ed31-11ea-0464-5bfccf397966
# ╟─f28af11e-ed31-11ea-2b46-7dff147ccb48
# ╟─572cf620-ecb2-11ea-0019-21666a30d9d2
# ╟─587092e4-ecb2-11ea-18fc-ad5e9778fb30
# ╟─991cb9b8-ecb8-11ea-3f80-5d95b2200259
