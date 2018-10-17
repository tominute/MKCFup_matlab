function k = dense_gauss_kernel(sigma, x1, x2)

% k = dense_gauss_kernel(sigma, x, y)
%
% Computes the kernel output for multi-dimensional feature maps x and y
% using a Gaussian kernel with standard deviation sigma.

c = ifft2(sum(fft2(x1) .* conj(fft2(x2)), 3));
d = x1(:)' * x1(:) + x2(:)' * x2(:) - 2 * c;
%  k = exp(-1 / sigma^2 * max(0, real(d)) / numel(d));
k = exp(-1 / sigma^2 * d / numel(d));
end