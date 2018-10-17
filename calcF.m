function F = calcF(yf, d, cn2K, hogK, alpha_f, lambda1, lambda2)
y=ifft2(yf);
K = d(1) * cn2K + d(2) * hogK;
temp = alpha_f .* conj(K);
temp(isnan(temp)) = 0;
response = real(ifft2(temp));
a = y - response;
a = a(:)' * a(:);

alpha = ifft2(alpha_f);
b = alpha(:)' * response(:);

c = (sum(d) - 1) * (sum(d) - 1);

F = 0.5 * a + 0.5 * lambda1 * b + 0.5 * lambda2 * c;
end