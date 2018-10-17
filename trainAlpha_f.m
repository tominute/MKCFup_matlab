function alpha_f = trainAlpha_f(cn2K, hogK, d, yf, lambda1)

K = cn2K * d(1) + hogK * d(2);
alpha_f = yf ./ (K + lambda1);
end