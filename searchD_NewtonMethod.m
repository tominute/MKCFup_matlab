function x = searchD_NewtonMethod(B, c, x0)
maxk = 100;
epsilon = 1e-3;  k = 0;
x = x0;
while k < maxk
    g = 0.5 * (B * x - c);
    if norm(g) < epsilon
        break;
    end
    G = 0.5 * B;
    d = G \ (-g);
    x = x + d;
    k = k + 1;
end
end