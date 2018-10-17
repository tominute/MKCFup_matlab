function psr = calcPsr(response)
respSize = size(response);
[maxR, maxC] = find(response == max(response(:)), 1);
r_up = maxR - 5;  r_down = maxR + 5;
c_left = maxC - 5;  c_right = maxC + 5;
if r_up <= 0
    r_up = 1;
end
if r_down > respSize(1)
    r_down = respSize(1);
end
if c_left <= 0;
    c_left = 1;
end
if c_right > respSize(2)
    c_right = respSize(2);
end
mask = ones(respSize);
mask(r_up:r_down, c_left:c_right) = 0;
response_v = response(:);
response_v(mask(:) == 0) = [];
bMean = mean(response_v);
bVar = var(response_v);
maxValue = response(maxR,maxC);
psr = (maxValue - bMean) / sqrt(bVar);
end