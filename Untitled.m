test             %n为2091152
test1(test)      %输出相对误差为0.0179
%%理论分析：最后结果在[8,16]范围内，单精度下的公差为2的-20次方，所以当加的数小于等于2的-21次方时，
%%         结果不变，即2的21次方为2097152

function n = test%%单精度浮点数计算n值，结果不变化
sum = 1;
n = 1;
e = 1;
while e ~=0
     n = n + 1;
     sum1 = single(1/n)+sum;
     e = sum1 - sum;
     sum = sum1;   
end
sum
end

function er = test1(n)
sum1 = 0;
sum2 = 0;
%单精度和双精度计算
for i =1:n
    sum1 = sum1 + single(1/i);
    sum2 = sum2 + 1/i;
end
er = (sum1 - sum2)/sum2;
end