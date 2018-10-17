function [alphaf,d,alphaf_num1,alphaf_num2,alphaf_den1,alphaf_den2,d_num1,d_num2,d_den1,d_den2]=trainModel(...
    xo_cn2,xo_hog,yf,frame,alphaf_num1,alphaf_num2,alphaf_den1,alphaf_den2,learning_rate_hog,learning_rate_cn ,d_num1,d_num2,d_den1,d_den2,cnSigma,hogSigma)
d=[0.5 ;0.5];
% d=[1 ;0];
dim = size(xo_cn2, 3);

kf_cn = fft2(dense_gauss_kernel(cnSigma, xo_cn2,xo_cn2));

kf_hog = fft2(dense_gauss_kernel(hogSigma, xo_hog,xo_hog));
count = 0;
stop = 0;
lambda1 = 0.01;
% lambda2=0.001;
threshold = 0.03;%0.03
prevD=d;
while (stop == 0)
    new_num1 = yf.*(d(1)*kf_cn);
    new_num2 = yf.*(d(2)*kf_hog);
    new_den1 = d(1)*kf_cn.*(d(1)*conj(kf_cn)+lambda1);
    new_den2 = d(2)*kf_hog.*(d(2)*conj(kf_hog)+lambda1);
    if frame == 1
        alphaf_num11=new_num1;
        alphaf_num22=new_num2;
        alphaf_num=alphaf_num11+alphaf_num22;
        alphaf_den11=new_den1;
        alphaf_den22=new_den2;
        alphaf_den =alphaf_den11+alphaf_den22;
    else
        alphaf_num11=alphaf_num1*(1-learning_rate_cn)+learning_rate_cn*new_num1;
        alphaf_num22=alphaf_num2*(1-learning_rate_hog)+learning_rate_hog*new_num2;
        alphaf_den11=alphaf_den1*(1-learning_rate_cn)+learning_rate_cn*new_den1;
        alphaf_den22=alphaf_den2*(1-learning_rate_hog)+learning_rate_hog*new_den2;
        alphaf_num = alphaf_num11+alphaf_num22;
        alphaf_den = alphaf_den11+alphaf_den22;
    end
    alphaf = alphaf_num./alphaf_den;
    alpha= ifft2(alphaf);
%     alphaf = trainAlpha_f(kf_cn, kf_hog, prevD, yf, lambda1);
%     alpha= ifft2(alphaf);

    [d,d_num1,d_num2,d_den1,d_den2,d_num11,d_num22,d_den11,d_den22] = trainD(...
        kf_cn, kf_hog, alphaf,alpha, yf,lambda1,learning_rate_cn,learning_rate_hog,frame,d_num1,d_num2,d_den1,d_den2, dim);
%     d_num11 = 0;
%     d_num22 = 0;
%     d_den11 = 0;
%     d_den22 = 0;
%     d=[1 ;0];
    count = count + 1;
    if (count > 1)
        deltaAlpha = abs(alpha - prevAlpha);
        deltaD = abs(d - prevD);
        if (sum(deltaAlpha(:)) <= threshold * sum(abs(prevAlpha(:))) && sum(deltaD(:)) <= threshold * sum(abs(prevD(:))))
            stop = 1;
        end
    end 
    prevAlpha = alpha;
    prevD = d;
    if (count >= 100)
%         disp ('WARNING: iteration not finish!');
        d=[0.5 ;0.5];
        break;
    end
end
alphaf_num1=alphaf_num11;
alphaf_num2=alphaf_num22;
alphaf_den1=alphaf_den11;
alphaf_den2=alphaf_den22;
d_num1=d_num11;
d_num2=d_num22;
d_den1=d_den11;
d_den2=d_den22;
% if sum(d < 0) > 0
%     d1 = [1;0];  d2 = [0;1];
%     alpha_f1 = trainAlpha_f(kf_cn, kf_hog, d1, yf, lambda1);
%     alpha_f2 = trainAlpha_f(kf_cn, kf_hog, d2, yf, lambda1);
%     F1 = calcF(yf, d1, kf_cn, kf_hog, alpha_f1, lambda1, lambda2);
%     F2 = calcF(yf, d2, kf_cn, kf_hog, alpha_f2, lambda1, lambda2);
%     if (F1 <= F2)
%         d = d1;
%         alphaf = alpha_f1;
%     else
%         d = d2;
%         alphaf = alpha_f2;
%     end
% end
%  d
end