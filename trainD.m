function [d,d_num1,d_num2,d_den1,d_den2,d_num11,d_num22,d_den11,d_den22] = trainD(kf_cn, kf_hog, alphaf,alpha, yf,lambda1,learning_rate_cn,learning_rate_hog,frame,d_num1,d_num2,d_den1,d_den2, dim)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%method-1
% temp1=ifft2(conj(kf_cn) .* alphaf);
% temp2=ifft2(conj(kf_hog) .* alphaf);
% A=[];B=[];
% Aeq=[1 1];
% Beq=1;
% lb=zeros(2,1);
% H=zeros(2,2);
% y=ifft2(yf);
% %options=optimset('Algorithm','active-set');
% H(1,1)=(temp1(:))'*temp1(:);
% H(1,2)=0;
% H(2,1)=H(1,2);
% H(2,2)=(temp2(:))'*temp2(:);
% f=zeros(2,1);
% temp=lambda1*alpha-2*y;
% f(1)=(temp(:))'*temp1(:);
% f(2)=(temp(:))'*temp2(:);
% f=f*0.5;
% options = optimoptions('quadprog','Display','none'); 
% d=quadprog(H,real(f),A,B,Aeq,Beq,lb,[],[],options);
% d_num11= 0;
% d_num22 = 0;
% d_den11 = 0;
% d_den22 = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%method-2
% y=ifft2(yf);
% lambda2=0.001;
% lambda1=0.01;
% B = zeros(2,2);
% c = zeros(2,1);
% rgbK_f = kf_cn;
% hogK_f = kf_hog;
% rgbK_c_f = rgbK_f;
% hogK_c_f = hogK_f;
% k11_f = (conj(rgbK_c_f) .* rgbK_f) * 2;
% k12_f = (conj(hogK_c_f) .* rgbK_f) + (conj(rgbK_c_f) .* hogK_f);
% k21_f = k12_f;
% k22_f = (conj(hogK_c_f) .* hogK_f) * 2;
% temp = ifft2(conj(k11_f) .* alphaf);
% B(1,1) = (alpha(:))' * temp(:) + 2 * lambda2;
% temp = ifft2(conj(k12_f) .* alphaf);
% B(1,2) = (alpha(:))' * temp(:) + 2 * lambda2;
% temp = ifft2(conj(k21_f) .* alphaf);
% B(2,1) = (alpha(:))' * temp(:) + 2 * lambda2;
% temp = ifft2(conj(k22_f) .* alphaf);
% B(2,2) = (alpha(:))' * temp(:) + 2 * lambda2;
% 
% temp = 2 * y - lambda1 * alpha;
% temp2 = ifft2(conj(rgbK_f) .* alphaf);
% c(1) = temp(:)' * temp2(:) + 2 * lambda2;
% temp2 = ifft2(conj(hogK_f) .* alphaf);
% c(2) = temp(:)' * temp2(:) + 2 * lambda2;
% 
% % d = searchD_GradientMethod(cn2K, hogK, alpha_f, y, lambda1, lambda2, B, c, prevD);
% d = searchD_NewtonMethod(B, c, prevD);
% % d = B \ c;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%method-3
d=zeros(2,1);

for i = 1:1
temp1=ifft2(conj(kf_cn) .* alphaf);
temp2=ifft2(conj(kf_hog) .* alphaf);
end
y=ifft2(yf);
temp=2*y-lambda1*alpha;
new_num1 =(temp(:))'*temp1(:);
new_num2 =(temp(:))'*temp2(:);
new_den1 =2*(temp1(:))'*temp1(:);
new_den2 =2*(temp2(:))'*temp2(:);
if frame == 1
    d_num11=new_num1;
    d_num22=new_num2;
    d_den11=new_den1;
    d_den22=new_den2;
else
    d_num11=d_num1*(1-learning_rate_cn)+learning_rate_cn*new_num1;
    d_num22=d_num2*(1-learning_rate_hog)+learning_rate_hog*new_num2;
    d_den11=d_den1*(1-learning_rate_cn)+learning_rate_cn*new_den1;
    d_den22=d_den2*(1-learning_rate_hog)+learning_rate_hog*new_den2;
end

if dim ~= 1
   
    d(1)=d_num11/d_den11;
    d(2)=d_num22/d_den22;
    
    %normlize1
%     summ = sum(d);
%     d(1) = d(1)/summ;
%     d(2) = d(2)/summ;
    %normlize2
%     sum = d(1)*d(1)+d(2)*d(2);
%     d(1) = d(1)/sqrt(sum);
%     d(2) = d(2)/sqrt(sum);
    
else
    d(1)=d_num11/d_den11;
    d(2)=d_num22/d_den22;
    
    if d(2)>1
        d(2)=0.5;
    end
    d(1)=1-d(2);
end

% if d(1)>1
%     d(1)=0.5;
% end
% d(2)=1-d(1);
end