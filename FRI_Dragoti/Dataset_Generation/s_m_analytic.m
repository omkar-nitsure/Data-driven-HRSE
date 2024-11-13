function [s_m_a]= s_m_analytic(t_k,a_k)
K = 2;

%lambda = 1; % Lambda value
P=20;
omega_0 = -(P*pi)/ (P+1) ;
%omega_0 =0 ;
lambda = 2*pi/(P+1);
T = 1/21; % Period T
%t_k = [0.1, 0.4];%linspace(0, 1, K); 
%a_k = [9.0505,9.0505];%rand(1, K) + 1j * rand(1, K); 
m_max = 59; % Maximum value of m
m = 0:m_max;


s_m_a = zeros(1, length(m));


for mi = 1:length(m)
    for k = 0:K-1
        s_m_a(mi) = s_m_a(mi) + a_k(k+1) * exp(1j * omega_0 * t_k(k+1) / T) * (exp(1j * lambda * t_k(k+1) / T) ^ m(mi));
    end
end


% figure;
% subplot(2, 1, 1);
% plot(m, real(s_m_a), '-o');
% title('Real part of s[m]');
% xlabel('m');
% ylabel('Real(s[m])');

% subplot(2, 1, 2);
% plot(m, imag(s_m_a), '-o');
% title('Imaginary part of s[m]');
% xlabel('m');
% ylabel('Imag(s[m])');


%disp('s[m] = ');
%disp(size(s_m));
