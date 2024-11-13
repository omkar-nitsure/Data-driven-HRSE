function [s_m_denoised] = denoise_signal(s_m_org, K, enh_method)
% -------------------------------------------------------------------------
% Communications and Signal Processing Group
% Department of Electrical and Electronic Engineering
% Imperial College London, 2020
% 
% 
% File        : denoise_signal.m
% -------------------------------------------------------------------------
% Given a signal s_m_org, number of exponentials K, and an enhancement method,
% denoise the signal.
% 
% USAGE:
%  [s_m_denoised] = denoise_signal(s_m_org, K, enh_method)
%
% INPUT:
%  - s_m_org   : sum of exponentials (original signal)
%  - K         : Number of exponentials
%  - enh_method: Enhancement method: "cadzow", "wirtinger", "none"
%
% OUTPUT:
%  - s_m_denoised : Denoised signal
%
P = length(s_m_org)-1;

enh_method_array = split(enh_method,'_');
enh_method = enh_method_array{1};
if length(enh_method_array) == 1
   maxIter = +Inf; 
else
   maxIter = str2double(enh_method_array{2});
end

%% Clean the moments
if mod(P,2) == 1
    J = floor((P+1)/2 - 1);
else
    J = floor((P+1)/2);   
end

s_m = s_m_org;
iter = 0;

switch lower(enh_method)
    case "cadzow"
        more = 1;
        while more
            %%% SVD
            A0 = toeplitz(s_m(K+1:P+1), s_m(K+1:-1:1));
            [~, S0, ~] = svd(A0, 0);
            
            %%% CADZOW ENHANCEMENT
            A = toeplitz(s_m(J+1:P+1), s_m(J+1:-1:1));
            [U, S, V] = svd(A, 0);
            S = diag(S);
            S((K+1):end) = 0;
            S = diag(S);
            A = U * S * V';
            s_m = zeros(P+1, 1);
            for p = 0:P
                s_m(p+1) = mean(diag(A, J-p));
            end
            iter = iter + 1;
            if maxIter == Inf
               more = (S0(K+1, K+1) / norm(s_m) > 1e-7);
            else
               more = (iter < maxIter);
            end
        end
    case "wirtinger"
        Ht_new = toeplitz(s_m(J+1:P+1), s_m(J+1:-1:1)); 
        Lt = zeros(size(Ht_new));
        delta1 = 0.9999;
        delta2 = 0.9999;
        more = 1;
        while more
            %%% Projected Wirtinger Gradient Descent
            Ht = Ht_new;
            
            % Step 1
            A = Lt - delta1 * (Lt - Ht);
            [U, S, V] = svd(A, 'econ');
            S = diag(S);
            S((K+1):J+1) = 0;
            S = diag(S);
            Lt = U * S * V';
            
            % Step 2
            B = Ht - delta2 * (Ht - Lt);
            s_m = zeros(P+1, 1);
            for p = 0:P
                s_m(p+1) = mean(diag(B, J-p));
            end
            Ht_new = toeplitz(s_m(J+1:P+1), s_m(J+1:-1:1));
            iter = iter + 1;
            if maxIter == Inf
               more = (norm(Ht_new - Ht, 'fro') / norm(Ht, 'fro') > 1e-4);
            else
               more = (iter < maxIter);
            end
        end
    case "cadzowup"
        Ht_new = toeplitz(s_m(J+1:P+1), s_m(J+1:-1:1));
        Tp = Ht_new;
        Tp0 = Tp;
        Sp = Ht_new;
        
        mu = 0.1;
        gamma = 0.51 * mu;
        
        allones = ones(size(Ht_new));
        w = zeros(P+1, 1);
        for p = 0:P
            w(p+1) = sum(diag(allones, J-p));
        end
        W = 1 ./ toeplitz(w(J+1:P+1), w(J+1:-1:1));

        more = 1;
        while more
            % Step 1
            A = Sp + gamma * (Tp - Sp) - mu * W .* (Tp - Tp0);
            [U, S0, V] = svd(A, 'econ');
            S = diag(S0);
            S((K+1):J+1) = 0;
            S = diag(S);
            Tp = U * S * V';
            
            % Step 2
            B = 2 * Tp - Sp;
            s_m = zeros(P+1, 1);
            for p = 0:P
                s_m(p+1) = mean(diag(B, J-p));
            end
            tmp = toeplitz(s_m(J+1:P+1), s_m(J+1:-1:1));
            Sp = Sp - Tp + tmp;

            iter = iter + 1;
            if maxIter == Inf
               more = (S0(K+1, K+1) / norm(s_m) > 1e-7);
            else
               more = (iter < maxIter);
            end
        end
        
        s_m = zeros(P+1, 1);
        for p = 0:P
            s_m(p+1) = mean(diag(Sp, J-p));
        end
    case "wirtingerapproxsvd"
        Ht_new = toeplitz(s_m(J+1:P+1), s_m(J+1:-1:1));
        Lt = zeros(size(Ht_new));
        delta1 = 0.9999;
        delta2 = 0.9999;
        more = 1;
        while more
            %%% Projected Wirtinger Gradient Descent
            Ht = Ht_new;
            
            % Step 1
            A = Lt - delta1 * (Lt - Ht);
            A = [real(A), imag(A)]; % Concatenate the real and imag parts
            [U, S, V] = svd(A, 'econ');
            S = diag(S);
            S((K+1):end) = 0;
            S = diag(S);
            Ltmp = U * S * V';
            
            Lt = Ltmp(:, 1:J+1) + 1j .* Ltmp(:, J+2:end); % Reconstruct Lt back to complex matrix
            
            % Step 2
            B = Ht - delta2 * (Ht - Lt);
            s_m = zeros(P+1, 1);
            for p = 0:P
                s_m(p+1) = mean(diag(B, J-p));
            end
            Ht_new = toeplitz(s_m(J+1:P+1), s_m(J+1:-1:1));
            iter = iter + 1;
            
            if maxIter == Inf
               more = (norm(Ht_new - Ht, 'fro') / norm(Ht, 'fro') > 1e-4);
            else
               more = (iter < maxIter);
            end
        end
    otherwise
        % Do nothing, return the original signal
        s_m = s_m_org;
end

s_m_denoised = s_m;

end
