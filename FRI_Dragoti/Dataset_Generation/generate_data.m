% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%% Generating FRI Signals Data (Main Routine) %%%%%%%%%%%%%%%%%%%
clc
close all
clear all

%% Initialise User Input Parameters
N             = 21;             % number of temporal samples
kernel        = "eMOMS";
P             = N-1;            % Order of sampling kernel

% kernel        = "ESpline";
% P             = 6;     % Number of exponentials


K             = 2;              % Number of Diracs, can be a range --> randomly picked between that
T             = 1/N;            % sampling period
resolution    = 64;             % how many points of phi within a sampling period T
samp_mode     = "anticausal";   % 'peak', 'causal', 'anticausal'
PERIODIC      = 0;              % 1 if the signal is periodic
m_max = 20; % Maximum value of m
m1 = 0:m_max;
% since FRIED-Net does not require c_mn or h or s[m], we can choose whether
% to save them in the dataset
% "": don't save any of c_mn or h or s[m], 
% "c": save c_mn and h
% "s": save s[m], \tilde{s}[m] and h
% "all": save c_mn, s[m], \tilde{s}[m], and h
SAVE_FREQ = "all";              


% Data settings
output_dir    = "/Users/sampath/Downloads/LearningBasedFRI-main 2/datasets/dataset_21";
out_filename_list =  ["train_%sdB_test.h5"];%,"test_%sdB.h5"];
num_data_list = [1000];%, 1];
PSNR_range = 70:-5:70;          % If empty, don't add noise

%%
T_s           = T/resolution;    % "continuous" time resolution

if isscalar(K)
    Kmax = K;
else
    Kmax = max(K);
end

m = 0:P;
if strcmp(kernel,'eMOMS')
    [phi,t_phi] = eMOMS(P, resolution);
    alpha_0 = - 1j * pi/ (P+1) *P ;
    %alpha_0 =0;
    lambda  = 2j*  pi / (P+1) ;
    alpha_vec    = alpha_0 + lambda * m;
elseif strcmp(kernel,'ESpline')
    L = 3.5*(P+1);
    alpha_0 = - 1j * pi/ L *P ;
    lambda  = 2* 1j * pi / L ;
    alpha_vec    = alpha_0 + lambda * m;
    [phi,t_phi] = ESpline(alpha_vec, resolution);
else
    error("Wrong Kernel Selected")
end

% change phi according to different sampling mode
if strcmp(samp_mode,'peak')
    [~,idx] = max(abs(phi));
    t_phi = t_phi - t_phi(idx);
elseif strcmp(samp_mode, 'causal')
    phi = phi(end:-1:1);
    t_phi   = -t_phi(end:-1:1);
elseif ~strcmp(samp_mode, 'anticausal')
    error('Wrong sampling mode selected')
end

%% Plot sampling filter h(t) = phi(-t/T)
figure;
plot(-t_phi*T, phi);    % From sampling kernel to sampling filter, t axis change
title('Impulse response $h(t) = \varphi(-t/T)$','Interpreter','latex')

%% Find c_mn 
mkdir(output_dir)

if SAVE_FREQ ~= ""
    [n_vec,~,~]=n_vec(N);
    c_mn = get_c_mn_exp(alpha_vec, n_vec, phi, t_phi);
    if SAVE_FREQ == "c" || SAVE_FREQ == "all"
        save(fullfile(output_dir, "c_mn.mat"), "c_mn")
    end
end

%% Main Loop
for m = 1:length(out_filename_list)
    num_data = num_data_list(m);

    y_n = zeros(num_data,N);
    t_k = zeros(num_data,Kmax);
    a_k = zeros(num_data,Kmax);
    if SAVE_FREQ ~= ""
        s_m = zeros(num_data, P+1);
        s_m_a = zeros(num_data, 60);%always give no.of samples value
        h = zeros(num_data, K+1);
    end

    % Generate the ground truth (noiseless) t_k, a_k, y[n]
    for i =1:num_data 
        [t_k(i,:),a_k(i,:),y_n(i,:),s_m_a(i,:)] = generate_data_single143(N,K,T,T_s,phi,samp_mode,PERIODIC);

        if SAVE_FREQ ~= ""
            s_m(i,:) = y_n(i,:) * c_mn.';
            S_toe = toeplitz(s_m(i,K+1:end), s_m(i,K+1:-1:1));
            [~,~,V]=svd(S_toe);
            h(i,:) = V(:,end);
        end
%         figure;
%         %subplot(2, 1, 1);
%         plot(m1, real(s_m),'r');
%         title('Real part of s[m] dragoti dataset');
%         xlabel('m');
%         ylabel('Real(s[m])');

%         subplot(2, 1, 2);
%         plot(m1, real(s_m_noisy),'r');
%         title('Imaginary part of s[m]  dragoti dataset');
%         xlabel('m');
%         ylabel('Imag(s[m])');
%     disp(size(s_m))
%           
    end
%     figure;
%     subplot(2, 1, 1);
%     plot(m1, real(s_m),'r');
%     title('Real part of s[m] dragoti dataset');
%     xlabel('m');
%     ylabel('Real(s[m])');
% 
%     subplot(2, 1, 2);
%     plot(m1, imag(s_m),'r');
%     title('Imaginary part of s[m]  dragoti dataset');
%     xlabel('m');
%     ylabel('Imag(s[m])');
%     disp(size(s_m))
%           


    j = 1;
    while exist("PSNR_range", "var")
        if isempty(PSNR_range)
            PSNR = [];
        else
            PSNR = PSNR_range(j);
        end

        savepath = fullfile(output_dir, sprintf(out_filename_list(m), num2str(PSNR)));

        % writing to HDF5
        %h5create(savepath,'/t_k',[Kmax num_data],'Datatype','single')
        %h5write(savepath,'/t_k',single(t_k).')
        %h5create(savepath,'/a_k',[Kmax num_data],'Datatype','single')
        %h5write(savepath,'/a_k',single(a_k).')
        h5create(savepath,'/y_n',[N num_data],'Datatype','single')
        h5write(savepath,'/y_n',single(y_n).')
        %if SAVE_FREQ ~= ""
            %if SAVE_FREQ == "s" || SAVE_FREQ == "all"
                h5create(savepath,'/s_m_real', [N num_data], 'Datatype', 'double');
                h5create(savepath, '/s_m_imag', [N num_data], 'Datatype', 'double');
                h5write(savepath, '/s_m_real', real(s_m).');
                h5write(savepath, '/s_m_imag', imag(s_m).');
                %h5write(savepath,'/s_m',single([real(s_m) + 1j*imag(s_m)]).')
            %end
            %h5create(savepath,'/h',[2*(K+1) num_data],'Datatype','single')
            %h5write(savepath,'/h',single([real(h), imag(h)]).')
        %end

        if isempty(PSNR_range)
            h5disp(savepath);
            break
        else
            %y_n_noisy = awgnPSNR(y_n, PSNR, a_k);
            y_n_noisy = awgn(y_n,15,'measured');
            h5create(savepath,'/y_n_noisy',[N num_data],'Datatype','single')
            h5write(savepath,'/y_n_noisy',single(y_n_noisy).')
            if SAVE_FREQ == "s" || SAVE_FREQ == "all"
                s_m_noisy = y_n_noisy * c_mn.';
                s_m_denoisy=denoise_signal1(s_m_noisy,K, "cadzow");
                %h5create(savepath,'/s_m_noisy',[2*(P+1) num_data],'Datatype','single')
                %h5write(savepath,'/s_m_noisy',single([real(s_m_noisy), imag(s_m_noisy)]).')
                h5create(savepath,'/s_m_noisy_real', [N num_data], 'Datatype', 'double');
                h5create(savepath, '/s_m_noisy_imag', [N num_data], 'Datatype', 'double');
                h5write(savepath, '/s_m_noisy_real', real(s_m_noisy).');
                h5write(savepath, '/s_m_noisy_imag', imag(s_m_noisy).');

                h5create(savepath,'/s_m_denoisy_real', [N num_data], 'Datatype', 'double');
                h5create(savepath, '/s_m_denoisy_imag', [N num_data], 'Datatype', 'double');
                h5write(savepath, '/s_m_denoisy_real', real(s_m_denoisy).');
                h5write(savepath, '/s_m_denoisy_imag', imag(s_m_denoisy).');
                h5create(savepath,'/s_m_a_real', [60 num_data], 'Datatype', 'double');
                h5create(savepath, '/s_m_a_imag', [60 num_data], 'Datatype', 'double');
                h5write(savepath, '/s_m_a_real', real(s_m_a).');
                h5write(savepath, '/s_m_a_imag', imag(s_m_a).');
            end
           
            h5disp(savepath);

%             figure;
%             %subplot(2, 1, 2);
%             plot(m1, real(s_m_noisy),'r');
%             title('noisy Imaginary part of s[m]  dragoti dataset');
%             xlabel('m');
%             ylabel('Imag(s[m])');
%  

            if j < length(PSNR_range)
                j = j + 1;
            else
                break
            end
        end
    end
end