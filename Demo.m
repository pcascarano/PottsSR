%DEMO
%On the Potts functional for single-image super-resolution

%%%citare articolo per copyright


clear all;close all; clc; rng(1234);

%% Adding directories
addpath('algos')
addpath('images')
addpath('utils')

%% Test problem parameters
up_factor=4;
blur_variance=1;
noise_level=0.01;

%% Download Real Image
gt=(imread('./images/butterfly.jpg'));gt=double(rgb2gray(gt));
gt=modcrop(gt,[up_factor,up_factor]);
HRsize=size(gt);

%% Rescaling ground-truth
%Image range [0,1]
gt=gt/max(max(gt));


%% Point-Spread-Function and Blurred image
kernel = fspecial('gaussian',[7 7],blur_variance);
PSFfft=psf2otf(kernel,HRsize);
gt_fft=fft2(gt);
b=real(ifft2(PSFfft.*gt_fft));

%% Downsampled image and Noisy image

%Downsampling
down='lanczos2';
b=imresize(b,1/up_factor,down);
%AWGN
eta=randn(size(b));
b=b + noise_level*eta;

%% Algorithm parameters
mu=1e-3; 
rel_chg_th=1e-4;
itr_th=300;

for i=1:length(mu)

    x=imresize(b,up_factor,down);

    %TV0_L2 
    %[out] = TV0I_L2_ADMM_SR(b, kernel,up_factor,x,mu(i),rel_chg_th,itr_th,down,1,gt);
    %name_HR=['TV0_iso'];

    %TV0Anisotrope + L2 
    [out] = TV0A_L2_ADMM_SR(b,kernel,up_factor,x,mu(i),rel_chg_th,itr_th,down,1,gt);
    name_HR=['TV0_anis'];      
    
end

%Saving in png
HR=(out.x)*255;
HR=uint8(HR);
noisestr=num2str(noise_level);
name=[name_HR,'_',noisestr,'.png'];
imwrite(HR,name);

