function [out] = TV0I_L2_ADMM_SR(b,blur_k,up_factor,x,mu,rel_chg_th,itr_th,down,debug,x_true)

% This function refers to the paper:
%
%
% Compute approximate solutions of the TV0I-L2 optimization problem: 
%
% x_mu = argmin J(x;mu) ,     J(x;mu) = mu*TV0I(x) + (1/2)||SH x - b||_2^2  
%           x
%
% for the non-blind Super-Resolution, that is downsampling+denoising+deblurring (non-blind 
% means that the blur operator is assumed to be known), of grayscale images 
% using the ADMM (iterative) optimization algorithm, as illustrated in [2].
%
% [2] Min Tao and Junfeng Yang, Alternating Direction Algorithms for 
%     Total Variation Deconvolution in Image Reconstruction, TR0918, 
%     Department of Mathematics, Nanjing University, 2009.
% 
% To be noted: convergence to a minimizer is not guaranteed in this framework, however
% in [1] is shown that the process has a limit point.

% INPUTS:
    % inputs defining the TV0I-L2 cost functional to be minimized:
        % b         --> observed corrupted (blurred and noisy and downsampled) image
        % blur_k    --> blur PSF (kernel of spatial 2D convolution)
        % mu        --> regularization parameter (positive real scalar)
        % up_factor --> augmentation factor 
        % down      --> define the downsampling matrix
    % inputs defining the ADMM (iterative) algorithm:
        % x         --> initial iterate 
        % ADMM stopping criteria:
            % itr_th     --> threshold: maximum number of ADMM iterations
            % rel_chg_th --> threshold: relative change in the iterates
    % (optional) inputs defining eventual debugs
        % debug     --> flag: 0->no debug; 1->debug
        % x_true    --> original uncorrupted image
% We remark the following:
%   - all images are stored as matrices, that is they are not vectorized!
%   - periodic boundary conditions are assumed for the unknown clean image 
%     to restore, that is both the blur and the first-order derivatives 
%     near the image boundaries are computed according by FFT!

% --------------------
% initialize algorithm
% --------------------
% Tolerance for the computation of TV0I regulariser
        tol0=1e-4;
% LR and HR dimensions

        size_b = size(b);
        HR_size=size_b*up_factor;

% Langrange multiplier 
        lambda_z_h      = zeros(HR_size);
        lambda_z_v      = zeros(HR_size);
        
% Operators and constant that are common in the following steps

        % sub-problem for the primal variable t (threshold)
        % ...none!
        
        % sub-problem for the primal variable x (linear system)
        K_DFT           = psf2otf(blur_k,HR_size);
        KT_DFT          = conj(K_DFT);
        Dh_DFT          = psf2otf([1,-1],HR_size);
        Dv_DFT          = psf2otf([1;-1],HR_size);
        DhT_DFT         = conj(Dh_DFT);
        DvT_DFT         = conj(Dv_DFT);
        
        % sub-problem for the dual variable lambda_t
        % ... nothing!
        
        % initialize other necessary quantities
        Dhx = (real(ifft2(Dh_DFT.*fft2(x))));
        Dvx = (real(ifft2(Dv_DFT.*fft2(x))));

% --------------------------------------
% (eventually) compute/store/show debugs
% --------------------------------------

if ( debug == 1 )
    % compute/initialize and store iteration-based quantities
    res                 = imresize(real(ifft2( K_DFT .* fft2(x))),1/up_factor,down) - b;
    out.J_regs          = (mu/2) * nnz((abs(Dhx(:)) + abs(Dvx(:)))>tol0);
    out.J_fids          = sum( res(:).^2 );
    out.rel_chgs        = 0;
    out.res_means       = mean( res(:) );
    out.res_stdvs       = sqrt( mean( res(:).^2 ) );
    out.relerrs         = compute_rel_err(x_true,x);
    out.SSIM            = ssim(x_true,x);
    out.PSNR            = psnr(x_true,x);
    % Matlab command window debugs
    ALG_NAME = 'TV0I-L2-ADMM-SR';
    fprintf('\n\n')
    fprintf('\n-----------------------------------------------');
    fprintf('\n%s (mu = %8.5f):',ALG_NAME,mu);
    fprintf('\n-----------------------------------------------');
    fprintf('\n');
    fprintf('\n%s  it%04d:   REL-CHG = %15.13f  J = %13.8f  DISCR = %7.3f  REL-ERR = %10.7f',ALG_NAME,0,out.rel_chgs(end),out.J_fids(end) + out.J_regs(end),255*out.res_stdvs(end),out.relerrs(end));
end


% -------------------------
% carry out ADMM iterations
% -------------------------


% initialize iteration index and stopping criteria flags
itr         = 0;   
stop_flags  = [0,0];

z_h=zeros(HR_size);
z_v=zeros(HR_size);

while ( sum(stop_flags) == 0 )    
    
    % update iteration index
    itr = itr + 1;
    beta=itr^(1 + 1e-4);
    thr=((2*mu) / (beta));

    % ----------------------------------------------------------------
    % solve the ADMM sub-problem for the primal variable t and s (closed form)
    % ----------------------------------------------------------------       

    % First min-subproblem: finding s,t from a closed form  
   
            qhx = Dhx(:) + lambda_z_h(:);
            qvx = Dvx(:) + lambda_z_v(:);
            Qx  = [qhx,qvx];
            z_h=z_h(:);
            z_v=z_v(:);
            
            for i=1:length(Qx)                
                if norm(Qx(i,:),2)^2 <=  thr
                    z_h(i)=0;
                    z_v(i)=0;
                else
                    z_h(i)=Qx(i,1);
                    z_v(i)=Qx(i,2);
                end
            end
        
            z_h=reshape(z_h,HR_size);
            z_v=reshape(z_v,HR_size);
            
            %figure(1);imagesc(z_h);colormap gray; axis equal; axis off;  
            %figure(2);imagesc(z_v);colormap gray; axis equal; axis off;  
    
            % --------------------------------------------------------------------
            % solve the ADMM sub-problem for the primal variable x (linear system)
            % --------------------------------------------------------------------   
    
            x_old = x;        
                
            b1 = real(ifft2(DhT_DFT.*fft2((z_h - lambda_z_h))));
            b2 = real(ifft2(DvT_DFT.*fft2((z_v - lambda_z_v))));
            b_up = imresize(b,up_factor,down);
            b3 = (1/beta)*real(ifft2(KT_DFT.*fft2(b_up))); 

            bb = b1 + b2 + b3;  

            bb = reshape(bb,[],1); 
            
            y = pcg(@(x)linear_system(x,HR_size,K_DFT,KT_DFT,Dh_DFT,DhT_DFT,Dv_DFT,DvT_DFT,beta,up_factor,down),bb,1e-6,1000,[],[],x_old(:));
           
            x=reshape(y,HR_size);
            %x=min(1,max(x,0));  
                       
           figure(3);imagesc(min(1,max(x,0)));colormap gray; axis equal; axis off;  
        
        
    % --------------------------------------------------
    % compute x relative change (for stopping criterion)
    % --------------------------------------------------
    % gradient of the current iterate
    Dhx = (real(ifft2(Dh_DFT.*fft2(x))));
    Dvx = (real(ifft2(Dv_DFT.*fft2(x))));

    rel_chg = norm(x(:) - x_old(:))/norm(x(:));
    %rel_chg=(norm(Dhx(:)-t(:)) + norm(Dvx(:)-s(:)) )/(norm(t(:)) + norm(s(:)));

    % -----------------------
    % check stopping criteria
    % -----------------------
    if ( itr == itr_th )
        stop_flags(1) = 1;
    end
    if ( rel_chg < rel_chg_th )
        stop_flags(2) = 1;
    end
    
    % -------------------------------
    % compute quantities used for the 
    % subsequent lambdas update step
    % and for the next iteration
    % -------------------------------
    
    % ------------------------------------------------------------------
    % update dual variables (Lagrange multipliers) lambdas (dual ascent)
    % ------------------------------------------------------------------  
        % Lagrange multipliers associated with the auxiliary variable t
        lambda_z_h     = lambda_z_h - (z_h - real(ifft2(Dh_DFT.*fft2(x))));
        lambda_z_v     = lambda_z_v - (z_v - real(ifft2(Dv_DFT.*fft2(x))));
    % --------------------------------------
    % (eventually) compute/store/show debugs
    % --------------------------------------
    if ( debug == 1 )
        % compute and store iteration-based quantities
        res                     = imresize(real(ifft2( K_DFT .* fft2(x))),1/up_factor,down) - b;
        out.J_regs(end+1)       = (mu/2) * nnz((abs(Dhx(:)) + abs(Dvx(:)))>tol0);
        out.J_fids(end+1)       = sum( res(:).^2 );
        out.rel_chgs(end+1)     = rel_chg;
        out.res_means(end+1)    = mean( res(:) );
        out.res_stdvs(end+1)    = sqrt( mean( res(:).^2 ) );
        out.relerrs(end+1)      = compute_rel_err(x_true,x);
        out.SSIM(end+1)         = ssim(x_true,x);
        out.PSNR(end+1)         = psnr(x_true,x);
        % Matlab command window debugs
        fprintf('\n%s  it%04d:   REL-CHG = %15.13f  J = %13.8f  Beta = %7.3f  REL-ERR = %10.7f',ALG_NAME,itr,out.rel_chgs(end),out.J_fids(end) + out.J_regs(end),beta,out.relerrs(end));
    end

end

%Setting the output in the range [0,1]
x=min(1,max(x,0));


% one of the two stopping criteria satisfied --> end iterations and store output results
out.x       = x;
out.itr     = itr;
    
end
