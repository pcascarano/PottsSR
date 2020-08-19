function z = linear_system(x,HR_size,K_DFT,KT_DFT,Dh_DFT,DhT_DFT,Dv_DFT,DvT_DFT,beta_t,up_factor,down)

%For TV0A we suppose beta_t=beta_s, then...

    x=reshape(x,HR_size);

    %termine DhT*Dh*x
    x=fft2(x);
    Dhx = Dh_DFT.*x;
    y1 = real(ifft2(DhT_DFT.*Dhx));

    %termine DvT*dv*x
    Dvx = Dv_DFT.*x;
    y2 = real(ifft2(DvT_DFT.*Dvx));

    %(1/beta)*KT*K*x
    Hx = real(ifft2(K_DFT.*x));
    SHx = imresize(Hx,1/(up_factor),down);
    STSHx=imresize(SHx,up_factor,down);   
    STSHx_fft = fft2(STSHx);
    y3 = (1/beta_t)*real(ifft2(KT_DFT.*STSHx_fft));

    z = y1 + y2 + y3 ; 
    
    z = reshape(z,[],1);
end