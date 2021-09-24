function [f] = form_Source(F,nx,ny,Nx,Ny)


hx = 1/Nx/nx;
hy = 1/Ny/ny;


idx = reshape(1:(Ny*ny+1)*(Nx*nx+1),(Ny*ny+1),(Nx*nx+1));


% %[idx1, idx2;
% % idx3, idx4]

idx1 = idx(1:end-1,1:end-1);
idx2 = idx1 + (Ny*ny+1);
idx3 = idx1 + 1;
idx4 = idx1 + 1 + (Ny*ny+1);


f = sparse([idx1(:);idx2(:);idx3(:);idx4(:)],...
            ones(4*Nx*nx*Ny*ny,1),...
            kron([1;1;1;1], F(:) )/4*hx*hy );

