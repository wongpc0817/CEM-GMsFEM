function [Global_DA,Global_M,boundary] = finematrix_2d(a,nx,ny,Nx,Ny)

hx = 1/Nx/nx;
hy = 1/Ny/ny;
ratiox = Nx*nx/Ny/ny;
ratioy = Ny*ny/Nx/nx;



idx = reshape(1:(Ny*ny+1)*(Nx*nx+1),Ny*ny+1,Nx*nx+1);

% %           boundary1
% %          -----------
% %          |         |
% %          |         |
% % boundary3|         |boundary4
% %          |         |
% %          |         |
% %          -----------
% %           boundary2

boundary1 = idx(1,:);
boundary2 = idx(Ny*ny+1,:);
boundary3 = idx(2:end-1,1);
boundary4 = idx(2:end-1,Nx*nx+1);

boundary = [boundary1,boundary2,boundary3',boundary4'];
% %[idx1, idx2;
% % idx3, idx4]

idx1 = idx(1:end-1,1:end-1);
idx2 = idx1 + Ny*ny+1;
idx3 = idx1 + 1;
idx4 = idx1 + 1 + Ny*ny+1;


Global_DA = sparse([idx1(:);idx2(:);idx3(:);idx4(:);  idx1(:);idx2(:);idx3(:);idx4(:);  idx1(:);idx2(:);idx3(:);idx4(:);  idx1(:);idx2(:);idx3(:);idx4(:)],...
                   [idx1(:);idx1(:);idx1(:);idx1(:);  idx2(:);idx2(:);idx2(:);idx2(:);  idx3(:);idx3(:);idx3(:);idx3(:);  idx4(:);idx4(:);idx4(:);idx4(:)],...
                   kron([2;-2;1;-1; -2;2;-1;1; 1;-1;2;-2; -1;1;-2;2]*ratiox+[2;1;-2;-1; 1;2;-1;-2; -2;-1;2;1; -1;-2;1;2]*ratioy, a(:) )/6 );
                         
% Global_M_a = sparse([idx1(:);idx2(:);idx3(:);idx4(:);  idx1(:);idx2(:);idx3(:);idx4(:);  idx1(:);idx2(:);idx3(:);idx4(:);  idx1(:);idx2(:);idx3(:);idx4(:)],...
%                    [idx1(:);idx1(:);idx1(:);idx1(:);  idx2(:);idx2(:);idx2(:);idx2(:);  idx3(:);idx3(:);idx3(:);idx3(:);  idx4(:);idx4(:);idx4(:);idx4(:)],...
%                    kron([1/9;1/18;1/18;1/36; 1/18;1/9;1/36;1/18;  1/18;1/36;1/9;1/18;  1/36;1/18;1/18;1/9], a(:) )*hx*hy );     
               
Global_M = sparse([idx1(:);idx2(:);idx3(:);idx4(:);  idx1(:);idx2(:);idx3(:);idx4(:);  idx1(:);idx2(:);idx3(:);idx4(:);  idx1(:);idx2(:);idx3(:);idx4(:)],...
                   [idx1(:);idx1(:);idx1(:);idx1(:);  idx2(:);idx2(:);idx2(:);idx2(:);  idx3(:);idx3(:);idx3(:);idx3(:);  idx4(:);idx4(:);idx4(:);idx4(:)],...
                   kron([1/9;1/18;1/18;1/36; 1/18;1/9;1/36;1/18;  1/18;1/36;1/9;1/18;  1/36;1/18;1/18;1/9], ones(numel(a),1) )*hx*hy );     
               