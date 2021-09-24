function [loc_basis,min_d] = eigenproblem_fouce_2d_delta_penal(a,nx,ny,Nx,Ny,num_basis,over_size,Global_A_dg,Global_DA,Global_M_dg,Global_M_cdg)

dispstat('','init');
hx = 1/Nx/nx;
hy = 1/Ny/ny;
ratiox = Nx*nx/Ny/ny;
ratioy = Ny*ny/Nx/nx;

loc_basis_V1((Nx-1)*(Ny-1)*(2*ny+1)*(2*nx+1)*num_basis) = 0;
loc_basis_V2((Nx-1)*(Ny-1)*(2*ny+1)*(2*nx+1)*num_basis) = 0;
loc_basis_V3((Nx-1)*(Ny-1)*(2*ny+1)*(2*nx+1)*num_basis) = 0;


idx = reshape(1:(2*(ny+over_size)+1)*(2*(nx+over_size)+1),2*(ny+over_size)+1,2*(nx+over_size)+1);
idx1 = idx(1:end-1,1:end-1);
idx2 = idx1 + 2*(ny+over_size)+1;
idx3 = idx1 + 1;
idx4 = idx1 + 1 + 2*(ny+over_size)+1;
loc_M_idx1 = [idx1(:);idx2(:);idx3(:);idx4(:);  idx1(:);idx2(:);idx3(:);idx4(:);  idx1(:);idx2(:);idx3(:);idx4(:);  idx1(:);idx2(:);idx3(:);idx4(:)];
loc_M_idx2 = [idx1(:);idx1(:);idx1(:);idx1(:);  idx2(:);idx2(:);idx2(:);idx2(:);  idx3(:);idx3(:);idx3(:);idx3(:);  idx4(:);idx4(:);idx4(:);idx4(:)];

idx = reshape(1:(Ny*ny+1)*(Nx*nx+1),Ny*ny+1,Nx*nx+1);
count = 0;

loc_boundary = [idx(:,1);idx(end,2:end)';idx((end-1):-1:1,end);idx(1,(end-1):-1:2)'];
int_idx = idx(:);
int_idx(loc_boundary) = [];
basis_count = 0;


min_d = inf;
idx_dg = reshape(1:Ny*(ny+1)*Nx*(nx+1),Ny*(ny+1),Nx*(nx+1));
%% main loop for solving eigenproblem
tic
V_tilde = zeros((nx+1)*(ny+1),num_basis,Nx,Ny);
loc_basis_V1_dg((Nx)*(Ny)*(ny+1)*(nx+1)*num_basis) = 0;
loc_basis_V2_dg((Nx)*(Ny)*(ny+1)*(nx+1)*num_basis) = 0;
loc_basis_V3_dg((Nx)*(Ny)*(ny+1)*(nx+1)*num_basis) = 0;
for i = 1:Nx
    for j = 1:Ny
%forming the local matrix
        loc_idx = idx_dg((j-1)*(ny+1) +(1:ny+1),(i-1)*(nx+1) +(1:nx+1));
        
        loc_M_a =  Global_M_dg(loc_idx(:),loc_idx(:));
        loc_A   =  Global_A_dg(loc_idx(:),loc_idx(:));
        [v,d] = eigs(loc_A,loc_M_a,num_basis+1,-1e-8);
        min_d = min(max(diag(d)),min_d);
        [v,d] = eigs(loc_A,loc_M_a,num_basis,-1e-8);
%         min_d = min(max(diag(d)),min_d);
        mass_s = diag(v'*loc_M_a*v);
        v = v*sparse(1:num_basis,1:num_basis,1./sqrt(mass_s));
        V_tilde(:,:,i,j) = v;
        
        loc_basis_V1_dg( basis_count + (1:(ny+1)*(nx+1)*num_basis) ) = ...
            repmat(loc_idx(:),[num_basis,1]);
        loc_basis_V2_dg( basis_count + (1:(ny+1)*(nx+1)*num_basis) ) = ...
            kron( count +(1:num_basis) , ones((ny+1)*(nx+1),1) );
        loc_basis_V3_dg( basis_count + (1:(ny+1)*(nx+1)*num_basis) ) = ...
            reshape(v,(ny+1)*(nx+1)*num_basis,1);
        count = count + num_basis;
        basis_count = basis_count + (ny+1)*(nx+1)*num_basis;
    end
end
basis_dg = sparse(loc_basis_V1_dg,loc_basis_V2_dg,loc_basis_V3_dg);
GMs_M_cdg = Global_M_cdg*basis_dg;
GMs_M_ccg = GMs_M_cdg*GMs_M_cdg';
basis_count = 0;
count = 0;
for i = 1:Nx
    for j = 1:Ny
%forming the local matrix

        
        loc_idx_idx_y = ((j-1)*(ny)+(1:(ny+2*over_size)+1))-over_size;
        loc_idx_idx_x = ((i-1)*(nx)+(1:(nx+2*over_size)+1))-over_size;
        non_used_idx_y = find((loc_idx_idx_y<1) + (loc_idx_idx_y>(Ny*ny+1)));
        non_used_idx_x = find((loc_idx_idx_x<1) + (loc_idx_idx_x>(Nx*nx+1)));
        loc_idx_idx_y(non_used_idx_y) = [];
        loc_idx_idx_x(non_used_idx_x) = [];
        
        loc_size_y = numel(loc_idx_idx_y);
        loc_size_x = numel(loc_idx_idx_x);
        
        
        idx0 = reshape(1:(loc_size_y)*(loc_size_x),loc_size_y,loc_size_x);

        loc_boundary = [idx0(:,1);idx0(end,2:end)';idx0((end-1):-1:1,end);idx0(1,(end-1):-1:2)'];
        int_idx = idx0(:);
        int_idx(loc_boundary) = [];
        
        loc_idx = idx(loc_idx_idx_y,loc_idx_idx_x);
        loc_idx = loc_idx(int_idx);
        
        loc_DA = Global_DA(loc_idx(:),loc_idx(:));
        loc_M_a_cdg = GMs_M_cdg(loc_idx(:),:);

        loc_full_A = loc_DA + GMs_M_ccg(loc_idx(:),loc_idx(:));
        loc_full_f = loc_M_a_cdg*[sparse(count + (1:num_basis),(1:num_basis),ones(1,num_basis),num_basis*Ny*Nx,num_basis)];
        full_v = loc_full_A\loc_full_f;
        
        v = full_v(1:size(loc_DA,1),:);

        
        loc_basis_V1(basis_count + (1:(loc_size_y-2)*(loc_size_x-2)*num_basis) ) = ...
            repmat(loc_idx,[num_basis,1]);
        loc_basis_V2( basis_count + (1:(loc_size_y-2)*(loc_size_x-2)*num_basis) ) = ...
            kron( count +(1:num_basis) , ones((loc_size_y-2)*(loc_size_x-2),1) );
        loc_basis_V3( basis_count + (1:(loc_size_y-2)*(loc_size_x-2)*num_basis) ) = ...
            reshape(v,(loc_size_y-2)*(loc_size_x-2)*num_basis,1);
        count = count + num_basis;
        basis_count = basis_count + (loc_size_y-2)*(loc_size_x-2)*num_basis;
            
    end
    dispstat( sprintf('Finished %.2f percent. Estimated time left %.2fs.',i/(Nx-1)*100,toc*(Nx-1-i)/i) );
end

loc_basis = sparse(loc_basis_V1,loc_basis_V2,loc_basis_V3,(ny*Ny+1)*(nx*Nx+1),num_basis*(Ny)*(Nx));

dispstat( 'Finished solving GMS basis' );
% min_d
