% This program use SIPDG method to solve the elliptic equation 
%     
%      - \nabla (a \nabla u) = f
%     
%
%=========================================================
%% initialize
%=========================================================
clear

%=========================================================
%% input parameter
%=========================================================

for ii = [2]
compute_partition = 1; % 0 - load multiscale partition, 1 - solve multiscale partition,  2 - bilinear
compute_snapshot = 1; %  0 - load snapshot space, 1 - compute snapshot space,       2 - snapshot space = V_h
compute_MS_basis = 1; %  0 - load MS basis, 1 - compute MS basis
plot_solution_fine = 1; % 1 - plot fine solution, 0 - do not plot fine solution
plot_solution_MS = 1; % 1 - plot GMS solution, 0 - do not plot GMS solution
solve_fine = 1; % 1 - solve fine solution, 0 - do not solve fine solution
solve_MS   = 1;  % 1 - solve MS solution, 0 - do not solve MS solution
compute_error = 1; % 1 - compute error, 0 - do not compute error

% Input the num of Corase block , num of fine block in one Corase block 
Nx=5; Ny=Nx;
nx=5; ny=nx;

% Input the num of multiscale basis used
num_basis = ii;
over_size = ceil((log(1/Nx)/log(1/10))*3)*ny;
% over_size = 3*ny;
num_rand = 30;
%=========================================================
%% input medium
%=========================================================

load data_a % (ny*Ny,nx*Nx)- matrix
a = a3;


% load channel_a_test
% a(:) = 1;

% load test_large_a
% load data_a_box a
% a(a==1e+4) = 1e+8;
% a(a==1e+6) = 1e+13;
[X,Y] = meshgrid(1/size(a,2)/2:1/size(a,2):1,1/size(a,1)/2:1/size(a,1):1);
[X1,Y1] = meshgrid(1/nx/Nx/2:1/nx/Nx:1,1/ny/Ny/2:1/ny/Ny:1);
a = interp2(X,Y,a,X1,Y1,'nearest');

% a(Ny*ny/2:Ny*ny/2+1,2*nx:(Nx-2)*nx) = 1e+5;
% 
% epsilon = (1e-1)/3;
% source_a = @(x,y) 1.1 + sin(pi*x./(1+x)/epsilon  ).*sin(pi*(1+y)./(2-y)/epsilon);
% a = source_a(X1,Y1);
% source_a = @(x,y) 1 + heaviside(y-.2).*heaviside(0.8-y).*exp(-(x-0.5).^2/epsilon^2)/epsilon^2*1000;
% source_a = @(x,y) 1 + exp(-(x-0.5).^2/epsilon^2)/epsilon^2*1000;
% a = source_a(X1,Y1);
% load SPE10_proj
%=========================================================
%% forming matrix
%=========================================================

disp('Forming fine-scale matrix')
% Global_DA is the fine-scale stiffness martix 
% Global_M  is the fine-scale mass martix
% boundary  is the boundary index

[Global_DA,Global_M,boundary] = finematrix_2d(a,nx,ny,Nx,Ny);

%=========================================================
%% forming RHS
%=========================================================
% input a source function (fun_F) in this part (or you can directly input a vector (F))

disp('Forming RHS')
fun_F = @(x,y) heaviside(0.8-y).*heaviside(y-0.2).*heaviside(0.12-x).*heaviside(x-0.1)...
    +heaviside(0.3-y).*heaviside(y-0.25).*heaviside(0.45-x).*heaviside(x-0.40);
% % 
% % fun_F = @(x,y) heaviside(1-y).*heaviside(y-0.95).*heaviside(0.05-x).*heaviside(x)...
% %     +heaviside(.05-y).*heaviside(y).*heaviside(1-x).*heaviside(x-0.95);
% fun_F = @(x,y) 2*pi^2*sin(pi*x).*sin(pi*y);
% fun_F = @(x,y) 1 + 0*x;
% fun_F = @(x,y)  x.^4;
% fun_F = @(x,y)1./(hH)^2*exp(-((x-.1).^2+(y-.5).^2)/(2/nx/Nx)^2);


[X,Y] = meshgrid(1/nx/Nx/2:1/nx/Nx:1,1/ny/Ny/2:1/ny/Ny:1);
F = fun_F(X,Y);
f = full(form_Source(F,nx,ny,Nx,Ny));


%=========================================================
%% forming MS basis
%=========================================================
if compute_partition == 1
    disp('Forming Partition of unity')
    [loc_basis] = MsFEM_2d_basis(Global_DA,nx,ny,Nx,Ny);
% %     save basis_data_MsFEM loc_basis
elseif compute_partition == 0
    load basis_data_MsFEM loc_basis
else
    [loc_basis] = bilinear_partition(nx,ny,Nx,Ny);
    
end
MS_loc_basis = loc_basis;


[Global_A_dg,Global_A1,Global_JA,Global_DBC,Global_M_dg,Boundarya,Boundaryb,Boundaryc,Boundaryd] = finematrix_DBC(a,a,nx,ny,Nx,Ny,MS_loc_basis);
Global_M_cdg = Global_M_dg;
Global_M_cdg(Boundarya(:),:)=Global_M_cdg(Boundarya(:),:)+Global_M_cdg(Boundaryb(:),:);
Global_M_cdg(Boundaryc(:),:)=Global_M_cdg(Boundaryc(:),:)+Global_M_cdg(Boundaryd(:),:);
Global_M_cdg([Boundaryb(:);Boundaryd(:)],:)=[];


if compute_MS_basis
    disp('Forming MS basis')
%     [loc_basis,min_d] = eigenproblem_fouce_2d_delta_4(a,nx,ny,Nx,Ny,num_basis,over_size,Global_A_dg,Global_DA,Global_M_dg,Global_M_cdg);
    [loc_basis,min_d] = eigenproblem_fouce_2d_delta_penal(a,nx,ny,Nx,Ny,num_basis,over_size,Global_A_dg,Global_DA,Global_M_dg,Global_M_cdg);
    
    num_basis_saved = num_basis;
    
else
    
    load basis_data_GMsFEM loc_basis num_basis_saved
    
    
end

Global_M_cdg = Global_M_dg;
Global_M_cdg(Boundarya(:),:)=Global_M_cdg(Boundarya(:),:)+Global_M_cdg(Boundaryb(:),:);
Global_M_cdg(Boundaryc(:),:)=Global_M_cdg(Boundaryc(:),:)+Global_M_cdg(Boundaryd(:),:);
Global_M_cdg([Boundaryb(:);Boundaryd(:)],:)=[];

Global_M_cdg(:,Boundarya(:))=Global_M_cdg(:,Boundarya(:))+Global_M_cdg(:,Boundaryb(:));
Global_M_cdg(:,Boundaryc(:))=Global_M_cdg(:,Boundaryc(:))+Global_M_cdg(:,Boundaryd(:));
Global_M_cdg(:,[Boundaryb(:);Boundaryd(:)])=[];

% loc_basis = real(loc_basis);
%=========================================================
%% forming MS Matrix
%=========================================================
disp('Forming MS Matrix')

interior_idx_fine = 1:(nx*Nx+1)*(ny*Ny+1);
interior_idx_fine(boundary) = [];

interior_idx_coarse = 1:(Nx+1)*(Ny+1);
interior_idx_coarse([1:Ny+1:(Nx+1)*(Ny+1), Ny+1:Ny+1:(Nx+1)*(Ny+1),  2:Ny  ,(Ny+1)*Nx + (2:Ny)]) = [];

Global_DA = Global_DA(interior_idx_fine,interior_idx_fine);
Global_M = Global_M(interior_idx_fine,interior_idx_fine);
Global_M_cdg = Global_M_cdg(interior_idx_fine,interior_idx_fine);
f(boundary) = [];
loc_basis = loc_basis(interior_idx_fine,:);


GMS_A = loc_basis'*Global_DA*loc_basis;
GMS_f = loc_basis'*f;

%=========================================================
%% solving GMS solution
%=========================================================
if solve_fine
    disp('Solving fine solution')

    Global_U = Global_DA\f;
end

disp('Solving GMS solution')

GMS_CU = GMS_A\GMS_f;
GMS_U = loc_basis*GMS_CU;


if solve_MS 
%=========================================================
%% solving MS solution
%=========================================================
MS_loc_basis = MS_loc_basis(interior_idx_fine,interior_idx_coarse);
MS_A = MS_loc_basis'*Global_DA*MS_loc_basis;
MS_f = MS_loc_basis'*f;
disp('Solving MS solution')

MS_CU = MS_A\MS_f;
MS_U = MS_loc_basis*MS_CU;

end
%=========================================================
%% visualization
%=========================================================
GMS_sol = reshape(GMS_U,ny*Ny-1,nx*Nx-1);
Global_sol = reshape(Global_U,ny*Ny-1,nx*Nx-1);

if plot_solution_fine
figure(50); imagesc(Global_sol)
end
if plot_solution_MS
    if solve_MS
        MS_sol = reshape(MS_U,ny*Ny-1,nx*Nx-1);
        figure(51); imagesc(MS_sol)
    end
figure(52); imagesc(GMS_sol)

end
%=========================================================
%% compute error
%=========================================================
if compute_error
    disp(' ')
    DG_error = sqrt(((GMS_U(:)-Global_U(:))'*Global_DA*(GMS_U(:)-Global_U(:)))/((Global_U(:))'*Global_DA*(Global_U(:))));
    L2_error = sqrt(((GMS_U(:)-Global_U(:))'*Global_M*(GMS_U(:)-Global_U(:)))/((Global_U(:))'*Global_M*(Global_U(:))));
    fprintf('The relative Energy Error for GMsFEM is %2.2f%%. \n',DG_error*100)
    fprintf('The relative L2 Error for GMsFEM is     %2.2f%%. \n',L2_error*100)
    disp(' ')
    min_d_ii(ii) = min_d;
    DG_error_ii(ii) = DG_error;
    L2_error_ii(ii) = L2_error;
    if solve_MS
        DG_error = sqrt(((MS_U(:)-Global_U(:))'*Global_DA*(MS_U(:)-Global_U(:)))/((Global_U(:))'*Global_DA*(Global_U(:))));
        L2_error = sqrt(((MS_U(:)-Global_U(:))'*Global_M*(MS_U(:)-Global_U(:)))/((Global_U(:))'*Global_M*(Global_U(:))));
        fprintf('The relative Energy Error for MsFEM is  %2.2f%%. \n',DG_error*100)
        fprintf('The relative L2 Error for MsFEM is      %2.2f%%. \n',L2_error*100)
        disp(' ')
    end
end


end
%=========================================================