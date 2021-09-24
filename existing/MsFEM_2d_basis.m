function [loc_basis] = MsFEM_2d_basis(Global_DA,nx,ny,Nx,Ny)


loc_basis = sparse((nx*Nx+1)*(ny*Ny+1),(Nx+1)*(Ny+1));

f = zeros((nx+1)*(ny+1),4);

loc_boundary1 = 1:(ny+1):(ny+1)*(nx+1);
loc_boundary2 = (ny+1):(ny+1):(ny+1)*(nx+1);
loc_boundary3 = 2:(ny);
loc_boundary4 = (nx*(ny+1)+2):(ny+1)*(nx+1)-1;
loc_boundary = [loc_boundary1,loc_boundary2,loc_boundary3,loc_boundary4];


[loc_X,loc_Y] = meshgrid(0:1/nx:1,0:1/ny:1);

f(loc_boundary,1) = (1-loc_X(loc_boundary)).*(1-loc_Y(loc_boundary));
f(loc_boundary,2) = (1-loc_X(loc_boundary)).*(  loc_Y(loc_boundary));
f(loc_boundary,3) = (  loc_X(loc_boundary)).*(1-loc_Y(loc_boundary));
f(loc_boundary,4) = (  loc_X(loc_boundary)).*(  loc_Y(loc_boundary));
idx = reshape(1:(Ny*ny+1)*(Nx*nx+1),Ny*ny+1,Nx*nx+1);

for i = 1:Nx
    for j = 1:Ny
    loc_idx = idx((j-1)*(ny)+(1:ny+1),(i-1)*(nx)+(1:nx+1));
    loc_A = Global_DA( loc_idx,loc_idx );
    loc_A(loc_boundary,:) = 0;
%     loc_A(loc_boundary,loc_boundary ) = speye(2*ny+2*nx);
    loc_A(loc_boundary+ (loc_boundary-1)*size(loc_A,1) ) = 1;
    
    loc_basis(loc_idx,[(i-1)*(Ny+1)+j, (i-1)*(Ny+1)+j+1, (i  )*(Ny+1)+j, (i  )*(Ny+1)+j+1]) = ...
        loc_A\f;
    

    end
end