function [loc_basis] = bilinear_partition(nx,ny,Nx,Ny)

loc_basis = sparse((nx*Nx+1)*(ny*Ny+1),(Nx+1)*(Ny+1));

[loc_X,loc_Y] = meshgrid(0:1/nx:1,0:1/ny:1);

idx = reshape(1:(Ny*ny+1)*(Nx*nx+1),Ny*ny+1,Nx*nx+1);

for i = 1:Nx
    for j = 1:Ny
    loc_idx = idx((j-1)*(ny)+(1:ny+1),(i-1)*(nx)+(1:nx+1));
    
    loc_basis(loc_idx,[(i-1)*(Ny+1)+j, (i-1)*(Ny+1)+j+1, (i  )*(Ny+1)+j, (i  )*(Ny+1)+j+1]) = ...
        [(1-loc_X(:)).*(1-loc_Y(:)),(1-loc_X(:)).*(  loc_Y(:)), (loc_X(:)).*(1-loc_Y(:)), (  loc_X(:)).*(  loc_Y(:))];
    end
end