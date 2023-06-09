

def goi_gradients(ts,maxorder,biascorrect):
    """
% Compute low order O-infomation descriptor of data
% INPUT
%       ts          input (observations x variables), time series or static/behavioral data
%       maxorder    maximum order for the gradients  
%       biascorrect apply or not bias correction for entropy calculation
% OUTPUT
%       goi_val     gradients of O-information
"""
goi   = struct;
if nargin < 3:
    biascorrect = 'true'
    

[N, nvartot] = size(ts);

for i in range( 1,nvartot):
    X[:,i] = goi_copnorm(ts[:,i]) 


for isize in range( 1,maxorder): # % Loop over gradient orders
    goi_val.clear
    C     = nchoosek[1:nvartot,isize]# % List all the combinations
    ncomb = size(C,1);
    subset_index = dec2bin[0:2^isize-1] - '0';# % List di index of every subset
    
    for icomb in range(1,ncomb) :#  % parfor
        goi_val[icomb] = 0;
        c = C[icomb,:]
        for n in range(1,height(subset_index)):    # % TO DO: this loop is quite inefficient as 'goi_oinfo' computes many times the same quantities
            idx = c(find(subset_index[n,:]))
            goi_val[icomb] = goi_val[icomb] + (-1)^numel(idx)*goi_oinfo(X[:,setdiff[1:nvartot,idx ]]);
        end
    end

    goi(isize).O_val  = goi_val;
    goi(isize).index_var   = C;
return goi 