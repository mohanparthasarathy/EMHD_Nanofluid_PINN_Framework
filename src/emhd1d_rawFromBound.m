function raw = emhd1d_rawFromBound(val, lo, hi)
%EMHD1D_RAWFROMBOUND Map a bounded value to its unconstrained logistic variable.
s = (val-lo)/(hi-lo);
s = min(max(s,1e-6),1-1e-6);
raw = log(s/(1-s));
end
