function val = emhd1d_boundFromRaw(raw, lo, hi)
%EMHD1D_BOUNDFROMRAW Map an unconstrained variable to a bounded interval.
s = 1./(1+exp(-raw));
val = lo + (hi-lo).*s;
end
