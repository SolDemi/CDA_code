function [pAdj, sig, info] = correct_pvalues(p, method, alpha)
%CORRECT_PVALUES Multiple-comparison correction for p values.
%
% Usage:
%   [pAdj, sig] = correct_pvalues(p)
%   [pAdj, sig] = correct_pvalues(p, method)
%   [pAdj, sig, info] = correct_pvalues(p, method, alpha)
%
% Inputs:
%   p      - numeric array of p values. NaNs are ignored and kept as NaN.
%   method - 'holm', 'fdr', 'bonferroni', or 'none'. Default: 'holm'.
%            'fdr' uses Benjamini-Hochberg FDR.
%   alpha  - significance threshold after correction. Default: 0.05.
%
% Outputs:
%   pAdj - adjusted p values, with the same size as p.
%   sig  - logical array, pAdj < alpha, with the same size as p.
%   info - struct with method, alpha, number of valid p values, and input size.
%
% Example:
%   p = [0.002 0.03 0.04 0.20 NaN];
%   [pHolm, sigHolm] = correct_pvalues(p, 'holm', 0.05);
%   [pFDR,  sigFDR ] = correct_pvalues(p, 'fdr',  0.05);
%
% Example for a table:
%   T.P_corr = correct_pvalues(T.P, 'holm');
%   T.Sig_corr = T.P_corr < 0.05;

if nargin < 2 || isempty(method)
    method = 'holm';
end

if nargin < 3 || isempty(alpha)
    alpha = 0.05;
end

origSize = size(p);
pVec = p(:);

pAdjVec = nan(size(pVec));
valid = ~isnan(pVec);
pValid = pVec(valid);

switch lower(method)
    case 'none'
        pAdjValid = pValid;

    case 'bonferroni'
        pAdjValid = min(pValid .* numel(pValid), 1);

    case 'holm'
        pAdjValid = holm_adjust(pValid);

    case {'fdr', 'bh', 'benjamini-hochberg', 'benjamini_hochberg'}
        pAdjValid = fdr_bh_adjust(pValid);

    otherwise
        error('Unknown correction method: %s. Use holm, fdr, bonferroni, or none.', method);
end

pAdjVec(valid) = pAdjValid;
pAdj = reshape(pAdjVec, origSize);
sig = pAdj < alpha;

info = struct();
info.method = lower(method);
info.alpha = alpha;
info.nValid = numel(pValid);
info.inputSize = origSize;

end

%% ========================================================================
function pAdj = holm_adjust(p)

p = p(:);
pAdj = nan(size(p));

[ps, idx] = sort(p);
m = numel(ps);

adj = (m - (1:m)' + 1) .* ps;
adj = cummax(adj);
adj = min(adj, 1);

pAdj(idx) = adj;

end

%% ========================================================================
function pAdj = fdr_bh_adjust(p)

p = p(:);
pAdj = nan(size(p));

[ps, idx] = sort(p);
m = numel(ps);

adj = ps .* m ./ (1:m)';
adj = flipud(cummin(flipud(adj)));
adj = min(adj, 1);

pAdj(idx) = adj;

end
