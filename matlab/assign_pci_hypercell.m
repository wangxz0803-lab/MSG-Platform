function bs_pcis = assign_pci_hypercell(num_sites, sectors, hypercell_size)
%ASSIGN_PCI_HYPERCELL  HyperCell PCI assignment for linear topology.
%
%   Consecutive *hypercell_size* sites share the same PCI group so that
%   a moving UE does not trigger handover within a HyperCell.
%
%   Formula: PCI = 3 * (group_index mod 168) + sector_id
%   where group_index = floor(site_id / hypercell_size)
%
%   Inputs:
%       num_sites      : number of physical sites
%       sectors        : 1 or 3
%       hypercell_size : sites per HyperCell group (>=1)
%
%   Output:
%       bs_pcis : [1 x K] integer PCIs  (K = num_sites * sectors)

    if nargin < 3 || isempty(hypercell_size); hypercell_size = 1; end

    K = num_sites * sectors;
    bs_pcis = zeros(1, K);
    idx = 0;
    for i = 0:(num_sites - 1)
        group_idx = floor(i / hypercell_size);
        for s = 0:(sectors - 1)
            idx = idx + 1;
            bs_pcis(idx) = 3 * mod(group_idx, 168) + s;
        end
    end
end
