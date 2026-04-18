function ensure_eeglab_ready(varargin)
%ENSURE_EEGLAB_READY Confirm EEGLAB is available and initialize batch mode.
%
% What this helper does:
%   Checks that EEGLAB is available on the MATLAB path, initializes it in
%   no-GUI mode for batch use, and optionally confirms that ICLabel is also
%   available.
%
% When it is used:
%   Called by the public Stage-2 script that reads the cleaned Stage-1
%   EEGLAB `.set` files during parcel export.
%
% Key options:
%   - `stage_label`: short label used in plain-language error messages
%   - `require_iclabel`: whether ICLabel must also be available

p = inputParser;
p.addParameter('stage_label', 'This script', @(x) ischar(x) || isstring(x));
p.addParameter('require_iclabel', false, @(x) islogical(x) && isscalar(x));
p.parse(varargin{:});

stage_label = char(p.Results.stage_label);
require_iclabel = p.Results.require_iclabel;

if exist('eeglab', 'file') ~= 2
    error(['%s needs EEGLAB on the MATLAB path.' newline ...
           'Add the EEGLAB folder to the MATLAB path, then run this script again.'], ...
        stage_label);
end

try
    evalc('eeglab(''nogui'');');
catch ME
    error(['%s found EEGLAB on the MATLAB path, but batch initialization failed.' newline ...
           'Start EEGLAB once in MATLAB or fix the EEGLAB setup, then try again.' newline ...
           'Original error: %s'], ...
        stage_label, ME.message);
end

if exist('pop_loadset', 'file') ~= 2
    error(['%s could not find EEGLAB''s pop_loadset function after initialization.' newline ...
           'Make sure the full EEGLAB folder is on the MATLAB path.'], ...
        stage_label);
end

if require_iclabel && exist('iclabel', 'file') ~= 2
    error(['%s needs the ICLabel plugin on the MATLAB path in addition to EEGLAB.' newline ...
           'Install or add ICLabel, then run this script again.'], ...
        stage_label);
end

end
