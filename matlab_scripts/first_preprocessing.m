%% Soft-coded paths

BASE_DIR = "/Users/tizianocausin/livingstone_lab_local";

data_formatted   = fullfile(BASE_DIR, 'Data-Formatted');
data_neuropixel  = fullfile(BASE_DIR, 'Data-Neuropixels-Preprocessed');

fig_dir_base = '/Users/tizianocausin/Desktop'; %'/n/data2/hms/neurobio/livingstone/marge/figimages';


%% Parameters
bkgwindow=5:35;
evkwindow=90:470;
Spikes.raster_window  = [1 3500];
raster_window  = [1 3500];
     rasterlength=1000;

chanpos_exp_name = 'baby1_260630';  % experiment used only for channel geometry
exp_name = 'temp';                  % output figure folder/name

exp_names = [
    % 'baby1_260713'
    'baby1_260715'
];

window_length = diff(Spikes.raster_window) + 1;

ii = 0;
% rasters = zeros(383, window_length, 4, 'single');


%% Load MUA and extract stimulus-aligned rasters

for indx = 1:size(exp_names, 1)

    curr_exp = strtrim(exp_names(indx, :));

    % Load stimulus/trial metadata.
    exp_path = fullfile(data_formatted, [curr_exp, '_experiment.mat']);
    load(exp_path)

    % Load continuous MUA, expected shape: channels x time_ms.
    mua_path = fullfile( ...
        data_neuropixel, curr_exp, ...
        ['catgt_', curr_exp, '_g0'], ...
        [curr_exp, '_g0_imec0'], ...
        [curr_exp, '-imec0-mua_cont.h5'] ...
    );

    mua = h5read(mua_path, '/mua_cont');

    % Quick sanity check of global MUA timecourse.
    % figure;
    % plot(nanmean(mua, 1))

    n_presentations = length(Stimuli);

    for i = 1:n_presentations

        ii = ii + 1;

        % Convert stimulus onset + raster offset into sample index.
        window = round(Stimuli(i).start_time + Spikes.raster_window(1));

        % Extract channels x time window around this stimulus.
        rasters(:, :, ii) = mua(:, window:window + window_length - 1);

        % Baseline-subtract each channel using the first 40 time bins.
        baseline = nanmean(rasters(:, 1:40, ii), 2);
        rasters(:, :, ii) = rasters(:, :, ii) - repmat(baseline, [1, size(rasters, 2)]);

        % Store stimulus metadata.
        allimages{ii} = Stimuli(i).filename;
        stim_xy(ii, :) = Stimuli(i).position;

    end

    clear Trials mua %Stimuli

end


%% Load channel positions and order channels by anatomical depth

% Use a fixed/reference experiment for channel geometry.
chposfile = fullfile( ...
    data_neuropixel, chanpos_exp_name, ...
    ['catgt_', chanpos_exp_name, '_g0'], ...
    [chanpos_exp_name, '_g0_imec0'], ...
    'channel_positions.mat' ...
);

load(chposfile)  % loads chan_pos

% Match the 383-channel MUA data by removing channel 192 from 384-position file.
sel = [1:191 193:384];
chan_pos2 = chan_pos(sel, :);

% Depth is the second coordinate, converted from microns to mm.
channel_depth = chan_pos2(:, 2) / 1e3;

% Sort channels from shallow to deep, or vice versa depending on coordinate convention.
[~, I] = sort(channel_depth);
channel_depth_sorted = channel_depth(I);

% % Save a sanity-check plot of sorted channel depths.
%
% filename = 'depths.jpg';
% fig_dir = fullfile(fig_dir_base, exp_name);
%
% if ~exist(fig_dir, 'dir')
%     mkdir(fig_dir)
% end
%
% imtosave = getframe(gcf);
% imwrite(imtosave.cdata, fullfile(fig_dir, filename), 'jpg')


%% Apply depth ordering to neural data

% Reorder channel dimension only: channels x time x presentations.
rasters = rasters(I, :, :);

%% Compute image-averaged natural rasters

% Group repeated presentations of the same image.
[uniqueImage, ~, imIndex] = unique(allimages);

nims = max(imIndex);

% natraster: channels x time x unique_images
natraster = nan(size(rasters, 1), size(rasters, 2), nims, 'single');

% image_resp: channels x unique_images, averaged over evoked window.
image_resp = nan(size(rasters, 1), nims, 'single');

for imno = 1:nims

    curr_trials = imIndex == imno;

    % Average over repeated presentations of the same image.
    natraster(:, :, imno) = nanmean(rasters(:, :, curr_trials), 3);

    % Mean evoked response for this image.
    image_resp(:, imno) = squeeze(nanmean(nanmean( ...
        rasters(:, evkwindow, curr_trials), 3), 2));

end


%% Save preprocessed outputs

save(fullfile(fig_dir, 'preprocessed_natraster.mat'), ...
    'natraster', ...
    'image_resp', ...
    'uniqueImage', ...
    'imIndex', ...
    'allimages', ...
    'stim_xy', ...
    'channel_depth_sorted', ...
    'I', ...
    '-v7.3');