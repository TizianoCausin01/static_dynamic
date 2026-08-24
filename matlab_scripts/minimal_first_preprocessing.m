%% Soft-coded paths
clear all; 
BASE_DIR = "/Users/tizianocausin/livingstone_lab_local";
data_dir = "/Users/tizianocausin/sd_local/data";
data_formatted   = fullfile(BASE_DIR, 'Data-Formatted');
data_neuropixel  = fullfile(BASE_DIR, 'Data-Neuropixels-Preprocessed');
final_name = "baby1_260716to27";

%% Parameters
bkgwindow=5:35; % what's for?
evkwindow=90:470; % what's for?
window_length_ms = 1000; % parameter
raster_window  = [1 window_length_ms];

chanpos_exp_name = 'baby1_260630';  % experiment used only for channel geometry (ask if needed to be always the same)

exp_names = [
    % 'baby1_260716',
    'baby1_260718',
    % 'baby1_260720',
    % 'baby1_260721',
    % 'baby1_260722',
    % 'baby1_260723',
    % 'baby1_260724'
    'baby1_260725', 
    'baby1_260726'
    'baby1_260727'
];


%% Load MUA and extract stimulus-aligned rasters
ii = 0; % what's for?
allimages = {};
for indx = 1:size(exp_names, 1)

    curr_exp = strtrim(exp_names(indx, :)); % strtrim just removes leading and trailing whitespaces

    % loads: config (experiment_metadata), Session (experiment_metadata),
    % Stimuli (trial_number, start and stop times, latency, filename(i.e.
    % stimulus name)),
    % Trials (start and stop times, success, error_code, eye_data, fixation_positions,
    % fixation_times, saccade_times) (CHECK HOW TO EPOCH ALSO EYE
    % MOVEMENTS)
    exp_path = fullfile(data_formatted, sprintf('%s_experiment.mat', curr_exp)); % loads the experiment from the formatted data
    load(exp_path)

    % Load continuous MUA, channels x time_ms (around 49 mins).
    mua_path = fullfile( ...
        data_neuropixel, char(curr_exp), ...
        ['catgt_', char(curr_exp), '_g0'], ...
        [char(curr_exp), '_g0_imec0'], ...
        [char(curr_exp), '-imec0-mua_cont.h5'] ...
    );
    mua = h5read(mua_path, '/mua_cont');

    n_presentations = length(Stimuli); % how many stimuli were presented
    for i = 1:n_presentations
            ii = ii + 1;

            % Convert stimulus onset + raster offset into sample index.
            window = round(Stimuli(i).start_time + raster_window(1)); % takes the window start and end, rounds up to ms and crops the mua there
            % WARNING: IT IS NOT FILTERING FOR TRIAL SUCCESS
            % Extract channels x time window around this stimulus.
            rasters(:, :, ii) = mua(:, window:window + window_length_ms - 1);

            % Baseline-subtract each channel using the first 40 time bins.
            baseline = nanmean(rasters(:, 1:40, ii), 2);
            rasters(:, :, ii) = rasters(:, :, ii) - repmat(baseline, [1, size(rasters, 2)]); % subtracts the baseline (0 to 40ms) to all the timecourse of the trial

            % Store stimulus metadata.
            allimages{ii} = Stimuli(i).filename; % all the images/videos in the order they were presented
            stim_xy(ii, :) = Stimuli(i).position; % position of the stimulus presentation (don't think it's needed)
    end
    clear mua %Stimuli Trials
end
%%
disp("B1")
disp(Session(1).image_root_dir)
plot(mean(rasters, [1,3]))
title("B1")
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
sel = [1:191 193:384]; % removes channel 192
chan_pos2 = chan_pos(sel, :);

% Depth is the second coordinate of channel_depth, converted from microns to mm.
channel_depth = chan_pos2(:, 2) / 1e3;

% sorts because the channels are organized like 0 half 1 half+1 etc
[~, I] = sort(channel_depth);
channel_depth_sorted = channel_depth(I);
% Reorder channel dimension only: channels x time x presentations.
rasters = rasters(I, :, :);

%% This is for natrasters (averaged across different repetitions of the same image)

% Group repeated presentations of the same image.
[uniqueImage, ~, imIndex] = unique(allimages); % counts the instances of images

nims = max(imIndex); %how many images there are

% natraster: channels x time x unique_images
natraster = nan(size(rasters, 1), size(rasters, 2), nims, 'single');

% image_resp: channels x unique_images, averaged over evoked window.
image_resp = nan(size(rasters, 1), nims, 'single');

for imno = 1:nims

    curr_trials = imIndex == imno; % selects which trials have the same index

    % Average over repeated presentations of the same image.
    natraster(:, :, imno) = nanmean(rasters(:, :, curr_trials), 3); % averages across repetitions of the same stimulus

end


%% Save preprocessed outputs

save(fullfile(data_dir, sprintf('%s_natraster.mat', final_name)), ...
    'natraster', ...
    'image_resp', ...
    'uniqueImage', ...
    'imIndex', ...
    'allimages', ...
    'stim_xy', ...
    'channel_depth_sorted', ...
    'I', ...
    '-v7.3');
%%
  repetitions_per_image = accumarray( ...                                                                                                       
      imIndex(:), 1, [numel(uniqueImage), 1]);                                                                                                  
                                                                                                                                                
  min_repetitions = min(repetitions_per_image);                                                                                                 
  max_repetitions = max(repetitions_per_image);                                                                                                 

  images_with_min_repetitions = ...
      uniqueImage(repetitions_per_image == min_repetitions);

  images_with_max_repetitions = ...
      uniqueImage(repetitions_per_image == max_repetitions);

  fprintf('Minimum repetitions per image: %d\n', min_repetitions);
  disp(images_with_min_repetitions)

  fprintf('Maximum repetitions per image: %d\n', max_repetitions);
  disp(images_with_max_repetitions)

  % Optional: display all images and their repetition counts.
  repetition_table = table( ...
      uniqueImage(:), repetitions_per_image, ...
      'VariableNames', {'image', 'n_repetitions'});

  repetition_table = sortrows( ...
      repetition_table, 'n_repetitions', 'descend');

  disp(repetition_table)