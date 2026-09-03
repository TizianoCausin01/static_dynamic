%% Configuration
clear
cfg = struct();

% Plexon preprocessing stores both formatted metadata and continuous,
% millisecond-binned spike rasters in the formatted-data directory.
cfg.BASE_DIR = "/Users/tizianocausin/livingstone_lab_local";
cfg.data_dir = "/Users/tizianocausin/sd_local/data";
cfg.data_formatted   = fullfile(cfg.BASE_DIR, 'Data-Formatted');

cfg.final_name = 'paul_20260831' %'red_20260720to24'; %to22';
cfg.exp_names = {
    % 'red_20260720',
    % 'red_20260721',
    % 'red_20260722',
    % 'red_20260723',
    % 'red_20260724',
    'paul_20260831',
};

% Keep the existing Plexon baseline convention while matching the temporal
% window used by the Neuropixels preprocessing script.
cfg.window_length_ms = 3500;
cfg.raster_window = [1 cfg.window_length_ms];
cfg.baseline_window = 1:50;
cfg.evoked_window = 90:470;


%% Load Plexon units and extract stimulus-aligned rasters

presentation_index = 0;
unit_names = [];
rasters = [];
allimages = {};
stim_xy = [];

for exp_index = 1:numel(cfg.exp_names)

    curr_exp = strtrim(cfg.exp_names{exp_index});

    % Load config, Session, Stimuli, and Trials. Stimuli supplies the onset,
    % filename, and position used here, as in the Neuropixels pipeline.
    exp_path = fullfile(cfg.data_formatted, [curr_exp, '_experiment.mat']);
    load(exp_path)

    % Plexon-specific input: /rasters contains continuous binned spike counts
    % with shape units x time_ms; /unit_names identifies the recorded units.
    rasters_path = fullfile(cfg.data_formatted, [curr_exp, '-rasters.h5']);
    long_rasters = double(h5read(rasters_path, '/rasters'));
    curr_unit_names = h5read(rasters_path, '/unit_names');

    % Multiple experiments can only be concatenated when their unit axes match.
    if isempty(unit_names)
        unit_names = curr_unit_names;
    elseif ~isequal(unit_names, curr_unit_names)
        error('Unit names differ across experiments; their rasters cannot be concatenated.');
    end % end if isempty(unit_names)

    n_presentations = length(Stimuli);

    for stimulus_index = 1:n_presentations

        presentation_index = presentation_index + 1;

        % Convert the stimulus onset and raster offset to a millisecond index.
        window_start = round( ...
            Stimuli(stimulus_index).start_time + cfg.raster_window(1));
        window_stop = window_start + cfg.window_length_ms - 1;

        % Extract units x time for this presentation. As in the Neuropixels
        % script, all presentations are retained regardless of trial success.
        rasters(:, :, presentation_index) = ...
            long_rasters(:, window_start:window_stop);

        % Baseline-subtract every unit using the original Plexon 50 ms window.
        baseline = nanmean( ...
            rasters(:, cfg.baseline_window, presentation_index), 2);
        rasters(:, :, presentation_index) = ...
            rasters(:, :, presentation_index) - ...
            repmat(baseline, [1 size(rasters, 2)]);

        % Preserve presentation order for repetition averaging and alignment.
        allimages{presentation_index} = Stimuli(stimulus_index).filename;
        stim_xy(presentation_index, :) = Stimuli(stimulus_index).position;

    end % end for stimulus_index

    %clear long_rasters Stimuli Trials

end % end for exp_index
%%
disp("R")
disp(Session(1).image_root_dir)
plot(mean(rasters, [1,3]))
title("R")

%% Keep the Plexon unit order

% Neuropixels channels are reordered using probe geometry. Plexon data instead
% consist of named sorted units, so their HDF5 order and names are preserved.


%% Average repetitions of each stimulus

% Group repeated presentations using the same stimulus-name mapping as the
% Neuropixels preprocessing script.
[uniqueImage, ~, imIndex] = unique(allimages);
n_stimuli = max(imIndex);

% natraster: units x time x unique_stimuli.
natraster = nan( ...
    size(rasters, 1), size(rasters, 2), n_stimuli, 'single');

% image_resp: units x unique_stimuli, averaged over the evoked window.
image_resp = nan(size(rasters, 1), n_stimuli, 'single');

for stimulus_number = 1:n_stimuli

    curr_trials = imIndex == stimulus_number;

    % Average the full time course over repetitions of this stimulus.
    natraster(:, :, stimulus_number) = ...
        nanmean(rasters(:, :, curr_trials), 3);

    % Summarize each unit's response over repetitions and evoked time bins.
    image_resp(:, stimulus_number) = squeeze(nanmean(nanmean( ...
        rasters(:, cfg.evoked_window, curr_trials), 3), 2));

end % end for stimulus_number


%% Save preprocessed outputs
% 
% Save the shared natraster fields plus Plexon unit identifiers. There is no
% channel-depth ordering for these sorted-unit recordings.
save(fullfile(cfg.data_dir, sprintf('%s_natraster.mat', cfg.final_name)), ...
    'natraster', ...
    'image_resp', ...
    'uniqueImage', ...
    'imIndex', ...
    'allimages', ...
    'stim_xy', ...
    'unit_names', ...
    'cfg', ...
    '-v7.3');

% EOF
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