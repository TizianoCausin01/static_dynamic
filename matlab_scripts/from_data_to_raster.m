%{
from_data_to_raster
Extracts image- or video-aligned neural responses from Plexon or Neuropixels
recordings, optionally averaging repeated presentations of each stimulus.

INPUT:
    - monkey_name: char|string -> monkey identifier used in experiment names.
    - recording_dates: char|string|cell -> dates or complete experiment names.
    - final_name: char|string -> user-defined prefix for the saved files.
    - condition: char|string -> images/img or videos/vid.
    - output_type: char|string -> raster or natraster.
    - varargin: name-value pairs -> optional path/environment overrides.

OUTPUT:
    - output_path: char -> path of the saved MAT file.
    - sanity_path: char -> path of the saved average-response PNG.

COMMAND-LINE EXAMPLES:
    matlab -batch "addpath('matlab_scripts'); from_data_to_raster( ...
        'baby1', {'260716','260717'}, 'baby1_260716to17', ...
        'videos', 'raster')"
    matlab -batch "addpath('matlab_scripts'); from_data_to_raster( ...
        'red', {'20260726','20260727'}, 'red_20260726to27', ...
        'images', 'natraster')"
%}
function [output_path, sanity_path] = from_data_to_raster( ...
        monkey_name, recording_dates, final_name, condition, output_type, varargin)

    cfg = parse_inputs( ...
        monkey_name, recording_dates, final_name, condition, output_type, varargin{:});

    % Inspect the requested sessions before reading their continuous data.
    [recordings, cfg] = inspect_recordings(cfg);
    [allimages, stimulus_keys, presentation_experiments, stim_xy] = ...
        build_presentation_metadata(recordings, cfg);

    % Use the source scripts' sorted stimulus axis and retain its trial map.
    [unique_keys, unique_indices, imIndex] = unique(stimulus_keys);
    uniqueImage = allimages(unique_indices);
    imIndex = imIndex(:)';
    repetitions_per_stimulus = accumarray( ...
        imIndex(:), 1, [numel(unique_keys), 1]);

    % Neuropixels channels need one anatomical permutation for every trial.
    channel_depth_sorted = [];
    channel_order = [];
    channel_selector = [];
    unit_names = [];
    if cfg.backend == "neuropixels"
        [channel_depth_sorted, channel_order, channel_selector] = ...
            get_neuropixels_channel_order(recordings, cfg);
    else
        unit_names = recordings(1).unit_names;
    end % end if cfg.backend

    output_stem = sprintf( ...
        '%s_%s_%s', cfg.final_name, cfg.output_type, cfg.condition_short);
    output_path = fullfile(cfg.data_dir, [output_stem, '.mat']);
    sanity_path = fullfile(cfg.sanity_dir, [output_stem, '.png']);

    if isfile(output_path) && ~cfg.overwrite
        error( ...
            'from_data_to_raster:OutputExists', ...
            'Output already exists: %s. Pass ''Overwrite'', true to replace it.', ...
            output_path);
    end % end if isfile(output_path)

    % Raw rasters are streamed to disk to avoid holding all trials in RAM.
    if cfg.output_type == "raster"
        temporary_output_path = [tempname(cfg.data_dir), '.mat'];
        temporary_cleanup = onCleanup( ...
            @() delete_temporary_output(temporary_output_path));
        save(temporary_output_path, ...
            'allimages', ...
            'stimulus_keys', ...
            'uniqueImage', ...
            'imIndex', ...
            'repetitions_per_stimulus', ...
            'presentation_experiments', ...
            'stim_xy', ...
            'channel_depth_sorted', ...
            'channel_order', ...
            'channel_selector', ...
            'unit_names', ...
            'cfg', ...
            '-v7.3');
        output_file = matfile(temporary_output_path, 'Writable', true);
        output_file.raster( ...
            cfg.n_neural_features, cfg.window_length_ms, ...
            cfg.n_presentations) = single(0);
        average_sum = zeros(1, cfg.window_length_ms);
        average_count = zeros(1, cfg.window_length_ms);
    else
        % Accumulate unique-stimulus means without retaining every trial.
        n_stimuli = numel(uniqueImage);
        natraster_sum = zeros( ...
            cfg.n_neural_features, cfg.window_length_ms, ...
            n_stimuli, 'single');
        natraster_count = zeros( ...
            cfg.n_neural_features, cfg.window_length_ms, ...
            n_stimuli, 'uint16');
    end % end if cfg.output_type

    presentation_index = 0;
    for recording_index = 1:numel(recordings)

        recording = recordings(recording_index);
        metadata = load(recording.metadata_path, 'Stimuli');

        % MATLAB exposes these HDF5 arrays as neural features x milliseconds.
        continuous_data = single(h5read( ...
            recording.neural_path, recording.neural_dataset));
        if size(continuous_data, 1) ~= cfg.n_neural_features
            continuous_data = continuous_data';
        end % end if size(continuous_data, 1)
        if size(continuous_data, 1) ~= cfg.n_neural_features
            error( ...
                'from_data_to_raster:NeuralShapeMismatch', ...
                ['Expected %d neural features in %s, but the HDF5 dataset ' ...
                 'has shape %d x %d.'], ...
                cfg.n_neural_features, recording.neural_path, ...
                size(continuous_data, 1), size(continuous_data, 2));
        end % end if size(continuous_data, 1)

        for selected_index = 1:numel(recording.stimulus_indices)

            stimulus_index = recording.stimulus_indices(selected_index);
            stimulus = metadata.Stimuli(stimulus_index);
            presentation_index = presentation_index + 1;

            % Preserve the original pipelines' one-millisecond onset offset.
            window_start = round(double(stimulus.start_time) + 1);
            window_stop = window_start + cfg.window_length_ms - 1;
            if window_start < 1 || window_stop > size(continuous_data, 2)
                error( ...
                    'from_data_to_raster:WindowOutOfBounds', ...
                    ['Stimulus %d in %s requests samples %d:%d, but the ' ...
                     'recording contains %d milliseconds.'], ...
                    stimulus_index, recording.experiment_name, ...
                    window_start, window_stop, size(continuous_data, 2));
            end % end if window_start

            presentation = continuous_data(:, window_start:window_stop);
            baseline = mean( ...
                presentation(:, cfg.baseline_window), 2, 'omitnan');
            presentation = presentation - repmat( ...
                baseline, [1 cfg.window_length_ms]);

            if cfg.backend == "neuropixels"
                presentation = presentation(channel_order, :);
            end % end if cfg.backend

            if cfg.output_type == "raster"
                output_file.raster(:, :, presentation_index) = presentation;
                average_sum = average_sum + double(sum( ...
                    presentation, 1, 'omitnan'));
                average_count = average_count + sum( ...
                    ~isnan(presentation), 1);
            else
                stimulus_number = imIndex(presentation_index);
                valid_values = ~isnan(presentation);
                presentation(~valid_values) = 0;
                natraster_sum(:, :, stimulus_number) = ...
                    natraster_sum(:, :, stimulus_number) + presentation;
                natraster_count(:, :, stimulus_number) = ...
                    natraster_count(:, :, stimulus_number) + ...
                    uint16(valid_values);
            end % end if cfg.output_type

        end % end for selected_index

        clear continuous_data metadata

    end % end for recording_index

    if presentation_index ~= cfg.n_presentations
        error( ...
            'from_data_to_raster:PresentationCountMismatch', ...
            'Extracted %d presentations after inspecting %d.', ...
            presentation_index, cfg.n_presentations);
    end % end if presentation_index

    if cfg.output_type == "raster"
        average_response = average_sum ./ average_count;
        output_n_samples = cfg.n_presentations;
        clear output_file
        movefile(temporary_output_path, output_path, 'f');
        clear temporary_cleanup
    else
        natraster = natraster_sum ./ single(natraster_count);
        image_resp = squeeze(mean( ...
            natraster(:, cfg.evoked_window, :), 2, 'omitnan'));

        save(output_path, ...
            'natraster', ...
            'image_resp', ...
            'allimages', ...
            'stimulus_keys', ...
            'uniqueImage', ...
            'imIndex', ...
            'repetitions_per_stimulus', ...
            'presentation_experiments', ...
            'stim_xy', ...
            'channel_depth_sorted', ...
            'channel_order', ...
            'channel_selector', ...
            'unit_names', ...
            'cfg', ...
            '-v7.3');
        average_response = squeeze(mean(mean( ...
            natraster, 1, 'omitnan'), 3, 'omitnan'));
        output_n_samples = numel(uniqueImage);
    end % end if cfg.output_type

    save_average_plot(average_response, sanity_path, output_stem, cfg);

    fprintf('Saved %s with shape %d x %d x %d.\n', ...
        cfg.output_type, cfg.n_neural_features, ...
        cfg.window_length_ms, output_n_samples);
    fprintf('Neural data: %s\n', output_path);
    fprintf('Sanity check: %s\n', sanity_path);
    fprintf('Repetitions per stimulus: min=%d, max=%d.\n', ...
        min(repetitions_per_stimulus), max(repetitions_per_stimulus));

end % EOF


%{
parse_inputs
Validates public arguments and builds the preprocessing configuration.

INPUT:
    - monkey_name: char|string -> monkey identifier.
    - recording_dates: char|string|cell|numeric -> dates or experiment names.
    - final_name: char|string -> output prefix.
    - condition: char|string -> image or video condition.
    - output_type: char|string -> raster or natraster.
    - varargin: name-value pairs -> environment/path overrides.

OUTPUT:
    - cfg: struct -> validated preprocessing configuration.
%}
function cfg = parse_inputs( ...
        monkey_name, recording_dates, final_name, condition, output_type, varargin)

    default_environment = string(getenv('MY_ENV'));
    if strlength(default_environment) == 0
        default_environment = "tiziano_mac_mini";
    end % end if strlength(default_environment)

    parser = inputParser;
    parser.FunctionName = 'from_data_to_raster';
    addRequired(parser, 'monkey_name', @is_text_scalar);
    addRequired(parser, 'recording_dates', @is_valid_recording_dates);
    addRequired(parser, 'final_name', @is_text_scalar);
    addRequired(parser, 'condition', @is_text_scalar);
    addRequired(parser, 'output_type', @is_text_scalar);
    addParameter(parser, 'Environment', default_environment, @is_text_scalar);
    addParameter(parser, 'DataPath', "", @is_text_scalar);
    addParameter(parser, 'LivingstoneLabPath', "", @is_text_scalar);
    addParameter(parser, 'ChannelPositionExperiment', "", @is_text_scalar);
    addParameter(parser, 'Overwrite', false, ...
        @(value) islogical(value) && isscalar(value));
    parse(parser, ...
        monkey_name, recording_dates, final_name, condition, output_type, ...
        varargin{:});

    cfg = struct();
    cfg.monkey_name = strtrim(string(parser.Results.monkey_name));
    cfg.recording_dates = normalize_recording_dates( ...
        parser.Results.recording_dates);
    cfg.experiment_names = build_experiment_names( ...
        cfg.monkey_name, cfg.recording_dates);
    cfg.final_name = strtrim(string(parser.Results.final_name));
    cfg.environment = strtrim(string(parser.Results.Environment));
    cfg.channel_position_experiment = strtrim(string( ...
        parser.Results.ChannelPositionExperiment));
    cfg.overwrite = parser.Results.Overwrite;

    if strlength(cfg.monkey_name) == 0 || strlength(cfg.final_name) == 0
        error( ...
            'from_data_to_raster:EmptyName', ...
            'monkey_name and final_name cannot be empty.');
    end % end if strlength(cfg.monkey_name)

    condition_value = lower(strtrim(string(parser.Results.condition)));
    if any(condition_value == ["image", "images", "img"])
        cfg.condition = "images";
        cfg.condition_short = "img";
        cfg.window_length_ms = 1000;
    elseif any(condition_value == ["video", "videos", "vid"])
        cfg.condition = "videos";
        cfg.condition_short = "vid";
        cfg.window_length_ms = 3500;
    else
        error( ...
            'from_data_to_raster:InvalidCondition', ...
            'condition must be images/img or videos/vid.');
    end % end if condition_value

    output_value = lower(strtrim(string(parser.Results.output_type)));
    if ~any(output_value == ["raster", "natraster"])
        error( ...
            'from_data_to_raster:InvalidOutputType', ...
            'output_type must be raster or natraster.');
    end % end if output_value
    cfg.output_type = output_value;

    % Preserve the baseline conventions from the two source pipelines.
    cfg.neuropixels_baseline_window = 1:40;
    cfg.plexon_baseline_window = 1:50;
    cfg.evoked_window = 90:470;
    cfg.include_unsuccessful_trials = true;

    project_root = fileparts(fileparts(mfilename('fullpath')));
    config_path = fullfile(project_root, 'config.yaml');
    project_paths = read_project_paths(config_path, cfg.environment);

    cfg.data_path = strtrim(string(parser.Results.DataPath));
    if strlength(cfg.data_path) == 0
        cfg.data_path = project_paths.data_path;
    end % end if strlength(cfg.data_path)

    cfg.livingstone_lab_path = strtrim(string( ...
        parser.Results.LivingstoneLabPath));
    if strlength(cfg.livingstone_lab_path) == 0
        cfg.livingstone_lab_path = project_paths.livingstone_lab;
    end % end if strlength(cfg.livingstone_lab_path)

    cfg.data_dir = fullfile(cfg.data_path, 'data');
    cfg.sanity_dir = fullfile(cfg.data_dir, 'sanity_checks');
    cfg.data_formatted = fullfile( ...
        cfg.livingstone_lab_path, 'Data-Formatted');
    cfg.data_neuropixels = fullfile( ...
        cfg.livingstone_lab_path, 'Data-Neuropixels-Preprocessed');

    if ~isfolder(cfg.data_dir)
        mkdir(cfg.data_dir);
    end % end if ~isfolder(cfg.data_dir)
    if ~isfolder(cfg.sanity_dir)
        mkdir(cfg.sanity_dir);
    end % end if ~isfolder(cfg.sanity_dir)

end % EOF


%{
inspect_recordings
Finds the acquisition backend and condition-matched trials for each recording.

INPUT:
    - cfg: struct -> preprocessing configuration.

OUTPUT:
    - recordings: struct array -> paths, selected trials, and neural metadata.
    - cfg: struct -> configuration updated with backend and total dimensions.
%}
function [recordings, cfg] = inspect_recordings(cfg)

    recordings = struct([]);
    backend = "";
    reference_unit_names = [];
    reference_channel_selector = [];
    n_presentations = 0;
    n_neural_features = [];

    for recording_index = 1:numel(cfg.experiment_names)

        experiment_name = cfg.experiment_names(recording_index);
        metadata_path = fullfile( ...
            cfg.data_formatted, experiment_name + "_experiment.mat");
        if ~isfile(metadata_path)
            error( ...
                'from_data_to_raster:MissingMetadata', ...
                'Missing experiment metadata: %s', metadata_path);
        end % end if ~isfile(metadata_path)

        plexon_path = fullfile( ...
            cfg.data_formatted, experiment_name + "-rasters.h5");
        neuropixels_folder = fullfile( ...
            cfg.data_neuropixels, experiment_name, ...
            "catgt_" + experiment_name + "_g0", ...
            experiment_name + "_g0_imec0");
        neuropixels_path = fullfile( ...
            neuropixels_folder, experiment_name + "-imec0-mua_cont.h5");

        has_plexon = isfile(plexon_path);
        has_neuropixels = isfile(neuropixels_path);
        if has_plexon == has_neuropixels
            error( ...
                'from_data_to_raster:AmbiguousBackend', ...
                ['Expected exactly one Plexon or Neuropixels neural file ' ...
                 'for %s. Plexon found=%d, Neuropixels found=%d.'], ...
                experiment_name, has_plexon, has_neuropixels);
        end % end if has_plexon == has_neuropixels

        if has_plexon
            current_backend = "plexon";
            neural_path = plexon_path;
            neural_dataset = '/rasters';
            unit_names = h5read(neural_path, '/unit_names');
            current_n_features = numel(unit_names);
            channel_selector = [];
        else
            current_backend = "neuropixels";
            neural_path = neuropixels_path;
            neural_dataset = '/mua_cont';
            unit_names = [];
            channel_selector = h5read( ...
                neural_path, '/mua_cont-meta/chan_sel');
            channel_selector = channel_selector(:);
            current_n_features = numel(channel_selector);
        end % end if has_plexon

        if strlength(backend) == 0
            backend = current_backend;
            n_neural_features = current_n_features;
            reference_unit_names = unit_names;
            reference_channel_selector = channel_selector;
        elseif backend ~= current_backend
            error( ...
                'from_data_to_raster:MixedBackends', ...
                'All requested recordings must use the same acquisition backend.');
        elseif n_neural_features ~= current_n_features
            error( ...
                'from_data_to_raster:FeatureCountMismatch', ...
                'Neural feature counts differ across requested recordings.');
        elseif backend == "plexon" && ...
                ~isequal(reference_unit_names, unit_names)
            error( ...
                'from_data_to_raster:UnitNameMismatch', ...
                'Plexon unit names differ across requested recordings.');
        elseif backend == "neuropixels" && ...
                ~isequal(reference_channel_selector, channel_selector)
            error( ...
                'from_data_to_raster:ChannelSelectorMismatch', ...
                'Neuropixels channel selectors differ across recordings.');
        end % end if strlength(backend)

        metadata = load(metadata_path, 'Stimuli');
        stimulus_indices = [];
        for stimulus_index = 1:numel(metadata.Stimuli)
            filename = metadata.Stimuli(stimulus_index).filename;
            if is_stimulus_for_condition(filename, cfg.condition)
                stimulus_indices(end + 1) = stimulus_index; %#ok<AGROW>
            end % end if is_stimulus_for_condition
        end % end for stimulus_index

        if isempty(stimulus_indices)
            error( ...
                'from_data_to_raster:NoConditionStimuli', ...
                'No %s stimuli were found in %s.', ...
                cfg.condition, experiment_name);
        end % end if isempty(stimulus_indices)

        recordings(recording_index).experiment_name = char(experiment_name);
        recordings(recording_index).metadata_path = char(metadata_path);
        recordings(recording_index).neural_path = char(neural_path);
        recordings(recording_index).neural_dataset = neural_dataset;
        recordings(recording_index).neuropixels_folder = ...
            char(neuropixels_folder);
        recordings(recording_index).stimulus_indices = stimulus_indices;
        recordings(recording_index).unit_names = unit_names;
        recordings(recording_index).channel_selector = channel_selector;
        n_presentations = n_presentations + numel(stimulus_indices);

        fprintf('Matched %d %s presentations in %s.\n', ...
            numel(stimulus_indices), cfg.condition, experiment_name);

    end % end for recording_index

    cfg.backend = backend;
    cfg.n_neural_features = n_neural_features;
    cfg.n_presentations = n_presentations;
    if backend == "plexon"
        cfg.baseline_window = cfg.plexon_baseline_window;
    else
        cfg.baseline_window = cfg.neuropixels_baseline_window;
    end % end if backend

end % EOF


%{
build_presentation_metadata
Collects metadata in the same order used later for neural extraction.

INPUT:
    - recordings: struct array -> recording paths and retained trial indices.
    - cfg: struct -> preprocessing configuration.

OUTPUT:
    - allimages: cell -> one stimulus filename per presentation.
    - stimulus_keys: cell -> canonical keys used to group repetitions.
    - presentation_experiments: cell -> source experiment per presentation.
    - stim_xy: single -> presentation positions with shape trials x 2.
%}
function [allimages, stimulus_keys, presentation_experiments, stim_xy] = ...
        build_presentation_metadata(recordings, cfg)

    allimages = cell(1, cfg.n_presentations);
    stimulus_keys = cell(1, cfg.n_presentations);
    presentation_experiments = cell(1, cfg.n_presentations);
    stim_xy = nan(cfg.n_presentations, 2, 'single');
    presentation_index = 0;

    for recording_index = 1:numel(recordings)

        recording = recordings(recording_index);
        metadata = load(recording.metadata_path, 'Stimuli');

        for selected_index = 1:numel(recording.stimulus_indices)

            stimulus_index = recording.stimulus_indices(selected_index);
            stimulus = metadata.Stimuli(stimulus_index);
            presentation_index = presentation_index + 1;

            stimulus_name = get_stimulus_name(stimulus.filename);
            allimages{presentation_index} = stimulus_name;
            stimulus_keys{presentation_index} = lower(strtrim(stimulus_name));
            presentation_experiments{presentation_index} = ...
                recording.experiment_name;

            position = single(stimulus.position(:));
            if numel(position) < 2
                error( ...
                    'from_data_to_raster:InvalidStimulusPosition', ...
                    'Stimulus %d in %s has fewer than two position values.', ...
                    stimulus_index, recording.experiment_name);
            end % end if numel(position)
            stim_xy(presentation_index, :) = position(1:2)';

        end % end for selected_index

    end % end for recording_index

end % EOF


%{
get_neuropixels_channel_order
Selects the recorded probe channels and determines their anatomical order.

INPUT:
    - recordings: struct array -> Neuropixels paths and channel selectors.
    - cfg: struct -> preprocessing configuration.

OUTPUT:
    - channel_depth_sorted: double -> selected channel depths in millimeters.
    - channel_order: double -> permutation from selected to depth-sorted channels.
    - channel_selector: integer -> zero-based probe channels stored in the HDF5.
%}
function [channel_depth_sorted, channel_order, channel_selector] = ...
        get_neuropixels_channel_order(recordings, cfg)

    channel_experiment = cfg.channel_position_experiment;
    if strlength(channel_experiment) == 0
        channel_experiment = string(recordings(1).experiment_name);
    elseif ~startsWith(channel_experiment, cfg.monkey_name + "_")
        channel_experiment = cfg.monkey_name + "_" + channel_experiment;
    end % end if strlength(channel_experiment)

    channel_folder = fullfile( ...
        cfg.data_neuropixels, channel_experiment, ...
        "catgt_" + channel_experiment + "_g0", ...
        channel_experiment + "_g0_imec0");
    channel_path = fullfile(channel_folder, 'channel_positions.mat');
    if ~isfile(channel_path)
        error( ...
            'from_data_to_raster:MissingChannelPositions', ...
            'Missing Neuropixels channel positions: %s', channel_path);
    end % end if ~isfile(channel_path)

    channel_data = load(channel_path, 'chan_pos');
    channel_selector = double(recordings(1).channel_selector(:));

    % chan_sel uses zero-based probe indices; MATLAB rows are one-based.
    selected_rows = channel_selector + 1;
    if any(selected_rows < 1) || any(selected_rows > size(channel_data.chan_pos, 1))
        error( ...
            'from_data_to_raster:InvalidChannelSelector', ...
            'The HDF5 channel selector is incompatible with chan_pos.');
    end % end if any(selected_rows)

    selected_positions = channel_data.chan_pos(selected_rows, :);
    channel_depth = selected_positions(:, 2) / 1e3;
    [channel_depth_sorted, channel_order] = sort(channel_depth);

end % EOF


%{
save_average_plot
Plots a global mean neural time course and saves it without opening a window.

INPUT:
    - average_response: numeric -> one global response value per time bin.
    - sanity_path: char|string -> destination PNG.
    - output_stem: char|string -> plot title.
    - cfg: struct -> condition and backend labels.
%}
function save_average_plot(average_response, sanity_path, output_stem, cfg)

    average_response = average_response(:);
    figure_handle = figure('Visible', 'off', 'Color', 'white');
    plot(0:(numel(average_response) - 1), average_response, ...
        'LineWidth', 1.5);
    xlabel('Time from stimulus onset (ms)');
    ylabel('Baseline-subtracted response');
    title(sprintf('%s | %s | %s', ...
        output_stem, cfg.backend, cfg.condition), 'Interpreter', 'none');
    grid on;
    exportgraphics(figure_handle, sanity_path, 'Resolution', 180);
    close(figure_handle);

end % EOF


%{
is_stimulus_for_condition
Matches condition prefixes first and uses file extensions only as a fallback.

INPUT:
    - filename: char|string -> stimulus filename from experiment metadata.
    - condition: string -> images or videos.

OUTPUT:
    - matches: logical -> whether the presentation belongs to the condition.
%}
function matches = is_stimulus_for_condition(filename, condition)

    stimulus_name = lower(string(get_stimulus_name(filename)));
    [~, stimulus_stem, extension] = fileparts(stimulus_name);

    if startsWith(stimulus_stem, 'img_')
        matches = condition == "images";
    elseif startsWith(stimulus_stem, 'vid_')
        matches = condition == "videos";
    else
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"];
        video_extensions = [".mp4", ".mov", ".avi", ".mpeg", ".mpg"];
        if condition == "images"
            matches = any(extension == image_extensions);
        else
            matches = any(extension == video_extensions);
        end % end if condition
    end % end if startsWith(stimulus_stem)

end % EOF


%{
get_stimulus_name
Removes any directory component so repeated stimuli match by file identity.

INPUT:
    - filename: char|string -> stimulus path or filename.

OUTPUT:
    - stimulus_name: char -> basename plus extension.
%}
function stimulus_name = get_stimulus_name(filename)

    [~, stem, extension] = fileparts(char(filename));
    stimulus_name = [stem, extension];

end % EOF


%{
normalize_recording_dates
Converts supported date inputs to a non-empty row string array.

INPUT:
    - recording_dates: char|string|cell|numeric -> dates or experiment names.

OUTPUT:
    - dates: string -> normalized row array.
%}
function dates = normalize_recording_dates(recording_dates)

    if isnumeric(recording_dates)
        dates = compose('%.0f', recording_dates);
    elseif ischar(recording_dates)
        dates = string(regexp(strtrim(recording_dates), '[,;\s]+', 'split'));
    elseif isstring(recording_dates)
        if isscalar(recording_dates)
            dates = string(regexp( ...
                strtrim(recording_dates), '[,;\s]+', 'split'));
        else
            dates = recording_dates;
        end % end if isscalar(recording_dates)
    elseif iscell(recording_dates)
        dates = string(recording_dates);
    else
        error( ...
            'from_data_to_raster:InvalidRecordingDates', ...
            'recording_dates must contain dates or experiment names.');
    end % end if isnumeric(recording_dates)

    dates = reshape(strtrim(dates), 1, []);
    dates(strlength(dates) == 0) = [];
    if isempty(dates)
        error( ...
            'from_data_to_raster:EmptyRecordingDates', ...
            'At least one recording date is required.');
    end % end if isempty(dates)

end % EOF


%{
build_experiment_names
Prefixes bare dates with the monkey name while preserving complete names.

INPUT:
    - monkey_name: string -> monkey identifier.
    - recording_dates: string -> dates or complete experiment names.

OUTPUT:
    - experiment_names: string -> complete experiment names.
%}
function experiment_names = build_experiment_names(monkey_name, recording_dates)

    experiment_names = strings(size(recording_dates));
    experiment_prefix = monkey_name + "_";
    for date_index = 1:numel(recording_dates)
        if startsWith(recording_dates(date_index), experiment_prefix)
            experiment_names(date_index) = recording_dates(date_index);
        else
            experiment_names(date_index) = ...
                experiment_prefix + recording_dates(date_index);
        end % end if startsWith(recording_dates)
    end % end for date_index

end % EOF


%{
read_project_paths
Reads data_path and livingstone_lab from one environment in config.yaml.

INPUT:
    - config_path: char|string -> project YAML path.
    - environment: char|string -> top-level environment key.

OUTPUT:
    - paths: struct -> data_path and livingstone_lab strings.
%}
function paths = read_project_paths(config_path, environment)

    if ~isfile(config_path)
        error( ...
            'from_data_to_raster:MissingConfig', ...
            'Missing project configuration: %s', config_path);
    end % end if ~isfile(config_path)

    config_text = string(fileread(config_path));
    environment_pattern = "(?m)^" + ...
        regexptranslate('escape', char(environment)) + ...
        ":\s*\r?\n(?<block>(?:[ \t]+[^\r\n]*(?:\r?\n|$))*)";
    environment_match = regexp( ...
        config_text, environment_pattern, 'names', 'once');
    if isempty(environment_match)
        error( ...
            'from_data_to_raster:UnknownEnvironment', ...
            'Environment %s was not found in %s.', ...
            environment, config_path);
    end % end if isempty(environment_match)

    paths = struct();
    paths.data_path = read_yaml_path( ...
        string(environment_match.block), 'data_path', environment);
    paths.livingstone_lab = read_yaml_path( ...
        string(environment_match.block), 'livingstone_lab', environment);

end % EOF


%{
read_yaml_path
Extracts one quoted or unquoted path value from an environment YAML block.

INPUT:
    - environment_block: string -> indented YAML block.
    - path_name: char|string -> path key.
    - environment: char|string -> environment name used in errors.

OUTPUT:
    - path_value: string -> configured filesystem path.
%}
function path_value = read_yaml_path(environment_block, path_name, environment)

    path_pattern = "(?m)^\s+" + string(path_name) + ...
        ":\s*[""']?([^""'\r\n]+)";
    path_match = regexp( ...
        environment_block, path_pattern, 'tokens', 'once');
    if isempty(path_match)
        error( ...
            'from_data_to_raster:MissingConfiguredPath', ...
            'Path %s is not configured for environment %s.', ...
            path_name, environment);
    end % end if isempty(path_match)

    path_value = strtrim(string(path_match{1}));

end % EOF


%{
delete_temporary_output
Removes an incomplete MAT file if streamed raster generation fails.

INPUT:
    - temporary_output_path: char|string -> task-owned temporary MAT path.
%}
function delete_temporary_output(temporary_output_path)

    if isfile(temporary_output_path)
        delete(temporary_output_path);
    end % end if isfile(temporary_output_path)

end % EOF


function valid = is_text_scalar(value)

    valid = ischar(value) || (isstring(value) && isscalar(value));

end % EOF


function valid = is_valid_recording_dates(value)

    valid = ischar(value) || isstring(value) || iscell(value) || isnumeric(value);

end % EOF
