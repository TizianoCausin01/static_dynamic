#!/usr/bin/env bash
# Run the neural-to-model RSA for every registered model, for one monkey, with
# one shared parameter family: the session's reliable image channels inside its
# reliability range, cosine_cnt RDMs on both sides, Spearman RSA at 100 Hz, and
# a 1000 ms static crop. Models whose features are missing are reported and
# skipped.
#
# Usage from the project root:
#   bash bash_scripts/run_model_zoo_neural_model_rsa.sh <monkey> [model_name ...]
#
# <monkey> is one of baby1, red, paul; omit it to run baby1.
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

DEFAULT_MODELS=(
    alexnet convnext_base swin_base hiera_base dino_v3_h ijepa_vith14_1k
    siglip2_so400m r3d_18 videomae_base videomae_base_ssv2
    vjepa2_vitl_fpc64_256 pixel_values_rgb_step30 optical_flow_farneback
)

# Per-monkey sessions and the channel range its reliability list was built in.
MONKEY="${1:-baby1}"
case "${MONKEY}" in
    baby1)
        STATIC_EXP=baby1_260718to27
        DYNAMIC_EXP=baby1_260716to24
        FIRST_CHANNEL=84
        LAST_CHANNEL=186
        ;;
    red)
        STATIC_EXP=red_20260726to27
        DYNAMIC_EXP=red_20260720to24
        FIRST_CHANNEL=1
        LAST_CHANNEL=64
        ;;
    paul)
        STATIC_EXP=paul_20260901
        # The combined session shares 89 stimuli with the image session; the
        # single-day paul_20260831 shares only 73.
        DYNAMIC_EXP=paul_20260831to0902
        FIRST_CHANNEL=1
        LAST_CHANNEL=64
        ;;
    *)
        echo "Unknown monkey ${MONKEY}; expected baby1, red, or paul." >&2
        exit 1
        ;;
esac
shift || true

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=("${DEFAULT_MODELS[@]}")
fi

for model_name in "${MODELS[@]}"; do
    echo "===== ${MONKEY} / ${model_name} ====="
    # The baselines were extracted without pooling; every network used mean.
    case "${model_name}" in
        pixel_values_*|optical_flow_*) POOLING=none ;;
        *) POOLING=mean ;;
    esac
    "${PYTHON}" "${PROJECT_ROOT}/python_scripts/scripts/run_static_dynamic_neural_model_rsa.py" \
        --monkey_name "${MONKEY}" \
        --static_experiment_name "${STATIC_EXP}" \
        --dynamic_experiment_name "${DYNAMIC_EXP}" \
        --good_channels "${FIRST_CHANNEL}" "${LAST_CHANNEL}" \
        --reliable_channels_config "${PROJECT_ROOT}/reliable_channels.yaml" \
        --reliable_channels_key "${STATIC_EXP}" \
        --signal_rdm_metric cosine_cnt \
        --model_rdm_metric cosine_cnt \
        --rsa_metric spearman \
        --new_fs 100 \
        --static_crop_ms 1000 \
        --model_pooling "${POOLING}" \
        --model_name "${model_name}" \
        || echo "FAILED ${MONKEY} ${model_name}"
done
