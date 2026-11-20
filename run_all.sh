#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

python_bin="${GEOMETRY_PYTHON:-python}"
data_dir="${GEOMETRY_DATA:-Datasets/Cache}"
reproduction="${GEOMETRY_REPRODUCTION:-Results/Reproduction}"
workers="${GEOMETRY_JOBS:-1}"
threads="${GEOMETRY_THREADS:-1}"
export OPENBLAS_NUM_THREADS="$threads" OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"

make_report() {
    "$python_bin" -m Experiments.analyze --results "$1"
    "$python_bin" -m Experiments.expanded --results "$1"
    "$python_bin" -m Experiments.figures --results "$1"
    "$python_bin" -m Experiments.report --results "$1"
}

case "${1:-report}" in
    test)
        GEOMETRY_TEST_DATA="$data_dir" "$python_bin" -m unittest discover -s tests -v
        ;;
    report)
        make_report Results
        ;;
    reproduce)
        if [[ "${GEOMETRY_SPLINE_BACKEND:-native_cpu}" == "pytorch" ]]; then
            unset TEMPO_KAN_NATIVE
        elif [[ -z "${TEMPO_KAN_NATIVE:-}" ]]; then
            TEMPO_KAN_NATIVE="$("$python_bin" -m Proposed_Method.native --output "$reproduction/Build/geometry_spline.so")"
            export TEMPO_KAN_NATIVE
        fi
        "$python_bin" -m Experiments.run --plan Experiments/Plans/All.json --data-dir "$data_dir" \
            --output "$reproduction" --jobs "$workers" --threads "$threads"
        "$python_bin" -m Experiments.stress --results "$reproduction" --data-dir "$data_dir" --jobs "$workers"
        "$python_bin" -m Experiments.stress_sweeps --results "$reproduction" --data-dir "$data_dir" --jobs "$workers"
        make_report "$reproduction"
        ;;
    *)
        printf 'Usage: bash run_all.sh {test|report|reproduce}\n' >&2
        exit 2
        ;;
esac
