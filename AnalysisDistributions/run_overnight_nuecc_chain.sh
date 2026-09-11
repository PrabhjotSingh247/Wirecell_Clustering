#!/bin/bash
# Overnight chain: run the 7 CosmicTagger Draw_* notebooks + the unmatched-true-
# neutrino investigation on the first 10 nuecc chunks (chunk_00..chunk_09,
# ~1000 events, one combined run each). Each target runs to completion; a failure
# is logged and the chain moves on. Status: <RUNDIR>/STATUS.txt.
set -u
REPO="/Users/prabhjotsingh/Experiments/SBND/WireCell_Reconstruction"
ND="$REPO/AnalysisDistributions"
RUNDIR="$ND/multi_file_plots_charge_light_matching/overnight_nuecc_10chunks_20260904"
mkdir -p "$RUNDIR"
STATUS="$RUNDIR/STATUS.txt"

{
  echo "OVERNIGHT nuecc 10-chunk chain -- started $(date)"
  echo "chunks: chunk_00..chunk_09 (~1000 events), one combined run per notebook"
  echo "outputs: each notebook's own multi_file_plots_charge_light_matching/<name>/combined_apa_<ts>/"
  echo
} > "$STATUS"

run_nb () {
  local nb="$1" t0 rc dt out
  t0=$(date +%s)
  echo "[$(date +%H:%M:%S)] START  $nb" | tee -a "$STATUS"
  ( cd "$ND" && python3 -m jupyter nbconvert --to notebook --execute \
      --ExecutePreprocessor.timeout=-1 \
      --output "$RUNDIR/${nb}.executed.ipynb" "${nb}.ipynb" ) > "$RUNDIR/${nb}.log" 2>&1
  rc=$?
  dt=$(( ($(date +%s)-t0)/60 ))
  if [ $rc -eq 0 ] && ! grep -qE "Traceback \(most recent call last\)|CellExecutionError" "$RUNDIR/${nb}.log"; then
    out=$(ls -dt "$ND"/multi_file_plots_charge_light_matching/"$nb"/combined_apa_* 2>/dev/null | head -1)
    echo "[$(date +%H:%M:%S)] PASS   $nb  (${dt}m)" | tee -a "$STATUS"
    echo "    -> ${out:-<no combined_apa dir found>}" >> "$STATUS"
  else
    echo "[$(date +%H:%M:%S)] FAIL   $nb  (${dt}m, rc=$rc)" | tee -a "$STATUS"
    grep -A25 "Traceback (most recent call last)" "$RUNDIR/${nb}.log" | head -30 | sed 's/^/    /' >> "$STATUS"
    tail -n 8 "$RUNDIR/${nb}.log" | sed 's/^/    | /' >> "$STATUS"
  fi
  echo >> "$STATUS"
}

for nb in \
  Draw_InVolumeSignal_Removed_After_CosmicTagger \
  Draw_InVolumeSignal_Removed_Before_CosmicTagger \
  Draw_InVolumeSignal_Survived_After_CosmicTagger \
  Draw_OutOfVolumeNeutrinos_Removed_After_CosmicTagger \
  Draw_OutOfVolumeNeutrinos_Survived_After_CosmicTagger \
  Draw_Cosmics_Removed_After_CosmicTagger \
  Draw_Cosmics_Survived_After_CosmicTagger ; do
  run_nb "$nb"
done

# --- investigate_unmatched_true_neutrinos.py (standalone) --------------------
t0=$(date +%s)
echo "[$(date +%H:%M:%S)] START  investigate_unmatched_true_neutrinos.py" | tee -a "$STATUS"
( cd "$REPO" && python3 investigate_unmatched_true_neutrinos.py ) > "$RUNDIR/investigate.log" 2>&1
rc=$?
dt=$(( ($(date +%s)-t0)/60 ))
if [ $rc -eq 0 ] && ! grep -qE "Traceback \(most recent call last\)" "$RUNDIR/investigate.log"; then
  echo "[$(date +%H:%M:%S)] PASS   investigate  (${dt}m)" | tee -a "$STATUS"
  out=$(ls -dt "$REPO"/multi_file_plots_charge_light_matching/unmatched_true_neutrino_investigation/*/ 2>/dev/null | head -1)
  echo "    -> ${out:-<no output dir>}" >> "$STATUS"
else
  echo "[$(date +%H:%M:%S)] FAIL   investigate  (${dt}m, rc=$rc)" | tee -a "$STATUS"
  grep -A25 "Traceback (most recent call last)" "$RUNDIR/investigate.log" | head -30 | sed 's/^/    /' >> "$STATUS"
  tail -n 8 "$RUNDIR/investigate.log" | sed 's/^/    | /' >> "$STATUS"
fi

{ echo; echo "CHAIN COMPLETE $(date)"; } >> "$STATUS"
