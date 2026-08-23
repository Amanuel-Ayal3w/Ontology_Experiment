#!/usr/bin/env bash
# Experiment status dashboard. Usage: ./status.sh [watch]
BEAM=.venv/bin/beam
cd "$(dirname "$0")"

show() {
  echo "════════ ONTO-SELECT STATUS  $(date +%H:%M:%S) ════════"

  echo; echo "── Beam tasks (top = newest; PENDING = waiting for a GPU) ──"
  $BEAM task list 2>/dev/null | head -14 | tail -10

  echo; echo "── Beam client processes running locally ──"
  ps aux | grep "beam_app.py" | grep -v grep | awk '{print "  ", $NF, "(elapsed", $10")"}' \
    || echo "   none"

  echo; echo "── Finished runs on Beam volume ──"
  $BEAM ls onto-select/results/runs 2>/dev/null | grep -E "No|items" | head -12 \
    || echo "   none yet"

  echo; echo "── Local gates ──"
  [ -f results/tags/diagnostics.json ] && \
    echo "   gate 1 (density): PASSED  ($(python3 -c "import json;d=json.load(open('results/tags/diagnostics.json'));print(f\"{d['mean_tags_per_doc']} tags/doc, {d['zero_tag_rate']:.0%} zero-tag\")"))"
  [ -f results/tags/kstar.json ] && \
    echo "   gate 2 (k*):      PASSED  ($(python3 -c "import json;d=json.load(open('results/tags/kstar.json'));print(f\"observed k*={d['k_star_observed']} docs\")"))"
  n=$(ls results/selections/*_1000000_*.json 2>/dev/null | grep -vc cost)
  echo "   selections:       $n / 12 cut at budget 1M"
  if ls results/runs/*/metrics.json >/dev/null 2>&1; then
    echo; echo "── Downloaded run metrics ──"
    for m in results/runs/*/metrics.json; do
      python3 -c "import json;d=json.load(open('$m'));print(f\"   {d['arm']:9} seed {d['seed']}  loss={d.get('final_loss','?')}  acc={d.get('accuracy_overall','pending')}  tail={d.get('accuracy_tail','pending')}\")"
    done
  fi
  echo
}

if [ "$1" = "watch" ]; then
  while true; do clear; show; sleep 20; done
else
  show
fi
