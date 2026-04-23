# Repository Scope

This repository is the public workflow package for the manuscript:

**Fusion hidden Markov modeling reveals a reproducible shared state architecture in simultaneous resting-state EEG-fMRI**

## Included In This Public Release

- the active Stage 1 to Stage 6 `stepNN_*` workflow files
- the active MATLAB helpers used by Stage 1 and Stage 2
- the active Python helper and backend modules used by Stages 3 to 6
- manual-step markdown files that are part of the real user workflow
- core manuscript-facing documentation under `docs/`
- Python environment templates in `environment.yml` and `requirements.txt`

## Not Included In This Public Release

- raw data
- large derived outputs
- historical development notebooks and scripts
- internal refactor-planning and validation memos
- scratch folders, smoke-test artifacts, notebook checkpoints, and caches
- empty stage folders that were not part of the intended public workflow

## Practical Scope

This repository is meant to help a careful outside scientist:

1. understand the manuscript workflow,
2. identify which steps are manual or hybrid,
3. run the active analysis sequence in the intended order,
4. reproduce the manuscript-facing intermediate outputs and summary products,
5. trace how figures and tables map onto the workflow.

It is not intended to be:

- a raw-data repository
- a complete dump of historical notebook development
- a fully turnkey push-button pipeline that hides the manual Brainstorm steps

## Implemented Stage Structure

The public release uses six active stages:

1. `notebooks/1_eeg_sensor/`
2. `notebooks/2_eeg_source/`
3. `notebooks/3_bold/`
4. `notebooks/4_alignment/`
5. `notebooks/5_hmm_selection/`
6. `notebooks/6_hmm_final/`

Later manuscript summaries and figure-support products remain embedded in these stages where that is clearest for public use.
