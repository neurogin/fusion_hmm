# GitHub Release Checklist

Use this checklist before the first public push.

## Public Tree Check

- Confirm that only the active Stage 1 to Stage 6 workflow files remain under `notebooks/`
- Confirm that no old development notebooks or compatibility stubs remain in the public tree
- Confirm that no notebook checkpoints, cache folders, smoke-test artifacts, or scratch scripts remain

## Documentation Check

- Review [README.md](../README.md)
- Review [docs/setup.md](setup.md)
- Review [docs/public_workflow.md](public_workflow.md)
- Review [docs/manual_steps.md](manual_steps.md)
- Review [docs/methods_map.md](methods_map.md)
- Review [docs/final_dataset_spec.md](final_dataset_spec.md)
- Review [docs/reproducibility_notes.md](reproducibility_notes.md)
- Review [docs/figure_table_map.md](figure_table_map.md)

## Data And Output Check

- Confirm that raw data are not present in the repository
- Confirm that large derived outputs are not present in the repository
- Confirm that local derivatives, QC outputs, model fits, and panel exports remain outside version control

## Environment Check

- Confirm that `environment.yml` and `requirements.txt` still match the active public workflow
- Confirm that the setup notes still describe the real MATLAB, Brainstorm, and Python dependencies
- Confirm that the Stage-5 and Stage-6 TensorFlow or `osl_dynamics` caveats are still accurate for the intended runtime environment

## Final Metadata Check

- Confirm the repository license choice before adding a `LICENSE` file
- Confirm the author list and citation wording before adding `CITATION.cff`

## Final Readability Check

- Open one step file from each stage and verify that the public instructions are still easy to follow
- Confirm that manual or hybrid boundaries remain explicit and honest
- Confirm that the README stage order matches the actual `stepNN_*` files on disk
