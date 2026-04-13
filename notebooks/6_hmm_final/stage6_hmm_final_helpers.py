from __future__ import annotations

import contextlib
import io
import json
import re
from pathlib import Path
from typing import Any, Iterable


def _display_fallback(obj: Any) -> None:
    try:
        from IPython.display import display as ipy_display

        ipy_display(obj)
    except Exception:
        print(obj)


def path_expr(path: str | Path | None) -> str:
    if path is None:
        return "None"
    return f"Path({str(Path(path))!r})"


def literal_expr(value: Any) -> str:
    if isinstance(value, Path):
        return path_expr(value)
    if value is None:
        return "None"
    return repr(value)


def ensure_configured_path(path_value: str | Path, label: str, *, must_exist: bool = False) -> Path:
    path = Path(path_value)
    if "<SET_" in str(path):
        raise ValueError(f"{label} still uses a placeholder path. Edit the notebook before running.")
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _load_notebook_cells(source_notebook: str | Path) -> list[tuple[int, dict[str, Any]]]:
    source_notebook = Path(source_notebook)
    if not source_notebook.exists():
        raise FileNotFoundError(f"Notebook not found: {source_notebook}")

    data = json.loads(source_notebook.read_text(encoding="utf-8"))
    return list(enumerate(data.get("cells", [])))


def _patch_assignment_block(source: str, key: str, expr: str) -> tuple[str, int]:
    pattern = re.compile(
        rf"(?ms)^({re.escape(key)}\s*=\s*).*?(?=^\S|\Z)"
    )
    return pattern.subn(rf"\1{expr}\n", source, count=1)


def execute_source_notebook(
    source_notebook: str | Path,
    *,
    assignment_overrides: dict[str, str] | None = None,
    include_notebook_cells: Iterable[int] | None = None,
    suppress_first_code_output: bool = True,
) -> dict[str, Any]:
    source_notebook = Path(source_notebook)
    assignment_overrides = assignment_overrides or {}
    include_notebook_cells = set(include_notebook_cells) if include_notebook_cells is not None else None

    notebook_cells = _load_notebook_cells(source_notebook)
    selected_code_cells: list[tuple[int, str]] = []
    for idx, cell in notebook_cells:
        if cell.get("cell_type") != "code":
            continue
        if include_notebook_cells is not None and idx not in include_notebook_cells:
            continue
        selected_code_cells.append((idx, "".join(cell.get("source", []))))

    if not selected_code_cells:
        raise ValueError(f"No selected code cells found in {source_notebook}")

    patched_counts = {key: 0 for key in assignment_overrides}
    patched_cells: list[tuple[int, str]] = []
    for idx, source in selected_code_cells:
        patched = source
        for key, expr in assignment_overrides.items():
            patched, count = _patch_assignment_block(patched, key, expr)
            patched_counts[key] += count
        patched_cells.append((idx, patched))

    missing = [key for key, count in patched_counts.items() if count == 0]
    if missing:
        raise ValueError(
            f"Could not patch assignment(s) {missing!r} in {source_notebook.name}. "
            "This usually means the preserved source notebook changed shape."
        )

    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "display": _display_fallback,
    }

    print("Using source notebook:", source_notebook.name)
    for key in sorted(assignment_overrides):
        print(f"  override: {key}")

    for order, (cell_index, source) in enumerate(patched_cells):
        compiled_name = f"{source_notebook.name}::cell{cell_index}"
        if suppress_first_code_output and order == 0:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                exec(compile(source, compiled_name, "exec"), namespace)
        else:
            exec(compile(source, compiled_name, "exec"), namespace)

    return namespace


def run_stage6_final_fit_source(
    *,
    source_notebook: str | Path,
    final_root: str | Path,
    out_root: str | Path,
    manifest_tsv: str | Path | None = None,
    k_final: int = 3,
    data_variant: str = "intermediate",
    feature_mode: str = "nolags",
    minlen: int = 15,
    gpu_memory_limit_mb: int | None = None,
    save_gamma: bool = True,
    do_viterbi: bool = True,
) -> dict[str, Any]:
    overrides = {
        "K_FINAL": literal_expr(int(k_final)),
        "DATA_VARIANT": literal_expr(data_variant),
        "FEATURE_MODE": literal_expr(feature_mode),
        "MINLEN": literal_expr(int(minlen)),
        "FINAL_ROOT": path_expr(final_root),
        "MANIFEST_TSV": path_expr(manifest_tsv),
        "OUT_ROOT": path_expr(out_root),
        "GPU_MEMORY_LIMIT_MB": literal_expr(gpu_memory_limit_mb),
        "SAVE_GAMMA": literal_expr(bool(save_gamma)),
        "DO_VITERBI": literal_expr(bool(do_viterbi)),
    }
    return execute_source_notebook(source_notebook, assignment_overrides=overrides)


def run_stage6_review_source(
    *,
    source_notebook: str | Path,
    result_root: str | Path,
    output_figure_dir: str | Path,
) -> dict[str, Any]:
    overrides = {
        "RESULT_ROOT": path_expr(result_root),
        "OUT_FIG_DIR": path_expr(output_figure_dir),
    }
    return execute_source_notebook(source_notebook, assignment_overrides=overrides)


def run_stage6_state_physiology_source(
    *,
    source_notebook: str | Path,
    result_root: str | Path,
    templateflow_home: str | Path,
    output_dir: str | Path,
    parcel_labels_file: str | Path | None = None,
    preproc_params_file: str | Path | None = None,
    reference_state: int | None = None,
) -> dict[str, Any]:
    overrides = {
        "RESULT_ROOT": path_expr(result_root),
        "TEMPLATEFLOW_HOME": path_expr(templateflow_home),
        "FIG_DIR": path_expr(output_dir),
        "PARCEL_LABELS_FILE": path_expr(parcel_labels_file),
        "PREPROC_PARAMS_FILE": path_expr(preproc_params_file),
        "REFERENCE_STATE": literal_expr(reference_state),
    }
    return execute_source_notebook(source_notebook, assignment_overrides=overrides)


def run_stage6_crossmodal_source(
    *,
    source_notebook: str | Path,
    result_root: str | Path,
    templateflow_root: str | Path,
    output_dir: str | Path,
    parcel_labels_file: str | Path | None = None,
    reference_state_override: str | None = None,
) -> dict[str, Any]:
    templateflow_root = Path(templateflow_root)
    default_tsv = templateflow_root / "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg.tsv"
    default_txt = templateflow_root / "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg.txt"

    overrides = {
        "RESULT_ROOT": path_expr(result_root),
        "FINAL_DIR": path_expr(Path(result_root) / "final"),
        "DEFAULT_SCHAEFER_TSV": path_expr(default_tsv),
        "DEFAULT_BRAINSTORM_TXT": path_expr(default_txt),
        "PARCEL_LABELS_FILE": path_expr(parcel_labels_file),
        "OUT_DIR": path_expr(output_dir),
        "REFERENCE_STATE_OVERRIDE": literal_expr(reference_state_override),
    }

    include_cells = [1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14]
    return execute_source_notebook(
        source_notebook,
        assignment_overrides=overrides,
        include_notebook_cells=include_cells,
    )


def run_stage6_brainmaps_source(
    *,
    source_notebook: str | Path,
    result_root: str | Path,
    templateflow_home: str | Path,
    output_dir: str | Path,
    reference_state: int = 2,
    contrast_states: list[int] | None = None,
    include_option_b: bool = False,
) -> dict[str, Any]:
    overrides = {
        "RESULT_ROOT": path_expr(result_root),
        "TEMPLATEFLOW_HOME": path_expr(templateflow_home),
        "OUT_DIR": path_expr(output_dir),
        "REFERENCE_STATE": literal_expr(int(reference_state)),
        "CONTRAST_STATES": literal_expr(list(contrast_states or [1, 3])),
    }

    include_cells = [1]
    if include_option_b:
        include_cells.append(2)

    return execute_source_notebook(
        source_notebook,
        assignment_overrides=overrides,
        include_notebook_cells=include_cells,
    )


def run_stage6_panel_export_source(
    *,
    source_notebook: str | Path,
    result_root: str | Path,
    templateflow_home: str | Path,
    output_dir: str | Path,
    parcel_labels_file: str | Path | None = None,
    reference_state_override: str | None = None,
    top_n_bars: int = 10,
) -> dict[str, Any]:
    overrides = {
        "RESULT_ROOT": path_expr(result_root),
        "TEMPLATEFLOW_HOME": path_expr(templateflow_home),
        "PARCEL_LABELS_FILE": path_expr(parcel_labels_file),
        "REFERENCE_STATE_OVERRIDE": literal_expr(reference_state_override),
        "TOP_N_BARS": literal_expr(int(top_n_bars)),
        "OUT_DIR": path_expr(output_dir),
    }
    return execute_source_notebook(source_notebook, assignment_overrides=overrides)
