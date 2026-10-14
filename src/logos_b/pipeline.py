"""Raw synthetic observations -> fitted models -> matched controls -> verified outputs."""

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import numpy as np
import scipy
from . import __version__
from . import fitting as f
from .models import RemModel, WarModel
from .continuation import solve_floor
from .theoretical import run_frontier


def save(path, obj):
    Path(path).write_text(
        json.dumps(
            obj,
            indent=2,
            allow_nan=False,
            default=lambda x: (
                x.item()
                if isinstance(x, np.generic)
                else (_ for _ in ()).throw(TypeError(type(x).__name__))
            ),
        )
        + "\n"
    )


def table(path, rows):
    with Path(path).open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_dataset(data):
    if data.get("synthetic") is not True or data.get("schema_version") != 1:
        raise ValueError("This release accepts only its declared synthetic schema")
    ids = set()
    for key, channels in [
        ("remifentanil", ["cp_ng_ml", "eeg"]),
        ("warfarin", ["cp_mg_L", "pca_percent"]),
    ]:
        if not data.get(key):
            raise ValueError(f"Empty {key} cohort")
        for row in data[key]:
            if row["ID"] in ids or not row["ID"].startswith("SYN-"):
                raise ValueError("Unique synthetic IDs required")
            ids.add(row["ID"])
            t = np.asarray(row["time"], float)
            if (
                len(t) < 8
                or not np.isfinite(t).all()
                or t[0] != 0
                or np.any(np.diff(t) <= 0)
            ):
                raise ValueError("Invalid observation times")
            for ch in channels:
                y = np.asarray(row[ch], float)
                if len(y) != len(t) or not np.isfinite(y).all() or np.any(y < 0):
                    raise ValueError(f"Invalid {ch}")
            design = (
                ["T_min", "rate_ug_min"]
                if key == "remifentanil"
                else ["dose_mg", "ka_h"]
            )
            if any(not np.isfinite(row[k]) or row[k] <= 0 for k in design):
                raise ValueError("Invalid positive design value")
            if key == "warfarin" and (
                not np.isfinite(row["lag_h"]) or row["lag_h"] < 0
            ):
                raise ValueError("Invalid lag")


def fit_profiles(data):
    rem = []
    war = []
    diagnostics = []
    for row in data["remifentanil"]:
        t = np.array(row["time"])
        w, k, pk = f.fit_rem_pk(
            t, np.array(row["cp_ng_ml"]), row["rate_ug_min"], row["T_min"]
        )
        pd = f.fit_rem_pd(
            t, np.array(row["eeg"]), w, k, row["rate_ug_min"], row["T_min"]
        )
        if not pk.success or not pd.success:
            raise RuntimeError("Synthetic remifentanil fit failed")
        ke, K, h = np.exp(pd.x[:3])
        rem.append(
            dict(
                ID=row["ID"],
                T_min=row["T_min"],
                rate_ug_min=row["rate_ug_min"],
                w1=float(w[0]),
                w2=float(w[1]),
                k1_min=float(k[0]),
                k2_min=float(k[1]),
                ke0_min=float(ke),
                K_ng_ml=float(K),
                Hill=float(h),
                E0=float(pd.x[3]),
                f=float(pd.x[4]),
            )
        )
        diagnostics.append(
            dict(
                ID=row["ID"],
                pk_success=bool(pk.success),
                pd_success=bool(pd.success),
                pk_loss=float(pk.fun @ pk.fun),
                pd_rmse=float(np.sqrt(np.mean(pd.fun**2))),
                pd_jacobian_condition=float(np.linalg.cond(pd.jac)),
            )
        )
    for row in data["warfarin"]:
        t = np.array(row["time"])
        pk = f.fit_war_pk(
            t, np.array(row["cp_mg_L"]), row["dose_mg"], row["ka_h"], row["lag_h"]
        )
        pd = f.fit_war_pd(
            t,
            np.array(row["pca_percent"]),
            pk.x,
            row["dose_mg"],
            row["ka_h"],
            row["lag_h"],
        )
        if not pk.success or not pd.success:
            raise RuntimeError("Synthetic warfarin fit failed")
        k, scale = np.exp(pk.x)
        ko, K = np.exp(pd.x)
        war.append(
            dict(
                ID=row["ID"],
                dose_mg=row["dose_mg"],
                ka_h=row["ka_h"],
                lag_h=row["lag_h"],
                k_h=float(k),
                scale_L_inv=float(scale),
                kout_h=float(ko),
                IC50_mg_L=float(K),
            )
        )
        diagnostics.append(
            dict(
                ID=row["ID"],
                pk_success=bool(pk.success),
                pd_success=bool(pd.success),
                pk_loss=float(pk.fun @ pk.fun),
                pd_rmse=float(np.sqrt(np.mean(pd.fun**2))),
                pd_jacobian_condition=float(np.linalg.cond(pd.jac)),
            )
        )
    return rem, war, diagnostics


def verify_candidate(drug, r):
    if not r["solver_success"]:
        raise RuntimeError("Optimizer did not converge")
    if r["amount_error"] > 1e-8:
        raise RuntimeError("Amount mismatch")
    if r["Cp_ratio"] > 1 + 1e-7:
        raise RuntimeError("Plasma cap exceeded")
    if r["independent_objective_error"] > (1e-7 if drug == "Remifentanil" else 1e-6):
        raise RuntimeError("Objective verification failed")
    if drug == "Remifentanil":
        if (
            r["Ce_ratio"] > 1 + 1e-7
            or r["tail_ratio"] > 1 + 1e-7
            or r["independent_tail_error"] > 1e-7
        ):
            raise RuntimeError("Effect-site verification failed")
    gap = (
        r.get("sampled_relaxation_supporting_gap")
        if drug == "Remifentanil"
        else r.get("gap")
    )
    if gap is None or not np.isfinite(gap):
        raise RuntimeError("Missing concave support bound")


def run(dataset, output):
    dataset = Path(dataset)
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be absent or empty")
    data = json.loads(dataset.read_text())
    validate_dataset(data)
    out.mkdir(parents=True, exist_ok=True)
    rem, war, diagnostics = fit_profiles(data)
    save(out / "fitted_models.json", dict(remifentanil=rem, warfarin=war))
    save(out / "fit_diagnostics.json", diagnostics)
    candidates = []
    increments = []
    stopping = []
    stopcontrols = []
    sens = []
    for drug, profiles, ctor, classes in [
        ("Remifentanil", rem, RemModel, [1.0, 1.25, 1.5, 2.0]),
        ("Warfarin", war, WarModel, [0, 24, 12, 6]),
    ]:
        for m in profiles:
            print(f'{drug} {m["ID"]}: fitting and matched comparisons', flush=True)
            model = ctor(m)
            prev = None
            for cl in classes:
                if drug == "Warfarin" and cl == 0:
                    J = model.objective([0], [1])
                    r = dict(
                        ID=m["ID"],
                        J=J,
                        J_ref=J,
                        gain_percent=0.0,
                        gap=0.0,
                        independent_objective_error=0.0,
                        reference=True,
                    )
                else:
                    r = (
                        model.solve(cap=cl)
                        if drug == "Remifentanil"
                        else model.solve(spacing=cl)
                    )
                    verify_candidate(drug, r)
                gap = r.get("sampled_relaxation_supporting_gap", r.get("gap", 0.0))
                candidates.append(dict(drug=drug, regimen_class=cl, **r))
                if prev is not None:
                    delta = r["J"] - prev["J"]
                    lower = (
                        delta
                        - prev["gap"]
                        - r["independent_objective_error"]
                        - prev["error"]
                    )
                    if delta < -1e-7:
                        raise RuntimeError("Nested candidate values decreased")
                    increments.append(
                        dict(
                            drug=drug,
                            ID=m["ID"],
                            transition=f'{prev["class"]}->{cl}',
                            delta_J=delta,
                            lower_bound=lower,
                            positive_lower_bound=lower > 1e-6,
                            unit="min" if drug == "Remifentanil" else "h",
                        )
                    )
                prev = dict(
                    J=r["J"],
                    gap=gap,
                    error=r["independent_objective_error"],
                    **{"class": cl},
                )
            free = solve_floor(model, 0.0, drug)
            for eps in [0.05, 0.25]:
                cont = solve_floor(model, eps, drug)
                delta = free["J"] - cont["J"]
                lower = (
                    delta
                    - cont["supporting_gap"]
                    - free["independent_objective_error"]
                    - cont["independent_objective_error"]
                )
                rr = dict(
                    drug=drug,
                    ID=m["ID"],
                    epsilon=eps,
                    delta_J=delta,
                    lower_bound=lower,
                    positive_lower_bound=lower > 1e-6,
                    committed_percent=100
                    * (eps / 4 if drug == "Remifentanil" else 4 * eps / 13),
                    unit="min" if drug == "Remifentanil" else "h",
                )
                stopping.append(rr)
                stopcontrols.append(dict(**rr, free=free, continuation=cont))
            if drug == "Remifentanil":
                hill = model.solve(h=m["Hill"])
                initial = RemModel(m, initial_fraction=0.01).solve()
                for name, r in [("fitted_Hill", hill), ("initial_state_1pct", initial)]:
                    if name == "initial_state_1pct":
                        verify_candidate(drug, r)
                    elif (
                        not r["solver_success"]
                        or r["amount_error"] > 1e-8
                        or r["max_excess"] > 1e-7
                        or r["independent_objective_error"] > 1e-7
                    ):
                        raise RuntimeError("Hill candidate verification failed")
                    sens.append(dict(scenario=name, **r))
    frontier, controls = run_frontier()
    save(out / "regimen_controls.json", candidates)
    save(out / "continuation_controls.json", stopcontrols)
    save(out / "sensitivity_controls.json", sens)
    save(out / "theoretical_controls.json", controls)
    table(out / "paired_increments.csv", increments)
    table(out / "continuation_increments.csv", stopping)
    table(out / "theoretical_dose_frontier.csv", frontier)
    # Conditional profile bootstrap: fitted parameters held fixed, no residual refitting.
    summary = []
    rng = np.random.default_rng(data["seed"])
    for drug in ["Remifentanil", "Warfarin"]:
        transitions = list(
            dict.fromkeys(x["transition"] for x in increments if x["drug"] == drug)
        )
        for tr in transitions:
            rr = [x for x in increments if x["drug"] == drug and x["transition"] == tr]
            values = np.array([x["delta_J"] for x in rr])
            bs = np.median(
                rng.choice(values, (1000, len(values)), replace=True), axis=1
            )
            summary.append(
                dict(
                    drug=drug,
                    transition=tr,
                    n=len(rr),
                    median_delta_J=float(np.median(values)),
                    positive_lower_bounds=sum(x["positive_lower_bound"] for x in rr),
                    conditional_profile_bootstrap_low=float(np.quantile(bs, 0.025)),
                    conditional_profile_bootstrap_high=float(np.quantile(bs, 0.975)),
                    unit=rr[0]["unit"],
                )
            )
    table(out / "increment_summary.csv", summary)
    make_figures(out, increments, stopping, frontier, candidates, rem, war)
    save(
        out / "verification.json",
        dict(
            all_passed=True,
            scope="Synthetic fitting, matched controls, independent drug ODE objectives, full-path exposure searches, theoretical DOP853/Radau and quadrature checks",
            profiles_per_model=data["profiles_per_model"],
            candidate_count=len(candidates),
            continuation_comparisons=len(stopping),
            theoretical_targets=len(frontier),
        ),
    )
    source_hashes = {
        str(p.relative_to(Path(__file__).parent)): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    outputs = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.iterdir())
        if p.is_file()
    }
    save(
        out / "run_manifest.json",
        dict(
            version=__version__,
            seed=data["seed"],
            synthetic=True,
            input_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
            python=sys.version.split()[0],
            numpy=np.__version__,
            scipy=scipy.__version__,
            platform=platform.platform(),
            source_sha256=source_hashes,
            output_sha256=outputs,
        ),
    )


def make_figures(out, increments, stopping, frontier, candidates, rem, war):
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "logos-b-mpl")
    )
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update({"svg.hashsalt": "logos-b-0.5.0", "font.size": 10})

    from .figure_metadata import strip_figure_metadata

    def export(fig, name):
        fig.tight_layout()
        fig.savefig(
            out / f"{name}.png", dpi=160, metadata={"Software": None}
        )
        fig.savefig(out / f"{name}.svg", metadata={"Date": None})
        strip_figure_metadata(out / f"{name}.png")
        strip_figure_metadata(out / f"{name}.svg")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, drug in zip(axes, ["Remifentanil", "Warfarin"]):
        trs = list(
            dict.fromkeys(r["transition"] for r in increments if r["drug"] == drug)
        )
        ax.boxplot(
            [
                [
                    r["delta_J"]
                    for r in increments
                    if r["drug"] == drug and r["transition"] == tr
                ]
                for tr in trs
            ],
            positions=range(len(trs)),
            showfliers=False,
            widths=0.5,
        )
        for j, tr in enumerate(trs):
            vals = [
                r["delta_J"]
                for r in increments
                if r["drug"] == drug and r["transition"] == tr
            ]
            ax.scatter(np.full(len(vals), j), vals, s=28)
            ax.plot([j - 0.2, j + 0.2], [np.median(vals)] * 2, color="black")
        ax.set_xticks(range(len(trs)), trs)
        ax.set_ylabel(
            "Paired proxy increment ("
            + ("min" if drug == "Remifentanil" else "h")
            + ")"
        )
        ax.set_title(drug + " — synthetic")
    export(fig, "figure_01_paired_increments")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, drug in zip(axes, ["Remifentanil", "Warfarin"]):
        for eps in [0.05, 0.25]:
            vals = [
                r["delta_J"]
                for r in stopping
                if r["drug"] == drug and r["epsilon"] == eps
            ]
            ax.scatter(np.full(len(vals), eps), vals)
        ax.set_xlabel("Final-quarter floor ε")
        ax.set_ylabel(
            "Stopping increment (" + ("min" if drug == "Remifentanil" else "h") + ")"
        )
        ax.set_title(drug + " — synthetic")
    export(fig, "figure_02_continuation")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        [r["target"] for r in frontier],
        [r["realized_D"] for r in frontier],
        "o-",
        label="Flexible finite family",
    )
    rr = [r for r in frontier if r["constant_feasible"]]
    ax.plot(
        [r["target"] for r in rr],
        [r["constant_minimum_dose"] for r in rr],
        "s--",
        label="Safe constant family",
    )
    ax.set_xlabel("Benefit target (dimensionless)")
    ax.set_ylabel("Administered amount (dimensionless)")
    ax.set_title("Theoretical fixed-target dose frontier")
    ax.legend()
    export(fig, "figure_03_dose_frontier")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rm = RemModel(rem[0])
    rr = next(
        r
        for r in candidates
        if r["drug"] == "Remifentanil"
        and r["ID"] == rem[0]["ID"]
        and r["regimen_class"] == 2.0
    )
    tt = np.linspace(0, rm.W + 30, 300)
    axes[0].plot(tt, rm.values(tt, np.ones(rm.n), "ce"), label="Reference")
    axes[0].plot(tt, rm.values(tt, rr["controls_rate_factors"], "ce"), label="Flexible")
    axes[0].set_xlabel("Time (min)")
    axes[0].set_ylabel("Effect-site concentration (ng/mL)")
    axes[0].set_title("Synthetic remifentanil")
    axes[0].legend()
    wm = WarModel(war[0])
    ww = next(
        r
        for r in candidates
        if r["drug"] == "Warfarin"
        and r["ID"] == war[0]["ID"]
        and r["regimen_class"] == 6
    )
    tt = np.linspace(0, 120, 241)
    axes[1].plot(tt, 100 * wm.response(tt), label="Reference")
    axes[1].plot(
        tt,
        100 * wm.response(tt, ww["dose_times"], ww["dose_fractions"]),
        label="Flexible",
    )
    axes[1].set_xlabel("Time (h)")
    axes[1].set_ylabel("PCA (%)")
    axes[1].set_title("Synthetic warfarin")
    axes[1].legend()
    export(fig, "figure_04_trajectories")
