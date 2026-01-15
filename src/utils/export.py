# src/utils/export.py

"""Export detection results to Excel."""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OutputManager:
    """Export change detection results to Excel with multiple sheets."""

    def __init__(self, output_dir: str, config: Optional[Dict[str, Any]] = None):
        self.output_dir = output_dir
        self.config = config or {}
        os.makedirs(output_dir, exist_ok=True)

    def export_to_csv(
        self,
        detection_results: Dict[str, Any],
        true_change_points: List[int],
        individual_trials: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Export detection results to Excel file."""
        if not individual_trials or not any(individual_trials):
            logger.warning("No individual trial results to export")
            return

        try:
            n_timesteps = self._get_timestep_count()
            excel_path = os.path.join(self.output_dir, "detection_results.xlsx")

            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                trial_dfs = []
                for i, trial in enumerate(individual_trials):
                    if trial:
                        df = self._create_trial_df(trial, true_change_points, n_timesteps)
                        df.to_excel(writer, sheet_name=f"Trial{i+1}", index=False)
                        trial_dfs.append(df)

                if individual_trials:
                    self._create_summary_sheet(individual_trials, true_change_points, writer)
                    self._create_details_sheet(individual_trials, true_change_points, writer)
                    if len(trial_dfs) > 1:
                        self._create_aggregate_sheet(trial_dfs, true_change_points, writer)

            logger.info(f"Results saved to {excel_path}")
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            raise

    def _get_timestep_count(self) -> int:
        """Get sequence length from config."""
        if "model" in self.config and "params" in self.config:
            params = self.config["params"]
            if hasattr(params, "seq_len"):
                return params.seq_len
        if "params" in self.config and isinstance(self.config["params"], dict):
            return self.config["params"].get("seq_len", 200)
        return 200

    def _create_trial_df(self, result: Dict, true_cps: List[int], n: int) -> pd.DataFrame:
        """Create dataframe for single trial."""
        threshold = self.config.get("detection", {}).get("threshold", 60.0)
        data = {"timestep": list(range(n)), "true_change_point": [1 if t in true_cps else 0 for t in range(n)]}

        for key in ["traditional_martingales", "traditional_sum_martingales", "traditional_avg_martingales",
                    "horizon_martingales", "horizon_sum_martingales", "horizon_avg_martingales"]:
            if key in result and len(result[key]) > 0:
                vals = result[key]
                data[key] = list(vals[:n]) + [None] * (n - len(vals))

        for key in ["individual_traditional_martingales", "individual_horizon_martingales"]:
            if key in result:
                for i, feat_vals in enumerate(result[key]):
                    if len(feat_vals) > 0:
                        data[f"{key}_f{i}"] = list(feat_vals[:n]) + [None] * (n - len(feat_vals))

        for det_key, col in [("traditional_change_points", "traditional_detected"), ("horizon_change_points", "horizon_detected")]:
            data[col] = [0] * n
            if det_key in result:
                for idx in result[det_key]:
                    if 0 <= idx < n:
                        data[col][idx] = 1

        return pd.DataFrame(data)

    def _create_summary_sheet(self, trials: List[Dict], true_cps: List[int], writer: pd.ExcelWriter) -> None:
        """Create summary sheet with detection counts per trial."""
        data = {"True CP": true_cps}
        for i, trial in enumerate(trials):
            for key, label in [("traditional_change_points", "Trad"), ("horizon_change_points", "Horizon")]:
                if key in trial:
                    dets = trial[key]
                    data[f"T{i+1} {label} Count"] = [sum(1 for d in dets if abs(d - cp) <= 30) for cp in true_cps]
                    data[f"T{i+1} {label} Latency"] = [self._latency(dets, cp) for cp in true_cps]
        pd.DataFrame(data).to_excel(writer, sheet_name="Summary", index=False)

    def _create_details_sheet(self, trials: List[Dict], true_cps: List[int], writer: pd.ExcelWriter) -> None:
        """Create detailed detection sheet."""
        rows = []
        for i, trial in enumerate(trials):
            for det_type, key in [("Traditional", "traditional_change_points"), ("Horizon", "horizon_change_points")]:
                if key in trial:
                    for j, det in enumerate(trial[key]):
                        nearest, dist = self._nearest_cp(det, true_cps)
                        rows.append({"Trial": i+1, "Type": det_type, "Det#": j+1, "Index": det,
                                    "Nearest CP": nearest, "Distance": dist, "Within30": abs(dist) <= 30})
        pd.DataFrame(rows or [{"Trial": None, "Type": None, "Det#": None, "Index": None,
                               "Nearest CP": None, "Distance": None, "Within30": None}]).to_excel(
            writer, sheet_name="Details", index=False)

    def _create_aggregate_sheet(self, trial_dfs: List[pd.DataFrame], true_cps: List[int], writer: pd.ExcelWriter) -> None:
        """Create aggregate statistics sheet."""
        n = len(trial_dfs[0])
        agg = {"timestep": list(range(n)), "true_change_point": [1 if t in true_cps else 0 for t in range(n)]}

        for col in ["traditional_detected", "horizon_detected"]:
            count = np.zeros(n)
            for df in trial_dfs:
                if col in df.columns:
                    count += df[col].values[:n]
            agg[f"{col}_count"] = count.astype(int)
            agg[f"{col}_rate"] = count / len(trial_dfs)

        mart_cols = [c for c in trial_dfs[0].columns if "_martingales" in c and not c.startswith("individual")]
        for col in mart_cols:
            if all(col in df.columns for df in trial_dfs):
                values = np.array([df[col].values[:n] for df in trial_dfs])
                masked = np.ma.masked_invalid(values)
                with np.errstate(all="ignore"):
                    mean = np.ma.mean(masked, axis=0).filled(np.nan)
                    std = np.ma.std(masked, axis=0).filled(np.nan)
                agg[f"{col}_mean"] = mean
                agg[f"{col}_std"] = std

        pd.DataFrame(agg).to_excel(writer, sheet_name="Aggregate", index=False)

        cp_data = []
        for cp in true_cps:
            trad_delays, horizon_delays = [], []
            for df in trial_dfs:
                for col, delays in [("traditional_detected", trad_delays), ("horizon_detected", horizon_delays)]:
                    if col in df.columns:
                        dets = df.loc[(df[col] == 1) & (df["timestep"] >= cp), "timestep"].values
                        if len(dets) > 0:
                            delays.append(min(dets) - cp)
            avg_t = np.mean(trad_delays) if trad_delays else np.nan
            avg_h = np.mean(horizon_delays) if horizon_delays else np.nan
            cp_data.append({"cp": cp, "trad_delay": avg_t, "horizon_delay": avg_h,
                           "reduction": (avg_t - avg_h) / avg_t if avg_t and not np.isnan(avg_t) and avg_t > 0 else np.nan})
        pd.DataFrame(cp_data).to_excel(writer, sheet_name="CPMetadata", index=False)

    def _latency(self, dets: List[int], cp: int, window: int = 30) -> float:
        """Calculate detection latency."""
        valid = [d for d in dets if d >= cp and d - cp <= window]
        return min(valid) - cp if valid else np.nan

    def _nearest_cp(self, det: int, cps: List[int]) -> Tuple[int, int]:
        """Find nearest change point."""
        if not cps:
            return None, np.inf
        nearest = min(cps, key=lambda cp: abs(cp - det))
        return nearest, det - nearest
