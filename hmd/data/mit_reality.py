"""MIT Reality Mining loader — daily proximity graphs (§V, Table II).

The Bluetooth-proximity trace is the paper's real-world stress test: genuine
human-driven structure (weekday rhythms, holidays, semester transitions)
that the synthetic generators can't produce.

Critical design choices
-----------------------
- **Daily (24-h) aggregation.** Sub-minute Bluetooth readings would pick up
  coffee-break noise; day-scale preserves the event-driven shifts Table II
  annotates and matches the paper's "edges represent proximity interactions
  within 24-hour windows."
- **Edge threshold on `prob2`.** Default 0.0 (any reading = edge) for
  compatibility with unweighted-graph centralities. Paper does not specify.
- **Day 0 = 2008-09-19.** Reverse-engineered from Table II (Day 23 =
  Columbus Day 2008-10-12). The raw CSV includes pre-2008 data from the
  original Eagle/Pentland collection; we slice to the paper's 289-day
  window.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from hmd.data.synthetic import Sequence

# Default CSV path: bundled with the package under hmd/data/mit_reality/.
# Using __file__-relative so `load()` works regardless of the user's cwd.
_DEFAULT_CSV_PATH = Path(__file__).parent / "mit_reality" / "Proximity.csv"

# Paper-aligned constants (Table II)
MIT_DAY0 = datetime(2008, 9, 19)
MIT_N_DAYS = 289  # paper: "T_MIT = 289 days (September 2008 to June 2009)"

# Academic event ground truth (paper Table II).
# Day indices relative to MIT_DAY0.
MIT_EVENTS: dict[int, str] = {
    23: "Columbus Day / Fall Break",
    68: "Thanksgiving Holiday",
    97: "Christmas / New Year",  # midpoint of Dec 25 – Jan 1 range (paper shows 94-100)
    173: "Spring Break",
    234: "End of Spring Semester",
}


@dataclass(frozen=True)
class MITMetadata:
    day0: datetime
    n_days: int
    events: dict[int, str]
    threshold: float
    n_users: int


def load(
    csv_path: str | Path | None = None,
    threshold: float = 0.0,
    day0: datetime = MIT_DAY0,
    n_days: int = MIT_N_DAYS,
) -> tuple[Sequence, MITMetadata]:
    """Load MIT Reality proximity CSV → daily network snapshots.

    Parameters
    ----------
    csv_path : str | Path, optional
        Path to Proximity.csv. Defaults to the bundled copy at
        hmd/data/mit_reality/Proximity.csv (package-relative).
    threshold : float
        Minimum `prob2` value to count as an edge. 0.0 = any co-occurrence.
        Paper does not specify; 0.0 is the most inclusive choice.
    day0 : datetime
        Zero-day for indexing. Default 2008-09-19 aligns with paper Table II.
    n_days : int
        Number of daily snapshots to construct (default 289 = paper window).

    Returns
    -------
    sequence : hmd.data.synthetic.Sequence
        graphs (length n_days list of nx.Graph), change_points = list of
        MIT_EVENTS day indices as "ground truth" for TPR/FPR evaluation,
        scenario = "mit_reality", params = dict with threshold/day0/n_days.
    meta : MITMetadata
        Bookkeeping (event dict, user count after filtering).

    Why we align day0 at load time rather than inside the detector: change-point
    ground truth is calendar-anchored; the detector is timestep-anchored. Baking
    the alignment into the loader keeps the detector dataset-agnostic.
    """
    csv_path = Path(csv_path) if csv_path is not None else _DEFAULT_CSV_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"MIT Reality CSV not found at {csv_path}. "
            "Obtain from http://realitycommons.media.mit.edu/ and place under "
            "hmd/data/mit_reality/Proximity.csv."
        )

    # Parse incrementally — 2M+ rows, but we only need the (u, v, day) triples.
    df = pd.read_csv(
        csv_path,
        dtype={"user.id": "Int32", "remote.user.id.if.known": "Int32", "prob2": "float32"},
        parse_dates=["time"],
    )
    df = df.rename(columns={"user.id": "u", "remote.user.id.if.known": "v", "prob2": "p"})

    # Filter: rows with a known remote user, probability above threshold,
    # and within the paper's date window [day0, day0 + n_days).
    end_date = day0 + timedelta(days=n_days)
    mask = (
        df["v"].notna()
        & (df["p"].fillna(0.0) >= threshold)
        & (df["time"] >= pd.Timestamp(day0))
        & (df["time"] < pd.Timestamp(end_date))
    )
    df = df.loc[mask, ["u", "v", "time"]]

    # Day index.
    df["day"] = ((df["time"] - pd.Timestamp(day0)).dt.days).astype(np.int32)

    # Collect unique users for a fixed node set (≤ 94 per paper).
    users = sorted(set(df["u"].unique()) | set(df["v"].unique()))
    # Why a stable node index: feature extractor is invariant to node IDs, but
    # NetworkX graphs need a consistent vertex set across time to avoid
    # spurious "new node" effects. We fix V across all snapshots.
    node_idx = {u: i for i, u in enumerate(users)}

    # Build one graph per day. Undirected, no self-loops.
    df["u_i"] = df["u"].map(node_idx).astype(np.int32)
    df["v_i"] = df["v"].map(node_idx).astype(np.int32)
    df = df[df["u_i"] != df["v_i"]]  # drop self-loops if any
    # Canonicalize (u < v) so (u,v) == (v,u).
    min_uv = np.minimum(df["u_i"].to_numpy(), df["v_i"].to_numpy())
    max_uv = np.maximum(df["u_i"].to_numpy(), df["v_i"].to_numpy())
    df["u_i"] = min_uv
    df["v_i"] = max_uv

    # Deduplicate edges per day.
    edges_by_day = (
        df.drop_duplicates(subset=["day", "u_i", "v_i"])
        .groupby("day")[["u_i", "v_i"]]
        .apply(lambda g: list(map(tuple, g.to_numpy())))
    )

    graphs: list[nx.Graph] = []
    for d in range(n_days):
        g = nx.Graph()
        g.add_nodes_from(range(len(users)))
        if d in edges_by_day.index:
            g.add_edges_from(edges_by_day.loc[d])
        graphs.append(g)

    # True change points = paper's event days (sorted).
    change_points = sorted(MIT_EVENTS.keys())

    seq = Sequence(
        graphs=graphs,
        change_points=change_points,
        scenario="mit_reality",
        params={"threshold": threshold, "day0": str(day0.date()), "n_days": n_days},
        seed=0,
    )
    meta = MITMetadata(
        day0=day0, n_days=n_days, events=dict(MIT_EVENTS), threshold=threshold, n_users=len(users)
    )
    return seq, meta


if __name__ == "__main__":
    seq, meta = load()
    n_edges = [g.number_of_edges() for g in seq.graphs]
    print(f"Loaded MIT Reality: {meta.n_days} days, {meta.n_users} users, threshold={meta.threshold}")
    print(f"Events (day → label): {meta.events}")
    print(f"Edge-count stats: mean={np.mean(n_edges):.1f}, min={min(n_edges)}, max={max(n_edges)}")
    print(f"Zero-edge days: {sum(1 for x in n_edges if x == 0)}")
