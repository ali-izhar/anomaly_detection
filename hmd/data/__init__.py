"""Graph-sequence data sources for the horizon martingale detector.

Contains:
    synthetic.py   — Table I generators: SBM, ER, BA, NWS
    mit_reality.py — loader for data/mit_reality/Proximity.csv → daily snapshots

Both return the same shape: `(graphs: list[nx.Graph], change_points: list[int])`
so experiments can swap data sources by changing one import.
"""
