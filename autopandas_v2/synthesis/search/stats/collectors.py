import collections
from typing import Dict


class StatsCollector:
    def __init__(self):
        self.num_cands_generated: Dict[int, int] = collections.defaultdict(int)
        self.num_cands_error: Dict[int, int] = collections.defaultdict(int)
        self.num_cands_propagated: Dict[int, int] = collections.defaultdict(int)
        self.num_seqs_explored = 0
