from typing import Any, Dict


class OpTracker:
    """
    This tracker is used to record the choices taken within any DSL operator.
    This is fundamental to tracking the choices made within a generator
    """
    def __init__(self):
        self.record: Dict[str, Dict[Any, Any]] = {}

    def copy(self):
        res = OpTracker()
        res.record = self.record.copy()
        return res
