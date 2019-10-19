from typing import Set, Dict, List, Any


def topological_ordering(adjacency_list: Dict[str, Set[str]],
                         tiebreak_ranking: Dict[str, Any] = None) -> List[str]:
    """
    Note : This assumes that the adjacency list can be safely modified
    This algorithm is not actually linear-time, but because it's not the bottle-neck,
    It's been implemented in an easy-to-follow way
    """

    if tiebreak_ranking is None:
        tiebreak_ranking = {k: 0 for k in adjacency_list.items()}

    worklist = [k for k, v in adjacency_list.items() if len(v) == 0]
    worklist = list(reversed(sorted(worklist, key=tiebreak_ranking.get)))
    result = []

    while len(worklist) > 0:
        elem = worklist.pop()
        if elem not in adjacency_list:
            continue

        result.append(elem)
        adjacency_list.pop(elem)
        for k, v in adjacency_list.items():
            v.discard(elem)

        worklist += [k for k, v in adjacency_list.items() if len(v) == 0]
        worklist = list(reversed(sorted(worklist, key=tiebreak_ranking.get)))

    return result
