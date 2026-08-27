from typing import Dict, List, Tuple


WRIST_INDEX = 0
FINGER_CHAINS: List[List[int]] = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
]


def build_auxiliary_targets() -> Dict[int, List[int]]:
    # Wrist predicts the whole hand; each finger joint only predicts its distal descendants.
    targets: Dict[int, List[int]] = {WRIST_INDEX: list(range(1, 21))}
    for chain in FINGER_CHAINS:
        for idx, joint_id in enumerate(chain):
            targets[joint_id] = chain[idx + 1 :]
    return targets


AUXILIARY_TARGETS = build_auxiliary_targets()


def build_topology_distances() -> Dict[Tuple[int, int], int]:
    distances: Dict[Tuple[int, int], int] = {}

    for chain in FINGER_CHAINS:
        for depth, joint_id in enumerate(chain, start=1):
            distances[(WRIST_INDEX, joint_id)] = depth

        # Within one finger, the distance is the number of hops along the kinematic chain.
        for src_offset, src_joint in enumerate(chain):
            for dst_offset, dst_joint in enumerate(chain[src_offset + 1 :], start=1):
                distances[(src_joint, dst_joint)] = dst_offset
    return distances


TOPOLOGY_DISTANCES = build_topology_distances()
MAX_TOPOLOGY_DISTANCE = max(TOPOLOGY_DISTANCES.values())


def enumerate_auxiliary_pairs() -> List[Tuple[int, int, int]]:
    pairs: List[Tuple[int, int, int]] = []
    for src in sorted(AUXILIARY_TARGETS):
        for dst in AUXILIARY_TARGETS[src]:
            pairs.append((src, dst, TOPOLOGY_DISTANCES[(src, dst)]))
    return pairs


AUXILIARY_PAIRS = enumerate_auxiliary_pairs()
