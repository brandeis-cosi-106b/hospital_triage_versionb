"""
Internal patient storage layer.

Provides the backing store used by PatientRegistry to maintain patients and
support fast admission, discharge, range counting, and ward partitioning.

This module is private to the er_triage package.  Consumers should interact
only with PatientRegistry and Ward, which provide the stable public interface.
"""

from __future__ import annotations

import random
from typing import Iterator, List, Optional, Tuple

from .patient import Patient

# ---------------------------------------------------------------------------
# Internal tree node
# ---------------------------------------------------------------------------


class _Node:
    """A single node in the acuity-sorted index tree.

    Each node stores one patient, a randomly-assigned structural key used to
    maintain balance, and left/right child pointers.
    """

    __slots__ = ("patient", "_balance_key", "left", "right")

    def __init__(self, patient: Patient) -> None:
        self.patient: Patient = patient
        # The balance key is assigned once at insertion and never changes.
        # It drives the heap-like structural invariant that keeps the tree
        # balanced in expectation without explicit rebalancing passes.
        self._balance_key: float = random.random()
        self.left: Optional[_Node] = None
        self.right: Optional[_Node] = None

    # Convenience aliases so callers can read node.acuity / node.patient_id
    # directly without going through node.patient each time.

    @property
    def acuity(self) -> int:
        return self.patient.acuity

    @property
    def patient_id(self) -> str:
        return self.patient.patient_id

    def __repr__(self) -> str:
        return f"_Node(acuity={self.acuity}, id={self.patient_id!r})"


# ---------------------------------------------------------------------------
# Patient store
# ---------------------------------------------------------------------------


class _AcuityIndex:
    """In-memory patient store ordered by acuity.

    Patients are kept in acuity order to support efficient range queries and
    threshold partitioning.  The structural balance key stored in each node
    keeps the expected height at O(log n) without explicit rebalancing.

    Supported operations
    --------------------
    insert(patient)               O(log n) expected
    remove(patient_id)            O(n) search + O(log n) removal
    contains(patient_id)          O(n)
    find(patient_id)              O(n)
    maximum() / minimum()         O(log n)
    count_in_range(lo, hi)        O(log n)
    split_above(threshold)        O(log n)  — destructive partition
    merge_from(other)             O(log n)  — absorb another index
    inorder()                     O(n)
    """

    def __init__(self) -> None:
        self._root: Optional[_Node] = None
        self._count: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def insert(self, patient: Patient) -> None:
        """Add *patient* to the index."""
        self._root = self._insert(self._root, patient)
        self._count += 1

    def remove(self, patient_id: str) -> bool:
        """Remove the patient with the given ID.  Returns True if found."""
        new_root, removed = self._remove(self._root, patient_id)
        if removed:
            self._root = new_root
            self._count -= 1
        return removed

    def contains(self, patient_id: str) -> bool:
        """Return True if a patient with *patient_id* is in the index."""
        return self._find_node(self._root, patient_id) is not None

    def find(self, patient_id: str) -> Optional[Patient]:
        """Return the Patient with the given ID, or None if absent."""
        node = self._find_node(self._root, patient_id)
        return node.patient if node else None

    def maximum(self) -> Optional[Patient]:
        """Return the patient with the highest acuity score, or None."""
        node = self._rightmost(self._root)
        return node.patient if node else None

    def minimum(self) -> Optional[Patient]:
        """Return the patient with the lowest acuity score, or None."""
        node = self._leftmost(self._root)
        return node.patient if node else None

    def count_in_range(self, lo: int, hi: int) -> int:
        """Count patients whose acuity is in the closed interval [lo, hi].

        The index is temporarily restructured during the computation and
        fully restored before returning.  Behaviour is undefined if lo > hi.
        """
        if lo > hi:
            return 0
        # Carve out the [lo, hi] band with two partitions then count its size.
        left, mid_and_right = self._partition(self._root, lo)
        mid, right = self._partition(mid_and_right, hi + 1)
        result = self._subtree_size(mid)
        # Restore original structure.
        self._root = self._merge(self._merge(left, mid), right)
        return result

    def split_above(self, acuity_threshold: int) -> "_AcuityIndex":
        """Destructively partition: remove and return patients with acuity >= threshold.

        After this call the current index retains only patients whose acuity
        is strictly below *acuity_threshold*.  The returned index holds all
        patients at or above the threshold.
        """
        below, above = self._partition(self._root, acuity_threshold)
        self._root = below
        self._count = self._subtree_size(below)
        extracted = _AcuityIndex()
        extracted._root = above
        extracted._count = self._subtree_size(above)
        return extracted

    def merge_from(self, other: "_AcuityIndex") -> None:
        """Absorb all patients from *other* into this index.

        Precondition: every patient in *other* must have strictly higher acuity
        than every patient currently in this index, or vice-versa.  Callers
        are responsible for ensuring this invariant.
        """
        self._root = self._merge(self._root, other._root)
        self._count += other._count
        other._root = None
        other._count = 0

    def inorder(self) -> List[Patient]:
        """Return all patients sorted by acuity (ascending)."""
        result: List[Patient] = []
        self._collect_inorder(self._root, result)
        return result

    def __len__(self) -> int:
        return self._count

    def __bool__(self) -> bool:
        return self._count > 0

    def __iter__(self) -> Iterator[Patient]:
        return iter(self.inorder())

    # ------------------------------------------------------------------
    # Core structural primitives
    # ------------------------------------------------------------------

    def _partition(
        self, node: Optional[_Node], threshold: int
    ) -> Tuple[Optional[_Node], Optional[_Node]]:
        """Split the subtree rooted at *node* into two disjoint subtrees.

        Returns (lower, upper) where:
            lower — patients with acuity <  threshold
            upper — patients with acuity >= threshold
        """
        if node is None:
            return None, None
        if node.acuity < threshold:
            lower, upper = self._partition(node.right, threshold)
            node.right = lower
            return node, upper
        else:
            lower, upper = self._partition(node.left, threshold)
            node.left = upper
            return lower, node

    def _merge(
        self, left: Optional[_Node], right: Optional[_Node]
    ) -> Optional[_Node]:
        """Merge two subtrees where every key in *left* is less than every key
        in *right*.  The balance keys determine the merged tree's shape."""
        if left is None:
            return right
        if right is None:
            return left
        if left._balance_key < right._balance_key:
            left.right = self._merge(left.right, right)
            return left
        else:
            right.left = self._merge(left, right.left)
            return right

    def _insert(self, node: Optional[_Node], patient: Patient) -> _Node:
        """Insert *patient* and return the new subtree root."""
        new_node = _Node(patient)
        lower, upper = self._partition(node, patient.acuity)
        return self._merge(self._merge(lower, new_node), upper)

    def _remove(
        self, node: Optional[_Node], patient_id: str
    ) -> Tuple[Optional[_Node], bool]:
        """Search for *patient_id* and remove it.  Returns (new_root, found)."""
        if node is None:
            return None, False
        if node.patient_id == patient_id:
            # Splice out by merging the two children directly.
            return self._merge(node.left, node.right), True
        # The tree is ordered by acuity, not patient_id, so we must check
        # both subtrees.  Try left first; fall through to right if not found.
        new_left, found = self._remove(node.left, patient_id)
        if found:
            node.left = new_left
            return node, True
        new_right, found = self._remove(node.right, patient_id)
        node.right = new_right
        return node, found

    # ------------------------------------------------------------------
    # Traversal helpers
    # ------------------------------------------------------------------

    def _find_node(
        self, node: Optional[_Node], patient_id: str
    ) -> Optional[_Node]:
        """Full-tree search for a node by patient_id (O(n))."""
        if node is None:
            return None
        if node.patient_id == patient_id:
            return node
        found = self._find_node(node.left, patient_id)
        if found is not None:
            return found
        return self._find_node(node.right, patient_id)

    @staticmethod
    def _leftmost(node: Optional[_Node]) -> Optional[_Node]:
        if node is None:
            return None
        while node.left is not None:
            node = node.left
        return node

    @staticmethod
    def _rightmost(node: Optional[_Node]) -> Optional[_Node]:
        if node is None:
            return None
        while node.right is not None:
            node = node.right
        return node

    @staticmethod
    def _subtree_size(node: Optional[_Node]) -> int:
        if node is None:
            return 0
        return 1 + _AcuityIndex._subtree_size(node.left) + _AcuityIndex._subtree_size(node.right)

    @staticmethod
    def _collect_inorder(node: Optional[_Node], out: List[Patient]) -> None:
        if node is None:
            return
        _AcuityIndex._collect_inorder(node.left, out)
        out.append(node.patient)
        _AcuityIndex._collect_inorder(node.right, out)
