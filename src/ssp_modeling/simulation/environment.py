# src/ssp_modeling/simulation# EntryState imported from types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Any
import random

# Simple, direct imports - no circular dependencies!
from .types import (
    EntryId, EntryState, Grid, SimConfig, Topology, Cell,
    ProbabilityModel, EpochOutcome, EpisodeResult
)

# Remove the circular import - we'll use TYPE_CHECKING instead
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .policy import Policy

# ---------------------------------------------------------------------
# Environment (epoch/episode execution)
# ---------------------------------------------------------------------
    


# ---------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------



@dataclass
class EntryState:
    """Represents the solver’s current knowledge about one crossword entry."""
    L: int                                      # total length
    filled_indices: set[int] = field(default_factory=set)
    guess_with_current_letters: bool = False
    solved: bool = False

    @property
    def k(self) -> int:
        return len(self.filled_indices)

    @property
    def u(self) -> int:
        return self.L - len(self.filled_indices)

# Grid imported from types.py

# SimConfig imported from types.py


# ProbabilityModel imported from types.py



# ---------------------------------------------------------------------
# Environment (epoch/episode execution)
# ---------------------------------------------------------------------

# EpochOutcome and EpisodeResult imported from types.py

class Environment:
    def __init__(self, grid: Grid, topo: Topology, cfg: SimConfig):
        self.grid = grid
        self.topo = topo
        self.cfg = cfg
        self.rng = random.Random(cfg.rng_seed)

    def all_solved(self) -> bool:
        return all(e.solved for e in self.grid.entries.values())
    
    def _print_grid_state(self):
        """Print a clear summary of the current crossword state."""
        print("📊 Current Grid State:")
        for entry_id, entry in self.grid.entries.items():
            status = "✅" if entry.solved else ("🚫" if entry.guess_with_current_letters else "⭕")
            filled_ratio = f"{len(entry.filled_indices)}/{entry.L}"
            print(f"   {status} {entry_id}: {filled_ratio} letters {'[SOLVED]' if entry.solved else ('[GUESSED]' if entry.guess_with_current_letters else '[AVAILABLE]')}")
        
        solved_count = sum(1 for e in self.grid.entries.values() if e.solved)
        total_count = len(self.grid.entries)
        print(f"📈 Progress: {solved_count}/{total_count} entries solved")

    def _apply_success(self, entry_id: EntryId) -> None:
        e = self.grid.entries[entry_id]
        e.filled_indices = set(range(e.L))
        e.solved = True
        for nbr, idxs_in_src in self.grid.crossings.get(entry_id, {}).items():
            nbr_e = self.grid.entries[nbr]
            if nbr_e.solved:
                continue
            before = len(nbr_e.filled_indices)

            # neighbor-side indices (guarded)
            idxs_in_nbr = self.grid.crossings.get(nbr, {}).get(entry_id, [])
            nbr_e.filled_indices.update(idxs_in_nbr)

            if len(nbr_e.filled_indices) > before:
                nbr_e.guess_with_current_letters = False
            if len(nbr_e.filled_indices) == nbr_e.L:
                nbr_e.solved = True



    def _apply_hint(self, target: Optional[EntryId] = None) -> EntryId:
        """
        Reveal one letter in one unsolved entry.
        If target is None, choose the entry with most unrevealed letters.
        Returns the target entry_id.
        """

        # Find unsolved entries with at least one blank
        unsolved = [
            i for i, e in self.grid.entries.items()
            if not e.solved
        ]
        if not unsolved:
            print("⚠️  No unsolved entries for hint!")
            return ""
        
        # Choose target: heuristic = most unrevealed letters (for now)
        if target is None:
            target = max(unsolved, key=lambda i: self.grid.entries[i].u)
        e = self.grid.entries[target]
        print(f"🔍 Applying hint to '{target}': {e.u} letters remaining")

        # Pick one unrevealed position
        unrevealed = [i for i in range(e.L) if i not in e.filled_indices]
        if not unrevealed:
            e.solved = True
            return target  # nothing to do

        new_index = self.rng.choice(unrevealed)
        e.filled_indices.add(new_index)
        e.guess_with_current_letters = False  # new info → can try again
        print("Total filled letters: ", e.filled_indices)
        
        # ✅ NEW: if all letters revealed, mark solved + propagate
        if len(e.filled_indices) == e.L:
            e.solved = True
            # propagate crossings to neighbors (same as _apply_success)
        return target


    def epoch(self, policy: 'Policy', model: ProbabilityModel) -> EpochOutcome:
        # Show current grid state
        print(f"\n--- EPOCH START ---")
        print("📋 Current Crossword Grid:")
        render_ascii(self.topo, self.grid, show_numbers=False, filled_marker="X", empty_marker="·")
        print()
        self._print_grid_state()
        
        # Get policy's planned order for this epoch
        full_order = policy.plan_epoch(self.grid, model)
        order = [
            i for i in full_order
            if not self.grid.entries[i].solved
            and not self.grid.entries[i].guess_with_current_letters
        ]
        
        print(f"🎯 Policy recommends trying: {full_order}")
        print(f"📋 Available to attempt: {order}")

        if not order:
            print("❌ No entries available to attempt - using hint")
            tgt = self._apply_hint()
            print(f"💡 Applied hint to entry '{tgt}'")
            print(f"📋 Updated Grid:")
            render_ascii(self.topo, self.grid, show_numbers=False, filled_marker="X", empty_marker="·")
            return EpochOutcome(cost=self.cfg.hint_cost, solved_entry=None,
                                used_hint=True, attempts_order=[], attempts_taken=0,
                                info={"hint_target": tgt})

        print(f"🎲 Attempting entries in order...")
        attempts = 0
        for entry_id in order:
            e = self.grid.entries[entry_id]
            p = max(self.cfg.eps_floor, min(1 - self.cfg.eps_floor, model.prob_solve(self.grid, entry_id)))
            
            attempts += 1
            rv = self.rng.random()
            
            print(f"  🎯 Attempt #{attempts}: Entry '{entry_id}' (L={e.L}, filled={len(e.filled_indices)}/{e.L})")
            print(f"     📊 Model probability: {p:.3f}, Random roll: {rv:.3f}")
            
            if rv < p:
                print(f"     ✅ SUCCESS! Solved entry '{entry_id}'")
                self._apply_success(entry_id)
                print(f"     🔗 Crossings updated")
                print(f"📋 Updated Grid:")
                render_ascii(self.topo, self.grid, show_numbers=False, filled_marker="X", empty_marker="·")
                return EpochOutcome(cost=attempts, solved_entry=entry_id, used_hint=False,
                                    attempts_order=order, attempts_taken=attempts, info={"p": p})
            else:
                print(f"     ❌ FAILED - marking as 'guessed with current letters'")
                e.guess_with_current_letters = True

        print(f"💔 All attempts failed - using hint")
        tgt = self._apply_hint()
        print(f"💡 Applied hint to entry '{tgt}'")
        print(f"📋 Updated Grid:")
        render_ascii(self.topo, self.grid, show_numbers=False, filled_marker="X", empty_marker="·")

        return EpochOutcome(cost=self.cfg.hint_cost, solved_entry=None,
                                used_hint=True, attempts_order=order, attempts_taken=attempts,
                                info={"hint_target": tgt})
        

    def run_episode(self, policy: 'Policy', model: ProbabilityModel) -> EpisodeResult:
        total = 0; steps = 0; hints = 0
        log: List[EpochOutcome] = []
        
        print(f"\n🚀 STARTING EPISODE")
        self._print_grid_state()
        
        while not self.all_solved() and steps < self.cfg.max_steps:
            print(f"\n{'='*50}")
            print(f"🎯 EPOCH {steps + 1}")
            print(f"{'='*50}")
            
            outcome = self.epoch(policy, model)
            total += outcome.cost
            steps += 1
            hints += int(outcome.used_hint)
            log.append(outcome)
            
            print(f"📊 EPOCH {steps} SUMMARY:")
            print(f"   Cost: {outcome.cost} | Solved: {outcome.solved_entry or 'None'} | Used hint: {outcome.used_hint}")
            
            if self.all_solved():
                print(f"\n🎉 ALL ENTRIES SOLVED! Episode complete in {steps} epochs.")
        return EpisodeResult(
            total_cost=total,
            steps=steps,
            solved_count=sum(e.solved for e in self.grid.entries.values()),
            hints_used=hints,
            log=log,
            terminated=self.all_solved(),
        )



def render_ascii(
    topo: Topology,
    grid: Grid,
    show_numbers: bool = False,
    cell_w: int = 3,
    filled_marker: str = "X",
    empty_marker: str = " ",
) -> None:
    """
    Print the crossword as ASCII.
      - Black cells: '█' (solid block)
      - White cells: filled cells show `filled_marker`; others `empty_marker`
      - If show_numbers=True, clue starts display their number in the cell.
    """
    R, C = topo.shape
    # Precompute per-cell "is filled?" based on any entry that covers it.
    filled_cells: Dict[Cell, bool] = {}
    for eid, state in grid.entries.items():
        cells = topo.entry_cells[eid]
        # For each filled index, mark that board cell as filled
        for k in state.filled_indices:
            if 0 <= k < len(cells):
                filled_cells[cells[k]] = True

    def fmt_cell(r: int, c: int) -> str:
        if topo.layout[r][c] == 1:
            return "█".ljust(cell_w)
        # White cell
        num = topo.numbers.get((r, c))
        # decide glyph
        glyph = filled_marker if filled_cells.get((r, c), False) else empty_marker
        if show_numbers and num is not None and cell_w >= 3:
            # put number left-justified in a 2-char region, glyph at the end
            return f"{num:>2}{glyph}"[:cell_w]
        else:
            return f"{glyph}".ljust(cell_w)

    # Render
    for r in range(R):
        row_str = "".join(fmt_cell(r, c) for c in range(C))
        print(row_str)