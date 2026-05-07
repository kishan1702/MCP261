"""
Deep CFR Strategy Engine for PLO
---------------------------------
Implements the Deep CFR approach from PokerRL-Omaha:
  - 4 independent neural networks (preflop, flop, turn, river)
  - Trained via self-play with regret minimization
  - Preflop hand bucketing (isomorphic hand compression)
  - Supports 2–6 players
  - Persistent — learns and improves across sessions

Architecture (per network):
  Input:  39 features — hand_strength_bucket(8), board_texture(6), position+scalars(6),
          bet_size_bucket(5), spr_bucket(4), equity_edge+draws+made+nut+aggressor(7),
          street_one_hot(3)
  Hidden: 128 → 64 → 32 (Dense Residual, LeakyReLU)
  Output: [fold, check, call, bet_third, bet_half, bet_pot, raise, allin] (8)
"""

import json
import math
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from core.equity_v3 import classify_preflop, score_to_class, preflop_equity, ALL_CARDS, RANKS

# ── Action space ───────────────────────────────────────────────────────────────

ACTIONS = ["fold", "check", "call", "bet_third", "bet_half", "bet_pot", "raise", "all_in"] ###[2/3,1/4,3/4]
N_ACTIONS = len(ACTIONS)

ACTION_LABELS = {
    "fold":      "Fold",
    "check":     "Check",
    "call":      "Call",
    "bet_third": "Bet 1/3 pot",
    "bet_half":  "Bet 1/2 pot",
    "bet_pot":   "Bet pot",
    "raise":     "Raise (pot)",
    "all_in":    "All-in",
}

# ── Hand strength buckets (0–7) ────────────────────────────────────────────────

def equity_to_bucket(equity: float) -> int:
    """Map equity % to 0–7 strength bucket."""
    thresholds = [20, 30, 40, 48, 55, 63, 72]
    for i, t in enumerate(thresholds):
        if equity < t:
            return i
    return 7

######ADDED #######

from collections import Counter
from itertools import combinations

RANKS = "23456789TJQKA"


def rank(card):
    return RANKS.index(card[0]) + 2


def ranks(board):
    vals = [rank(c) for c in board]

    # Wheel ace support
    if 14 in vals:
        vals.append(1)

    return sorted(set(vals))


def suit_counts(board):
    return Counter(c[1] for c in board)


def rank_counts(board):
    return Counter(rank(c) for c in board)


# ---------------------------------------------------
# FLUSH TEXTURE
# ---------------------------------------------------

def flush_texture(board):
    counts = suit_counts(board)
    mx = max(counts.values())

    if mx == 3:
        return 1.0

    if mx == 2:
        return 0.55

    return 0.0


# ---------------------------------------------------
# STRAIGHT TEXTURE
# ---------------------------------------------------

def straight_texture(board):
    r = ranks(board)

    score = 0

    # Examine all straight windows
    for start in range(1, 11):

        window = set(range(start, start + 5))

        overlap = len(window.intersection(r))

        if overlap == 3:
            score += 1.0

        elif overlap == 2:
            score += 0.35

    # Normalize
    return min(score / 3.0, 1.0)


# ---------------------------------------------------
# DENSITY
# ---------------------------------------------------

def density_texture(board):
    vals = sorted(set(rank(c) for c in board))

    if len(vals) < 2:
        return 0

    gaps = [vals[i+1] - vals[i] for i in range(len(vals)-1)]

    avg_gap = sum(gaps) / len(gaps)

    # Smaller gaps => wetter
    return max(0, 1 - ((avg_gap - 1) / 6))


# ---------------------------------------------------
# HIGH CARD PRESSURE
# ---------------------------------------------------

def highcard_texture(board):
    vals = sorted([rank(c) for c in board], reverse=True)

    pressure = sum(max(v - 9, 0) for v in vals)

    return min(pressure / 15, 1.0)


# ---------------------------------------------------
# PAIRING
# ---------------------------------------------------

def pairing_texture(board):
    counts = rank_counts(board)

    mx = max(counts.values())

    if mx == 3:
        return -1.2

    if mx == 2:
        return -0.5

    return 0.0


# ---------------------------------------------------
# NUT POTENTIAL
# ---------------------------------------------------

def nut_texture(board):
    vals = sorted(set(rank(c) for c in board))

    score = 0

    # Broadway-heavy boards
    broadway = sum(v >= 10 for v in vals)

    score += broadway * 0.25

    # Connected broadway
    if max(vals) - min(vals) <= 4:
        score += 0.5

    return min(score, 1.0)


# ---------------------------------------------------
# TURN DYNAMICITY
# ---------------------------------------------------

def dynamicity(board):
    """
    Measures how many turn cards
    significantly alter board texture.
    """

    current_ranks = set(rank(c) for c in board)

    volatility = 0

    for turn in range(2, 15):

        # Ignore impossible duplicate turns
        if turn in current_ranks:
            continue

        simulated = list(current_ranks) + [turn]

        simulated = sorted(simulated)

        # Straight expansion
        for start in range(1, 11):

            window = set(range(start, start + 5))

            overlap = len(window.intersection(simulated))

            if overlap >= 4:
                volatility += 1
                break

    return min(volatility / 10, 1.0)


# ---------------------------------------------------
# MAIN TEXTURE FUNCTION
# ---------------------------------------------------

def board_texture(board):
    """
    Returns:
        texture score from 0 to 100

    Interpretation:
        0-20   = extremely dry
        20-40  = dry
        40-60  = semi-wet
        60-80  = wet
        80-100 = extremely wet
    """

    if len(board) < 3:
        return 0

    flop = board[:3]

    score = 0

    # Flush coordination
    score += 20 * flush_texture(flop)

    # Straight interaction
    score += 25 * straight_texture(flop)

    # Rank density
    score += 15 * density_texture(flop)

    # High-card interaction
    score += 10 * highcard_texture(flop)

    # Nut potential
    score += 15 * nut_texture(flop)

    # Turn volatility
    score += 20 * dynamicity(flop)

    # Pairing adjustment
    score += 10 * pairing_texture(flop)

    return round(max(0, min(score, 100)), 1)


#######END#########


# ── Board texture encoding (0–5) ──────────────────────────────────────────────

def board_texture(board: list[str]) -> int:
    if not board:
        return 0
    suits = [c[1] for c in board[:3]]
    ranks = sorted([RANKS.index(c[0]) for c in board[:3]], reverse=True)
    from collections import Counter
    sc = Counter(suits)
    mono     = max(sc.values()) == 3
    two_tone = max(sc.values()) == 2
    paired   = len(set(c[0] for c in board[:3])) < 3
    connected = (ranks[0] - ranks[2]) <= 4

    if mono and connected:   return 5   # Very wet
    if mono:                 return 4   # Monotone
    if connected and two_tone: return 3 # Wet connected
    if connected:            return 2   # Straight-heavy
    if two_tone:             return 1   # Two-tone
    return 0                            # Dry rainbow

# ── Feature helpers ────────────────────────────────────────────────────────────

def _bet_size_bucket(to_call: float, pot: float) -> int:
    """0=no bet, 1=small(≤0.35), 2=medium(0.35-0.65), 3=large(0.65-1.0), 4=overbet"""
    if to_call <= 0 or pot <= 0:
        return 0
    ratio = to_call / pot
    if ratio <= 0.35: return 1
    if ratio <= 0.65: return 2
    if ratio <= 1.00: return 3
    return 4

def _spr_bucket(stack: float, pot: float) -> int:
    """0=critical(<2), 1=low(2-5), 2=medium(5-15), 3=deep(15+)"""
    spr = stack / (pot + 1e-6)
    if spr < 2:  return 0
    if spr < 5:  return 1
    if spr < 15: return 2
    return 3

def _flush_draw(hole_cards: list[str], board: list[str]) -> float:
    """1.0 if hero has a flush draw (4 to a flush) but not a made flush."""
    if not hole_cards or not board:
        return 0.0
    all_cards = hole_cards + board
    from collections import Counter
    suit_counts = Counter(c[1] for c in all_cards)
    max_flush = max(suit_counts.values())
    if max_flush >= 5:
        return 0.0   # made flush, not a draw
    if max_flush >= 4:
        # Confirm at least 2 hole cards contribute
        for suit, cnt in suit_counts.items():
            if cnt >= 4:
                hole_in_suit = sum(1 for c in hole_cards if c[1] == suit)
                if hole_in_suit >= 2:
                    return 1.0
    return 0.0

def _straight_draw(hole_cards: list[str], board: list[str]) -> float:
    """1.0 if hero has an open-ended or wrap straight draw."""
    if not hole_cards or not board:
        return 0.0
    all_ranks = sorted(set(RANKS.index(c[0]) for c in hole_cards + board))
    # Check any 4-card window of width ≤ 4 (open-ended or wrap)
    for i in range(len(all_ranks) - 3):
        if all_ranks[i + 3] - all_ranks[i] <= 4:
            return 1.0
    return 0.0

def _made_two_pair_plus(hole_cards: list[str], board: list[str]) -> float:
    """Rough check: 1.0 if hero likely has two-pair or better (rank overlap heuristic)."""
    if not hole_cards or len(board) < 3:
        return 0.0
    board_ranks = [c[0] for c in board]
    hole_ranks  = [c[0] for c in hole_cards]
    paired_to_board = sum(1 for r in hole_ranks if r in board_ranks)
    return 1.0 if paired_to_board >= 2 else 0.0

def _nut_advantage(hole_cards: list[str], board: list[str]) -> float:
    """1.0 if holding nut flush draw (Ace of dominant flush suit) or top straight draw."""
    if not hole_cards or not board:
        return 0.0
    from collections import Counter
    suit_counts = Counter(c[1] for c in board)
    if suit_counts:
        dominant_suit = max(suit_counts, key=suit_counts.get)
        if suit_counts[dominant_suit] >= 2:
            if any(c == 'A' + dominant_suit for c in hole_cards):
                return 1.0
    # Top of a straight draw (holds the T+ end)
    all_ranks = sorted(set(RANKS.index(c[0]) for c in hole_cards + board))
    if len(all_ranks) >= 4:
        top_rank  = all_ranks[-1]
        hole_rank_set = {RANKS.index(c[0]) for c in hole_cards}
        if top_rank in hole_rank_set and top_rank >= RANKS.index('T'):
            return 1.0
    return 0.0


# ── Feature vector ─────────────────────────────────────────────────────────────

def build_features(
    equity:      float,
    board:       list[str],
    position:    str,      # "IP" or "OOP"
    pot:         float,
    to_call:     float,
    stack:       float,
    n_players:   int,
    n_actions_taken: int,
    hole_cards:  list[str] = None,
    is_aggressor: bool = False,
    street:      str = "",
) -> np.ndarray:
    """Build a 39-dimensional feature vector for the neural network."""
    strength_bucket = equity_to_bucket(equity)
    texture         = board_texture(board)
    is_ip           = 1.0 if position == "IP" else 0.0
    pot_odds        = to_call / (pot + to_call + 1e-6)
    spr             = min(stack / (pot + 1e-6), 20.0) / 20.0   # normalised 0-1
    n_pl_norm       = (n_players - 2) / 4.0                     # normalised 0-1
    facing_bet      = 1.0 if to_call > 0 else 0.0
    actions_norm    = min(n_actions_taken, 10) / 10.0

    # One-hot: strength bucket (8 dims)
    bucket_oh = [0.0] * 8
    bucket_oh[strength_bucket] = 1.0

    # One-hot: board texture (6 dims)
    texture_oh = [0.0] * 6
    texture_oh[texture] = 1.0

    # One-hot: bet size bucket (5 dims)
    bet_bucket = _bet_size_bucket(to_call, pot)
    bet_oh = [0.0] * 5
    bet_oh[bet_bucket] = 1.0

    # One-hot: SPR bucket (4 dims)
    spr_cat = _spr_bucket(stack, pot)
    spr_oh = [0.0] * 4
    spr_oh[spr_cat] = 1.0

    # Scalar extras
    equity_edge   = equity / 100.0 - (1.0 / max(n_players, 2))
    board_paired  = 1.0 if board and len(set(c[0] for c in board)) < len(board) else 0.0
    flush_draw    = _flush_draw(hole_cards, board) if hole_cards else 0.0
    str_draw      = _straight_draw(hole_cards, board) if hole_cards else 0.0
    made_twoplus  = _made_two_pair_plus(hole_cards, board) if hole_cards else 0.0
    nut_adv       = _nut_advantage(hole_cards, board) if hole_cards else 0.0
    aggressor_f   = 1.0 if is_aggressor else 0.0

    # Street one-hot (3 dims: flop / turn / river; preflop = all zero)
    street_oh = [
        1.0 if street == "flop"  else 0.0,
        1.0 if street == "turn"  else 0.0,
        1.0 if street == "river" else 0.0,
    ]

    features = (
        bucket_oh +          # 8
        texture_oh +         # 6
        [is_ip,              # 1
         pot_odds,           # 1
         spr,                # 1
         n_pl_norm,          # 1
         facing_bet,         # 1
         actions_norm] +     # 1   → subtotal 20
        bet_oh +             # 5
        spr_oh +             # 4
        [equity_edge,        # 1
         board_paired,       # 1
         flush_draw,         # 1
         str_draw,           # 1
         made_twoplus,       # 1
         nut_adv,            # 1
         aggressor_f] +      # 1   → subtotal 36
        street_oh            # 3   → total 39
    )
    # Total: 8+6+6 + 5+4 + 7 + 3 = 39
    return np.array(features, dtype=np.float32)

N_FEATURES = 39


# ── Dense Residual Network (PokerRL-Omaha architecture) ───────────────────────

class DenseResidualNet:
    """
    Lightweight dense residual network — pure NumPy, no PyTorch dependency.
    Architecture: Input(39) → 128 → 64 → 32 → Output(8)
    With skip connections and LeakyReLU (slope=0.1 as per PokerRL-Omaha).
    """
    def __init__(self, n_in: int = N_FEATURES, n_out: int = N_ACTIONS):
        self.n_in  = n_in
        self.n_out = n_out
        self._init_weights()

    def _init_weights(self):
        # He initialisation for LeakyReLU
        def he(fan_in, fan_out):
            std = math.sqrt(2.0 / fan_in)
            return np.random.randn(fan_in, fan_out).astype(np.float32) * std

        self.W1 = he(N_FEATURES, 128)
        self.b1 = np.zeros(128, dtype=np.float32)
        self.W2 = he(128, 64)
        self.b2 = np.zeros(64, dtype=np.float32)
        self.W3 = he(64, 32)
        self.b3 = np.zeros(32, dtype=np.float32)
        self.W4 = he(32, N_ACTIONS)
        self.b4 = np.zeros(N_ACTIONS, dtype=np.float32)
        # Skip connection: 128 → 32
        self.Wskip = he(128, 32)

    def _lrelu(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, 0.1 * x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h1 = self._lrelu(x @ self.W1 + self.b1)
        h2 = self._lrelu(h1 @ self.W2 + self.b2)
        h3 = self._lrelu(h2 @ self.W3 + self.b3)
        # Residual skip from h1
        skip = h1 @ self.Wskip
        h3 = h3 + self._lrelu(skip)
        out = h3 @ self.W4 + self.b4
        return out

    def regret_match(self, x: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
        """Convert network output to action probabilities via regret matching."""
        logits = self.forward(x)
        logits = logits * legal_mask
        positive = np.maximum(logits, 0)
        total = positive.sum()
        if total > 1e-6:
            return positive / total
        # Uniform over legal actions
        legal = legal_mask.astype(np.float32)
        return legal / legal.sum()

    def to_dict(self) -> dict:
        return {
            "W1": self.W1.tolist(), "b1": self.b1.tolist(),
            "W2": self.W2.tolist(), "b2": self.b2.tolist(),
            "W3": self.W3.tolist(), "b3": self.b3.tolist(),
            "W4": self.W4.tolist(), "b4": self.b4.tolist(),
            "Wskip": self.Wskip.tolist(),
        }

    def from_dict(self, d: dict):
        self.W1    = np.array(d["W1"],    dtype=np.float32)
        self.b1    = np.array(d["b1"],    dtype=np.float32)
        self.W2    = np.array(d["W2"],    dtype=np.float32)
        self.b2    = np.array(d["b2"],    dtype=np.float32)
        self.W3    = np.array(d["W3"],    dtype=np.float32)
        self.b3    = np.array(d["b3"],    dtype=np.float32)
        self.W4    = np.array(d["W4"],    dtype=np.float32)
        self.b4    = np.array(d["b4"],    dtype=np.float32)
        self.Wskip = np.array(d["Wskip"], dtype=np.float32)
        return self

    _WEIGHT_CLIP = 10.0   # max absolute weight value — prevents explosion

    def is_healthy(self) -> bool:
        """True if all weights are finite and within sane range."""
        return all(
            np.isfinite(arr).all() and np.abs(arr).max() < 1e6
            for arr in (self.W1, self.b1, self.W2, self.b2, self.W3,
                        self.b3, self.W4, self.b4, self.Wskip)
        )

    def _clip_weights(self):
        c = self._WEIGHT_CLIP
        for attr in ('W1','b1','W2','b2','W3','b3','W4','b4','Wskip'):
            setattr(self, attr, np.clip(getattr(self, attr), -c, c))

    def _clip_grad(self, g: np.ndarray, clip: float = 1.0) -> np.ndarray:
        norm = np.linalg.norm(g)
        return g * (clip / norm) if norm > clip else g

    def update(self, features: np.ndarray, target_regrets: np.ndarray, lr: float = 0.01):
        """Full backprop through all layers via SGD with gradient clipping."""
        target_regrets = np.clip(
            np.nan_to_num(target_regrets, nan=0.0, posinf=100.0, neginf=-100.0),
            -100.0, 100.0
        )
        # Forward pass — cache activations
        h1     = self._lrelu(features @ self.W1 + self.b1)
        h2     = self._lrelu(h1 @ self.W2 + self.b2)
        h3_pre = h2 @ self.W3 + self.b3
        h3     = self._lrelu(h3_pre) + self._lrelu(h1 @ self.Wskip)
        pred   = h3 @ self.W4 + self.b4
        error  = np.clip(
            np.nan_to_num(pred - target_regrets, nan=0.0),
            -10.0, 10.0
        )

        # Layer 4
        self.W4 -= lr * self._clip_grad(np.outer(h3, error))
        self.b4 -= lr * self._clip_grad(error)

        # Layer 3
        d3 = (error @ self.W4.T) * (h3_pre > 0).astype(np.float32)
        d3 = np.clip(d3, -10.0, 10.0)
        self.W3 -= lr * self._clip_grad(np.outer(h2, d3))
        self.b3 -= lr * self._clip_grad(d3)

        # Skip connection
        d_skip = (error @ self.W4.T) * (h1 @ self.Wskip > 0).astype(np.float32)
        d_skip = np.clip(d_skip, -10.0, 10.0)
        self.Wskip -= lr * self._clip_grad(np.outer(h1, d_skip))

        # Layer 2
        d2 = (d3 @ self.W3.T) * (h1 @ self.W2 + self.b2 > 0).astype(np.float32)
        d2 = np.clip(d2, -10.0, 10.0)
        self.W2 -= lr * self._clip_grad(np.outer(h1, d2))
        self.b2 -= lr * self._clip_grad(d2)

        # Layer 1
        d1 = (d2 @ self.W2.T + d_skip @ self.Wskip.T) * (features @ self.W1 + self.b1 > 0).astype(np.float32)
        d1 = np.clip(d1, -10.0, 10.0)
        self.W1 -= lr * self._clip_grad(np.outer(features, d1))
        self.b1 -= lr * self._clip_grad(d1)

        self._clip_weights()
        if not self.is_healthy():
            self._init_weights()


# ── Legal action mask ──────────────────────────────────────────────────────────

def legal_mask(facing_bet: bool, stack: float, pot: float, to_call: float = 0.0) -> np.ndarray:
    """Return a binary mask of legal actions enforcing PLO pot-limit rules."""
    mask = np.ones(N_ACTIONS, dtype=np.float32)
    if facing_bet:
        mask[ACTIONS.index("check")]     = 0
        mask[ACTIONS.index("bet_third")] = 0
        mask[ACTIONS.index("bet_half")]  = 0
        mask[ACTIONS.index("bet_pot")]   = 0
        pot_limit = pot + 2 * to_call
        if stack > pot_limit:
            mask[ACTIONS.index("all_in")] = 0
        if stack < pot_limit:
            mask[ACTIONS.index("raise")] = 0
    else:
        mask[ACTIONS.index("fold")]  = 0
        mask[ACTIONS.index("call")]  = 0
        mask[ACTIONS.index("raise")] = 0
        if stack > pot:
            mask[ACTIONS.index("all_in")] = 0
        if stack < pot / 3:
            mask[ACTIONS.index("bet_third")] = 0
        if stack < pot / 2:
            mask[ACTIONS.index("bet_half")]  = 0
        if stack < pot:
            mask[ACTIONS.index("bet_pot")]   = 0
    return mask


# ── Deep CFR Engine ────────────────────────────────────────────────────────────

@dataclass
class StrategyOutput:
    action:      str
    amount:      float
    probs:       dict[str, float]   # all action probabilities
    ev:          float
    reasoning:   list[str]

class DeepCFREngine:
    """
    4 neural networks — one per street.
    Trained via online CFR+ during self-play.
    Loads/saves weights to disk — learns across sessions.
    """

    STREETS = ["preflop", "flop", "turn", "river"]

    def __init__(self, model_path: str = "data/deepcfr_weights.json"):
        self.model_path = Path(model_path)
        self.nets: dict[str, DenseResidualNet] = {
            s: DenseResidualNet() for s in self.STREETS
        }
        self.regret_sums: dict[str, dict[str, np.ndarray]] = {
            s: {} for s in self.STREETS
        }
        self._update_count = 0
        self._load()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_strategy(
        self,
        street:    str,
        equity:    float,
        board:     list[str],
        position:  str,
        pot:       float,
        to_call:   float,
        stack:     float,
        n_players: int,
        n_actions: int = 0,
        hole_cards: list[str] = None,
        is_aggressor: bool = False,
    ) -> StrategyOutput:
        """Get action probabilities and recommended action for this spot."""
        features    = build_features(equity, board, position, pot, to_call, stack, n_players, n_actions,
                                     hole_cards=hole_cards, is_aggressor=is_aggressor, street=street)
        facing_bet  = to_call > 0
        mask        = legal_mask(facing_bet, stack, pot, to_call)
        net         = self.nets.get(street, self.nets["flop"])
        probs_arr   = net.regret_match(features, mask)

        probs = {a: float(probs_arr[i]) for i, a in enumerate(ACTIONS)}

        # Dominant action
        best_idx    = int(np.argmax(probs_arr))
        best_action = ACTIONS[best_idx]

        # Compute amount
        amount = self._action_amount(best_action, pot, to_call, stack)

        # EV estimate
        if best_action == "fold":
            ev = 0.0
        elif best_action == "check":
            ev = equity / 100 * pot
        elif amount > 0:
            ev = equity / 100 * (pot + amount) - amount
        else:
            ev = 0.0

        reasoning = self._reasoning(equity, pot, to_call, best_action, probs, n_players)

        return StrategyOutput(
            action=ACTION_LABELS.get(best_action, best_action),
            amount=amount,
            probs={ACTION_LABELS.get(k, k): round(v * 100, 1) for k, v in probs.items() if v > 0.01},
            ev=round(ev, 1),
            reasoning=reasoning,
        )

    def update_from_outcome(
        self,
        street:    str,
        equity:    float,
        board:     list[str],
        position:  str,
        pot:       float,
        to_call:   float,
        stack:     float,
        n_players: int,
        action_taken: str,
        outcome:   float,   # chips won/lost this hand
        hole_cards: list[str] = None,
        is_aggressor: bool = False,
        auto_save:  bool = True,
    ):
        """
        Online learning — update network weights after each hand outcome.
        Computes counterfactual regrets per action and does a gradient step.
        auto_save=False suppresses periodic saves (caller saves manually at end).
        """
        features   = build_features(equity, board, position, pot, to_call, stack, n_players, 0,
                                    hole_cards=hole_cards, is_aggressor=is_aggressor, street=street)
        facing_bet = to_call > 0
        mask       = legal_mask(facing_bet, stack, pot, to_call)
        net        = self.nets.get(street, self.nets["flop"])

        probs = net.regret_match(features, mask)
        s     = equity / 100.0
        target_regrets = np.zeros(N_ACTIONS, dtype=np.float32)

        for i, action in enumerate(ACTIONS):
            if mask[i] == 0:
                continue
            amt = self._action_amount(action, pot, to_call, stack)
            if action == "fold":
                cfr_ev = 0.0
            elif action == "check":
                cfr_ev = s * pot
            elif action == "call":
                cfr_ev = s * (pot + amt) - amt
            else:   # bet / raise / all_in — opponent also contributes amt if they call
                cfr_ev = s * (pot + 2.0 * amt) - amt
            target_regrets[i] = cfr_ev - outcome

        net.update(features, target_regrets, lr=0.003)
        self._update_count += 1
        if auto_save and self._update_count % 100 == 0:
            self._save()

    def train_self_play(self, n_iterations: int = 500, verbose: bool = True):
        """
        Self-play training loop.
        Generates random PLO situations and updates all 4 networks via CFR+.
        """
        if verbose:
            print(f"Training Deep CFR for {n_iterations} iterations...")
        for i in range(n_iterations):
            for street in self.STREETS:
                self._cfr_iteration(street)
            if verbose and (i + 1) % 1000 == 0:
                print(f"  {i+1}/{n_iterations} iterations complete")
        self._save()
        if verbose:
            print("Training complete.")

    # ── Internal ───────────────────────────────────────────────────────────────

    def _cfr_iteration(self, street: str):
        """Single CFR iteration on a realistic PLO game state."""
        # U-shaped beta distribution: real PLO hands are mostly strong or weak,
        # rarely exactly 50%. uniform(15,85) was biased — fold/call tied at 50%
        # so random noise dictated which action accumulated regret.
        raw    = random.betavariate(0.65, 0.65)   # U-shaped: peaks near 0 and 1
        equity = 12.0 + raw * 76.0                # maps to ~12–88% range

        position     = random.choice(["IP", "OOP"])
        pot          = random.uniform(75, 4000)
        # Mix of facing-bet (60%) and no-bet (40%) situations
        if random.random() < 0.60:
            to_call = pot * random.uniform(0.25, 1.0)
        else:
            to_call = 0
        stack        = random.uniform(pot * 0.5, pot * 8)
        n_players    = random.randint(2, 4)
        is_aggressor = random.random() > 0.5

        deck = ALL_CARDS.copy()
        random.shuffle(deck)
        hole_cards = deck[:4]
        if street == "preflop":   board = []
        elif street == "flop":    board = deck[4:7]
        elif street == "turn":    board = deck[4:8]
        else:                     board = deck[4:9]

        features   = build_features(equity, board, position, pot, to_call, stack, n_players, 0,
                                    hole_cards=hole_cards, is_aggressor=is_aggressor, street=street)
        facing_bet = to_call > 0
        mask       = legal_mask(facing_bet, stack, pot, to_call)
        net        = self.nets[street]
        probs      = net.regret_match(features, mask)

        utils     = self._action_utilities(equity, pot, to_call, stack, mask, street)
        node_util = float(probs @ utils)

        regrets  = utils - node_util
        info_key = f"{street}_{int(equity/10)}_{int(facing_bet)}_{int(pot/500)}"
        if info_key not in self.regret_sums[street]:
            self.regret_sums[street][info_key] = np.zeros(N_ACTIONS, dtype=np.float32)
        self.regret_sums[street][info_key] += regrets

        pos_regrets = np.clip(np.maximum(self.regret_sums[street][info_key], 0), 0, 1e4)
        net.update(features, pos_regrets, lr=0.002)

    def _action_utilities(
        self,
        equity:  float,
        pot:     float,
        to_call: float,
        stack:   float,
        mask:    np.ndarray,
        street:  str = "",
    ) -> np.ndarray:
        """
        Compute approximate utility for each action.

        Key fix: bet EV must account for the CALLER's contribution to the pot.
        When we bet `amt` and opponent calls `amt`, the final pot is (pot + 2*amt),
        not (pot + amt). The old formula `equity*(pot+amt)-amt` underestimated bet
        EV by ~50%, causing the network to always prefer check over bet.

        Fold equity: bluffing has realistic positive EV when opponent sometimes folds.
        """
        s = equity / 100
        # Fold frequency per street for a pot-sized bet
        # Value hands get called more; bluffs get folded more
        base_fold = {"preflop": 0.30, "flop": 0.38, "turn": 0.28, "river": 0.18}.get(street, 0.28)

        utils = np.zeros(N_ACTIONS, dtype=np.float32)
        for i, action in enumerate(ACTIONS):
            if mask[i] == 0:
                utils[i] = -1e6
                continue
            amt = self._action_amount(action, pot, to_call, stack)
            if action == "fold":
                utils[i] = 0.0
            elif action == "check":
                utils[i] = s * pot
            elif action == "call":
                # pot already includes the aggressor's bet; we add our call
                utils[i] = s * (pot + to_call) - to_call
            else:
                # Bet / raise
                # When called: final pot = pot + our_bet + opponent_call = pot + 2*amt
                # fold_prob: bluffs get more folds (opponent doesn't have enough equity to call)
                bet_ratio = min(amt / (pot + 1e-6), 1.5)
                fold_prob = base_fold * bet_ratio * (1.0 - s)   # bluff → more folds
                fold_prob = min(fold_prob, 0.55)                 # PLO: hard to fold >55%
                ev_if_fold = pot                                  # win pot uncontested
                ev_if_call = s * (pot + 2.0 * amt) - amt         # opponent also adds amt
                utils[i] = fold_prob * ev_if_fold + (1.0 - fold_prob) * ev_if_call
        return utils

    def _action_amount(self, action: str, pot: float, to_call: float, stack: float) -> float:
        if action == "fold":    return 0
        if action == "check":   return 0
        if action == "call":    return min(to_call, stack)
        if action == "bet_third": return min(pot / 3, stack)
        if action == "bet_half":  return min(pot / 2, stack)
        if action == "bet_pot":   return min(pot, stack)
        if action == "raise":     return min(pot + 2 * to_call, stack)
        if action == "all_in":    return stack
        return 0

    def _reasoning(self, equity, pot, to_call, action, probs, n_players) -> list[str]:
        lines = []
        lines.append(f"Equity: {equity:.1f}% vs {n_players-1} opponent(s)")
        if to_call > 0:
            pot_odds_needed = 100 * to_call / (pot + to_call)
            lines.append(f"Pot odds: need {pot_odds_needed:.1f}% to call profitably")
        top_actions = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        mix = "  |  ".join(f"{a} {p:.0f}%" for a, p in top_actions if p > 1)
        lines.append(f"GTO mix: {mix}")
        return lines

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        data = {street: net.to_dict() for street, net in self.nets.items()}
        with open(self.model_path, "w") as f:
            json.dump(data, f, allow_nan=False)

    def _load(self):
        if not self.model_path.exists():
            return
        try:
            with open(self.model_path) as f:
                data = json.load(f)
            for street, weights in data.items():
                if street in self.nets:
                    net = DenseResidualNet().from_dict(weights)
                    if not net.is_healthy():
                        raise ValueError(f"unhealthy weights for {street}")
                    self.nets[street] = net
        except Exception:
            pass  # Start fresh if corrupt
