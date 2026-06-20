"""
equity_calculator.py
 
CAP Equity Calculator — core engine.
Works with any registered game variant by reusing the existing scoring pipeline.
 
Usage:
    from app.equity.equity_calculator import EquityCalculator
    calc = EquityCalculator()
    result = calc.calculate(
        variant_name="holdem",
        players=[{"seat": 1, "hole_cards": ["Ah", "Kd"]}, {"seat": 2, "hole_cards": ["Qh", "Jc"]}],
        board_nodes=[{"node": 0, "card": "7s"}, {"node": 1, "card": None}, ...],
        street=1,
    )
"""

import sys
import time
import random
import itertools
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Any

project_root = Path(__file__).resolve().parents[3]
engine_root = project_root / "poker_engine"
if str(engine_root) not in sys.path:
    sys.path.insert(0, str(engine_root))

    