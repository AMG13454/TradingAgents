"""Run the graph over a grid of tickers and dates, and score what came back.

One run yields one decision, so it cannot say whether the system decides well.
This runs the same machinery over many (ticker, date) cells and reads the
aggregate. The memory log is the results table: every run already records its
rating and later settles it with realized and alpha return against the
instrument's regional benchmark, so there is nothing to record separately.

Scope: this evaluates decision quality. It is not a portfolio simulator, and
must not grow one. Turning a rating into a filled order needs a quantity, a fill
price and a cash ledger, none of which the system has; inventing them here would
put an execution model behind an evaluation tool. Cells are therefore
independent, and a portfolio, when given, is the same standing book for every
cell rather than a position carried forward.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from tradingagents.agents.rating import RATING_REVIEW
from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.symbols import safe_ticker_component
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.memory import TradingMemoryLog

logger = logging.getLogger(__name__)


def iter_grid(start_date: str, end_date: str, every_n_days: int = 1) -> list[str]:
    """Analysis dates from ``start_date``, never past today.

    A future date has no outcome to settle against, and the graph rejects one, so
    the grid stops at the present rather than producing cells that cannot score.
    """
    start, end = _canonical(start_date), _canonical(end_date)
    if every_n_days < 1:
        raise ValueError("every_n_days must be at least 1")
    if end < start:
        raise ValueError(f"the grid ends before it starts: {end_date} is before {start_date}")

    last = min(end, datetime.strptime(get_current_date(), "%Y-%m-%d"))
    dates, cursor = [], start
    while cursor <= last:
        dates.append(cursor.strftime("%Y-%m-%d"))
        cursor += timedelta(days=every_n_days)
    return dates


def _canonical(date: str) -> datetime:
    """Parse a grid bound, rejecting anything the run date would also reject."""
    try:
        parsed = datetime.strptime(str(date), "%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"grid dates must be in YYYY-MM-DD format, got {date!r}") from exc
    if parsed.strftime("%Y-%m-%d") != str(date):
        raise ValueError(f"grid dates must be in YYYY-MM-DD format, got {date!r}")
    return parsed


def _alpha(entry: dict) -> float | None:
    """Alpha return of a settled entry, or None when it has not settled.

    The log stores it as a percentage rounded to one decimal, so aggregates here
    are accurate to 0.1 of a percentage point, not to the raw quote.
    """
    text = (entry.get("alpha") or "").strip().rstrip("%")
    try:
        return float(text) / 100
    except ValueError:
        return None


@dataclass
class BacktestResult:
    run_id: str
    log_path: Path
    cells_run: int = 0
    skipped: int = 0
    failures: list[tuple[str, str, str]] = field(default_factory=list)
    settlement_failures: list[tuple[str, str]] = field(default_factory=list)


# What each rating claims will happen, so an outcome can be scored against it.
# Hold claims no direction, so nothing about alpha proves it right or wrong.
_DIRECTION = {"Buy": 1, "Overweight": 1, "Hold": 0, "Underweight": -1, "Sell": -1}


@dataclass
class RatingScore:
    count: int
    hit_rate: float | None
    mean_alpha: float


# Two-sided 95% Student-t critical values by degrees of freedom; 1.96 past 30.
_T_95 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
         2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
         2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)


@dataclass
class Edge:
    """Every directional call pooled: alpha earned in the direction each one called.

    A hit rate and a mean say nothing about whether they differ from chance, and
    a sweep of a dozen cells does not. The intervals here say how much the sample
    supports. Cells whose holding windows overlap for one ticker share return
    days, so they are not separate observations: the intervals are computed on
    ``independent``, the cells left when overlapping ones are counted once.
    """

    count: int
    independent: int
    hit_rate: float
    hit_interval: tuple[float, float]
    mean_alpha: float
    alpha_interval: tuple[float, float] | None    # None when there are too few to estimate a spread

    @property
    def verdict(self) -> str:
        if self.alpha_interval is None:
            return "too few independent calls to estimate"
        low, high = self.alpha_interval
        if low > 0:
            return "above zero at the 95% level"
        if high < 0:
            return "below zero at the 95% level: the calls were worse than no call"
        return "not distinguishable from zero"

    def render(self) -> str:
        line = (
            f"Directional calls: n={self.count} ({self.independent} with windows that do not overlap), "
            f"called the direction {self.hit_rate:.0%} "
            f"(95% interval {self.hit_interval[0]:.0%} to {self.hit_interval[1]:.0%}), "
            f"mean alpha in the called direction {self.mean_alpha:+.2%}"
        )
        if self.alpha_interval is not None:
            line += f" (95% interval {self.alpha_interval[0]:+.2%} to {self.alpha_interval[1]:+.2%})"
        return f"{line}: {self.verdict}."


def _independent(cells: list[tuple[str, str, int]]) -> int:
    """How many (ticker, date, holding days) cells have windows that do not overlap.

    Per ticker, a cell counts when it starts after the last counted cell's window
    has ended, taking a trading day as 7/5 of a calendar day.
    """
    count = 0
    free_from: dict[str, datetime] = {}
    for ticker, date, holding in sorted(cells):
        start = datetime.strptime(date, "%Y-%m-%d")
        if ticker not in free_from or start >= free_from[ticker]:
            count += 1
            free_from[ticker] = start + timedelta(days=math.ceil((holding + 1) * 7 / 5))
    return count


def _wilson(hits: float, n: int) -> tuple[float, float]:
    """95% Wilson interval for a proportion: honest at small n, where p +/- 2se is not."""
    z = 1.96
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _edge(resolved: list[tuple[dict, float]]) -> Edge | None:
    """The pooled directional edge of settled decisions, or None when none claimed a direction."""
    signed, cells = [], []
    for entry, alpha in resolved:
        direction = _DIRECTION.get(entry["rating"], 0)
        if not direction:
            continue
        signed.append(alpha * direction)
        holding = entry.get("holding") or ""
        days = int(holding[:-1]) if holding.endswith("d") and holding[:-1].isdigit() else 5
        cells.append((entry["ticker"], entry["date"], days))
    if not signed:
        return None
    n, independent = len(signed), _independent(cells)
    mean = sum(signed) / n
    hit_rate = sum(a > 0 for a in signed) / n
    interval = None
    if independent >= 2 and n >= 2:
        spread = math.sqrt(sum((a - mean) ** 2 for a in signed) / (n - 1))
        t = _T_95[independent - 2] if independent - 1 <= len(_T_95) else 1.96
        half = t * spread / math.sqrt(independent)
        interval = (mean - half, mean + half)
    return Edge(count=n, independent=independent, hit_rate=hit_rate,
                hit_interval=_wilson(hit_rate * independent, independent),
                mean_alpha=mean, alpha_interval=interval)


@dataclass
class BacktestSummary:
    resolved: int
    pending: int
    by_rating: dict[str, RatingScore]
    unscored: int = 0
    holding: str = ""
    edge: Edge | None = None

    def render(self) -> str:
        lines = [f"Resolved cells: {self.resolved} · pending: {self.pending}"
                 + (f" · unscored: {self.unscored}" if self.unscored else "")]
        for rating, score in self.by_rating.items():
            called = (f"called the direction {score.hit_rate:.0%}"
                      if score.hit_rate is not None else "no direction claimed")
            lines.append(
                f"- {rating}: n={score.count}, {called}, "
                f"mean alpha {score.mean_alpha:+.2%} vs the benchmark"
            )
        lines.append("")
        if self.edge is not None:
            lines.extend([self.edge.render(), ""])
        if self.pending:
            lines.append("Pending cells are not scored above; re-run to settle them.")
        lines.append(
            f"Alpha is measured over {self.holding} from the first close after each analysis date. "
            "One model sampling per cell, and text feeds are not archived, so "
            "these figures are indicative rather than repeatable."
        )
        return "\n".join(lines)


def _new_run(runs: Path) -> str:
    """A new run's id, stamped now, and its directory, claimed so no other sweep shares it."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs.mkdir(parents=True, exist_ok=True)
    for n in itertools.count(1):
        run_id = stamp if n == 1 else f"{stamp}_{n}"
        try:
            (runs / run_id).mkdir()
            return run_id
        except FileExistsError:
            continue


def run_backtest(
    tickers: list[str],
    dates: list[str],
    config: dict,
    asset_type: str = "stock",
    portfolio=None,
    selected_analysts=("market", "social", "news", "fundamentals"),
    run_id: str | None = None,
    progress: Callable[[int, int, str, str], None] | None = None,
) -> BacktestResult:
    """Analyze every ticker on every date, into a memory log of this run's own.

    The live log stays untouched: a sweep would otherwise flood the context that
    real runs read back. Cells already in this run's log are skipped, so an
    interrupted sweep resumes by being run again. ``progress(done, total,
    ticker, date)`` is called before each cell that runs.
    """
    runs = Path(config["results_dir"]) / "backtest"
    if run_id:
        # run_id becomes a path segment, so it is validated like a ticker: an
        # absolute or dotted value would otherwise place the run outside results_dir.
        run_id = safe_ticker_component(run_id)
        (runs / run_id).mkdir(parents=True, exist_ok=True)
    else:
        run_id = _new_run(runs)
    run_dir = runs / run_id
    run_config = {**config, "results_dir": str(run_dir),
                  "memory_log_path": str(run_dir / "trading_memory.md")}

    graph = TradingAgentsGraph(selected_analysts, config=run_config)
    result = BacktestResult(run_id=run_id, log_path=Path(run_config["memory_log_path"]))
    done = {(e["ticker"], e["date"]) for e in graph.memory_log.load_entries()}

    # A ticker or date given twice is one cell, run and settled once.
    tickers, dates = list(dict.fromkeys(tickers)), list(dict.fromkeys(dates))
    cells = [(ticker, date) for ticker in tickers for date in dates]
    todo = [cell for cell in cells if cell not in done]
    result.skipped = len(cells) - len(todo)
    for index, (ticker, date) in enumerate(todo, 1):
        if progress:
            progress(index, len(todo), ticker, date)
        try:
            graph.propagate(ticker, date, asset_type, portfolio=portfolio)
            result.cells_run += 1
        except Exception as exc:  # one unreachable vendor must not end the sweep
            logger.warning("Backtest cell %s %s failed: %s", ticker, date, exc)
            result.failures.append((ticker, date, str(exc)))

    # Settlement runs at the start of the next run for a ticker, so each ticker's
    # last cell would stay pending without this pass.
    for ticker in tickers:
        try:
            settlement = graph.settle_pending(ticker)
            result.settlement_failures.extend(
                (failed_ticker, f"{date}: {reason}")
                for failed_ticker, date, reason in settlement.failed
            )
        except Exception as exc:  # reflection calls an LLM; one failure is not the sweep's
            logger.warning("Settling %s failed: %s", ticker, exc)
            result.settlement_failures.append((ticker, str(exc)))
    return result


def summarize(source: BacktestResult | str | Path) -> BacktestSummary:
    """Score the settled decisions of a backtest, or of a memory log at a path, by rating."""
    if isinstance(source, BacktestResult):
        path = source.log_path      # a run whose cells all failed wrote no log: nothing to score
    elif Path(source).is_file():
        path = Path(source)
    else:
        raise FileNotFoundError(f"no memory log at {source}")
    entries = TradingMemoryLog({"memory_log_path": str(path)}).load_entries()
    # A decision with no readable rating has no direction, so it can neither
    # count for nor against the system; it is reported as unscored instead.
    resolved = [(e, _alpha(e)) for e in entries
                if not e["pending"] and e["rating"] != RATING_REVIEW]
    resolved = [(e, a) for e, a in resolved if a is not None]
    by_rating: dict[str, RatingScore] = {}
    for rating in dict.fromkeys(e["rating"] for e, _ in resolved):
        alphas = [a for e, a in resolved if e["rating"] == rating]
        direction = _DIRECTION.get(rating, 0)
        by_rating[rating] = RatingScore(
            count=len(alphas),
            hit_rate=(sum(a * direction > 0 for a in alphas) / len(alphas)) if direction else None,
            mean_alpha=sum(alphas) / len(alphas),
        )
    unscored = sum(1 for e in entries if e["rating"] == RATING_REVIEW)
    # Report the window the outcomes were actually measured over, from the log.
    windows = {f"{e['holding'][:-1]} trading days" for e, _ in resolved
               if (e.get("holding") or "").endswith("d")}
    return BacktestSummary(resolved=len(resolved),
                           pending=len(entries) - len(resolved) - unscored,
                           by_rating=by_rating, unscored=unscored,
                           holding=", ".join(sorted(windows)) or "the configured window",
                           edge=_edge(resolved))
