"""
NCA Fundraising Revenue Performance & Planning Assistant
GBA 479 — Development & Fundraising Department
Northbridge Community Alliance

Architecture:
  - Tool layer (Python/deterministic): 4 financial calculation functions from provided notebook
  - Router (LLM/Claude Haiku): classifies user intent → selects tool(s)
  - Generator (LLM/Claude Opus): interprets tool output → natural language response
  - Validator (Python/deterministic): checks response cites specific numbers
"""

import os
import json
import re
import pandas as pd
import numpy as np
from typing import Optional
from anthropic import Anthropic

# ─────────────────────────────────────────────
# DATA LOAD
# ─────────────────────────────────────────────

def load_data(path: str = "northbridge_fundraising_history.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m")
    return df


# ─────────────────────────────────────────────
# DETERMINISTIC TOOL FUNCTIONS (from notebook)
# ─────────────────────────────────────────────

def _as_month(d: str) -> pd.Timestamp:
    return pd.to_datetime(d, format="%Y-%m")

def _month_range_end(as_of: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=as_of.year, month=as_of.month, day=1)

def _last_n_months(df: pd.DataFrame, as_of_month: pd.Timestamp, n: int) -> pd.DataFrame:
    start = as_of_month - pd.DateOffset(months=n - 1)
    mask = (df["date"] >= start) & (df["date"] <= as_of_month)
    return df.loc[mask].copy()

def _ytd(df: pd.DataFrame, as_of_month: pd.Timestamp) -> pd.DataFrame:
    start = pd.Timestamp(year=as_of_month.year, month=1, day=1)
    mask = (df["date"] >= start) & (df["date"] <= as_of_month)
    return df.loc[mask].copy()

def _safe_div(a: float, b: float) -> Optional[float]:
    b = float(b)
    return None if b == 0 else float(a) / b


def get_fundraising_snapshot(data: pd.DataFrame, as_of: str) -> dict:
    """MTD / YTD / TTM snapshot + revenue breakdown."""
    as_of_month = _month_range_end(_as_month(as_of))
    row = data.loc[data["date"] == as_of_month]
    if row.empty:
        return {"ok": False, "error": f"No data for {as_of}."}
    row = row.iloc[0]
    m_rev, m_cost = float(row["revenue_total"]), float(row["costs_total"])
    ytd_df = _ytd(data, as_of_month)
    ttm_df = _last_n_months(data, as_of_month, 12)
    breakdown_cols = [c for c in data.columns if c.startswith("revenue_") and c != "revenue_total"]
    breakdown = {}
    if breakdown_cols:
        month_break = {c: float(row[c]) for c in breakdown_cols}
        shares = {c: _safe_div(v, m_rev) for c, v in month_break.items()}
        breakdown = {"month": month_break, "month_share_of_total": shares}
    ytd_rev = float(ytd_df["revenue_total"].sum())
    ttm_rev = float(ttm_df["revenue_total"].sum())
    return {
        "ok": True, "as_of": as_of,
        "month": {"revenue": round(m_rev, 2), "costs": round(m_cost, 2), "net": round(m_rev - m_cost, 2)},
        "ytd":   {"revenue": round(ytd_rev, 2), "costs": round(float(ytd_df["costs_total"].sum()), 2),
                  "net": round(ytd_rev - float(ytd_df["costs_total"].sum()), 2)},
        "ttm":   {"revenue": round(ttm_rev, 2), "costs": round(float(ttm_df["costs_total"].sum()), 2),
                  "net": round(ttm_rev - float(ttm_df["costs_total"].sum()), 2)},
        "breakdown": breakdown,
    }


def compare_yoy(data: pd.DataFrame, as_of: str, metric: str = "revenue_total") -> dict:
    """Year-over-year comparison for any metric."""
    as_of_month = _month_range_end(_as_month(as_of))
    prior = as_of_month - pd.DateOffset(years=1)
    if metric not in data.columns:
        return {"ok": False, "error": f"Unknown metric '{metric}'."}
    cur_row = data.loc[data["date"] == as_of_month]
    prv_row = data.loc[data["date"] == prior]
    if cur_row.empty or prv_row.empty:
        return {"ok": False, "error": f"Missing data for {as_of} or {prior.strftime('%Y-%m')}."}
    cur_val, prv_val = float(cur_row.iloc[0][metric]), float(prv_row.iloc[0][metric])
    cur_ytd = float(_ytd(data, as_of_month)[metric].sum())
    prv_ytd = float(_ytd(data, prior)[metric].sum())
    return {
        "ok": True, "as_of": as_of, "metric": metric,
        "month": {"current": round(cur_val, 2), "prior_year": round(prv_val, 2),
                  "yoy_pct": None if prv_val == 0 else round((cur_val - prv_val) / prv_val, 4)},
        "ytd":   {"current": round(cur_ytd, 2), "prior_year": round(prv_ytd, 2),
                  "yoy_pct": None if prv_ytd == 0 else round((cur_ytd - prv_ytd) / prv_ytd, 4)},
    }


def cost_effectiveness(data: pd.DataFrame, as_of: str, window_months: int = 12) -> dict:
    """Cost-to-raise and net margin over trailing window."""
    as_of_month = _month_range_end(_as_month(as_of))
    w = _last_n_months(data, as_of_month, window_months)
    rev = float(w["revenue_total"].sum())
    cost = float(w["costs_total"].sum())
    net = rev - cost
    return {
        "ok": True, "as_of": as_of, "window_months": window_months,
        "revenue": round(rev, 2), "costs": round(cost, 2), "net": round(net, 2),
        "cost_to_raise_1": round(_safe_div(cost, rev), 4) if rev else None,
        "net_margin": round(_safe_div(net, rev), 4) if rev else None,
    }


def goal_pacing(data: pd.DataFrame, as_of: str, annual_goal: float) -> dict:
    """Pacing toward annual revenue goal."""
    as_of_month = _month_range_end(_as_month(as_of))
    ytd_df = _ytd(data, as_of_month)
    raised_ytd = float(ytd_df["revenue_total"].sum())
    remaining = max(0.0, float(annual_goal) - raised_ytd)
    month_num = as_of_month.month
    remaining_months = 12 - month_num
    req = None if remaining_months == 0 else remaining / remaining_months
    avg = None if month_num == 0 else raised_ytd / month_num
    status = "on_track"
    if req and avg:
        status = "behind" if avg < req else "ahead"
    return {
        "ok": True, "as_of": as_of, "annual_goal": round(float(annual_goal), 2),
        "raised_ytd": round(raised_ytd, 2), "remaining_to_goal": round(remaining, 2),
        "remaining_months": int(remaining_months),
        "required_avg_per_remaining_month": round(req, 2) if req else None,
        "current_ytd_monthly_avg": round(avg, 2) if avg else None,
        "pace_status": status,
    }


# ─────────────────────────────────────────────
# WHAT-IF SCENARIO (bonus deterministic tool)
# ─────────────────────────────────────────────

def what_if_scenario(data: pd.DataFrame, as_of: str,
                     revenue_adjustment_pct: float = 0.0,
                     cost_adjustment_pct: float = 0.0,
                     window_months: int = 12) -> dict:
    """
    Projects adjusted net and margin if revenue/costs shift by given %.
    revenue_adjustment_pct: e.g. 0.10 = +10%
    cost_adjustment_pct:    e.g. -0.05 = -5%
    """
    base = cost_effectiveness(data, as_of, window_months)
    if not base["ok"]:
        return base
    adj_rev  = base["revenue"] * (1 + revenue_adjustment_pct)
    adj_cost = base["costs"]   * (1 + cost_adjustment_pct)
    adj_net  = adj_rev - adj_cost
    return {
        "ok": True, "as_of": as_of, "window_months": window_months,
        "scenario": {"revenue_adj_pct": revenue_adjustment_pct, "cost_adj_pct": cost_adjustment_pct},
        "baseline": {"revenue": base["revenue"], "costs": base["costs"], "net": base["net"],
                     "net_margin": base["net_margin"]},
        "adjusted": {
            "revenue": round(adj_rev, 2), "costs": round(adj_cost, 2), "net": round(adj_net, 2),
            "net_margin": round(_safe_div(adj_net, adj_rev), 4) if adj_rev else None,
        },
    }


# ─────────────────────────────────────────────
# AVAILABLE METRICS HELPER
# ─────────────────────────────────────────────

METRIC_MAP = {
    "revenue": "revenue_total",
    "total revenue": "revenue_total",
    "individual": "revenue_individual",
    "individual giving": "revenue_individual",
    "foundation": "revenue_foundation",
    "foundation grants": "revenue_foundation",
    "corporate": "revenue_corporate",
    "corporate sponsorship": "revenue_corporate",
    "events": "revenue_events",
    "online": "revenue_online",
    "online campaigns": "revenue_online",
    "costs": "costs_total",
    "total costs": "costs_total",
}

LATEST_MONTH = "2025-12"
DATA_RANGE   = "January 2021 – December 2025"


# ─────────────────────────────────────────────
# ROUTER PROMPT
# ─────────────────────────────────────────────

ROUTER_SYSTEM = f"""You are a routing classifier for the NCA Fundraising Assistant.
The assistant has these tools (Python functions):
  1. snapshot        — MTD/YTD/TTM revenue, costs, net, revenue breakdown
  2. yoy             — year-over-year comparison for a metric
  3. cost_kpi        — cost-to-raise-$1 and net margin over trailing window
  4. pacing          — goal pacing (requires annual_goal from user)
  5. what_if         — scenario modelling (revenue/cost % adjustments)
  6. multi           — combination of multiple tools needed

Data available: {DATA_RANGE}. Latest month: {LATEST_MONTH}.

Return ONLY a JSON object (no markdown) with:
{{
  "route": "<snapshot|yoy|cost_kpi|pacing|what_if|multi|out_of_scope>",
  "as_of": "<YYYY-MM or null>",
  "metric": "<column name or null>",
  "annual_goal": <number or null>,
  "revenue_adj_pct": <decimal or null>,
  "cost_adj_pct": <decimal or null>,
  "window_months": <number or null>,
  "reasoning": "<one sentence>"
}}

Rules:
- Default as_of to {LATEST_MONTH} if not specified.
- If metric mentioned, map to: revenue_total, costs_total, revenue_individual,
  revenue_foundation, revenue_corporate, revenue_events, revenue_online.
- route=out_of_scope if question is not about NCA fundraising financial data.
- route=pacing ONLY if user provides or implies an annual_goal number.
- If user asks for pacing but gives no goal, route=snapshot and note missing goal in reasoning.
"""

GENERATOR_SYSTEM = """You are the NCA Fundraising Performance Assistant for Northbridge Community Alliance.
You interpret deterministic financial tool outputs and deliver clear, professional analysis
to the Development & Fundraising team.

Rules:
- ALWAYS cite specific numbers from the tool output (dollar amounts, percentages, dates).
- Format dollar figures with $ and commas (e.g. $142,500).
- Format percentages to 1 decimal place (e.g. +8.3%).
- Use plain business language — no jargon, no hedging phrases like "it seems".
- Structure: 1-2 sentence headline → supporting detail → one actionable observation.
- Keep responses under 200 words.
- If tool returned ok=False, explain clearly what data is missing and what the user can try.
- Never fabricate numbers not in the tool output.
- End every response with: 📊 Data source: Northbridge fundraising history ({range}).
""".replace("{range}", DATA_RANGE)

OUT_OF_SCOPE_MSG = """This assistant is scoped to NCA fundraising financial data (revenue, costs, donor metrics, goal pacing) covering {range}.

I can help with:
• Monthly / YTD / TTM revenue and cost snapshots
• Year-over-year comparisons across any revenue stream
• Cost-to-raise efficiency and net margin analysis
• Goal pacing and required run-rate calculations
• Revenue / cost scenario modelling

Please rephrase your question in terms of fundraising performance, or ask about a specific time period or metric.
""".replace("{range}", DATA_RANGE)


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

def validate_response(response: str, tool_result: dict) -> tuple[bool, str]:
    """
    Deterministic checks:
    1. Response contains at least one dollar amount or percentage.
    2. Response contains the data source line.
    3. Response does not contain numbers not present in tool_result JSON.
    Returns (passed, reason).
    """
    has_number = bool(re.search(r'\$[\d,]+|\d+\.\d+%|[\+\-]\d+\.\d+%', response))
    has_source = "Data source" in response or "data source" in response
    if not has_number:
        return False, "Response contains no cited figures."
    if not has_source:
        return False, "Response missing data source citation."
    return True, "ok"


# ─────────────────────────────────────────────
# MAIN ASSISTANT LOOP
# ─────────────────────────────────────────────

def run_tool(route: dict, data: pd.DataFrame) -> dict:
    r = route.get("route")
    as_of = route.get("as_of") or LATEST_MONTH
    metric = route.get("metric") or "revenue_total"
    goal = route.get("annual_goal")
    rev_adj = route.get("revenue_adj_pct") or 0.0
    cost_adj = route.get("cost_adj_pct") or 0.0
    window = route.get("window_months") or 12

    if r == "snapshot":
        return get_fundraising_snapshot(data, as_of)
    elif r == "yoy":
        return compare_yoy(data, as_of, metric)
    elif r == "cost_kpi":
        return cost_effectiveness(data, as_of, window)
    elif r == "pacing":
        if not goal:
            return {"ok": False, "error": "Annual goal not provided. Please specify a goal amount (e.g. '$1.2M annual goal')."}
        return goal_pacing(data, as_of, goal)
    elif r == "what_if":
        return what_if_scenario(data, as_of, rev_adj, cost_adj, window)
    elif r == "multi":
        results = {}
        results["snapshot"] = get_fundraising_snapshot(data, as_of)
        if goal:
            results["pacing"] = goal_pacing(data, as_of, goal)
        results["cost_kpi"] = cost_effectiveness(data, as_of, window)
        return results
    else:
        return {"ok": False, "error": "out_of_scope"}


def chat(data: pd.DataFrame):
    client = Anthropic()
    conversation_history = []

    print("\n" + "═" * 60)
    print("  NCA FUNDRAISING PERFORMANCE ASSISTANT")
    print("  Development & Fundraising Department")
    print(f"  Data: {DATA_RANGE}")
    print("═" * 60)
    print("  Ask about revenue, costs, YoY trends, goal pacing,")
    print("  cost effectiveness, or scenario modelling.")
    print("  Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nAssistant: Goodbye. Good luck with your fundraising goals! 🎯")
            break

        conversation_history.append({"role": "user", "content": user_input})

        # ── STEP 1: ROUTE (Haiku — fast, cheap) ──
        try:
            route_resp = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=300,
                system=ROUTER_SYSTEM,
                messages=conversation_history,
            )
            route_raw = route_resp.content[0].text.strip()
            route = json.loads(route_raw)
        except Exception as e:
            print(f"\nAssistant: [Router error: {e}. Please try again.]\n")
            conversation_history.pop()
            continue

        # ── STEP 2: OUT OF SCOPE ──
        if route.get("route") == "out_of_scope":
            response = OUT_OF_SCOPE_MSG
            print(f"\nAssistant: {response}\n")
            conversation_history.append({"role": "assistant", "content": response})
            continue

        # ── STEP 3: RUN TOOL (Python — deterministic) ──
        tool_result = run_tool(route, data)

        # ── STEP 4: GENERATE (Opus — quality response) ──
        gen_messages = conversation_history + [{
            "role": "user",
            "content": (
                f"Tool results (deterministic calculations):\n"
                f"```json\n{json.dumps(tool_result, indent=2)}\n```\n\n"
                f"User question: {user_input}\n\n"
                f"Provide a clear business interpretation. Cite specific figures."
            )
        }]

        try:
            gen_resp = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=400,
                system=GENERATOR_SYSTEM,
                messages=gen_messages,
            )
            response = gen_resp.content[0].text.strip()
        except Exception as e:
            print(f"\nAssistant: [Generation error: {e}]\n")
            conversation_history.pop()
            continue

        # ── STEP 5: VALIDATE (Python — deterministic) ──
        passed, reason = validate_response(response, tool_result)
        if not passed:
            # Regenerate once with explicit instruction
            try:
                gen_messages[-1]["content"] += f"\n\nIMPORTANT: Previous response failed validation ({reason}). You MUST include specific dollar amounts and the data source line."
                retry_resp = client.messages.create(
                    model="claude-opus-4-5",
                    max_tokens=400,
                    system=GENERATOR_SYSTEM,
                    messages=gen_messages,
                )
                response = retry_resp.content[0].text.strip()
            except Exception:
                pass  # use original response

        print(f"\nAssistant: {response}\n")
        conversation_history.append({"role": "assistant", "content": response})


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    csv_path = "northbridge_fundraising_history.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    try:
        data = load_data(csv_path)
    except FileNotFoundError:
        # Try uploads directory
        try:
            data = load_data("/mnt/user-data/uploads/northbridge_fundraising_history.csv")
        except FileNotFoundError:
            print(f"Error: Could not find fundraising data CSV.")
            sys.exit(1)

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    chat(data)
