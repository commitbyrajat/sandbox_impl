"""Logical queries used by deterministic model fixtures on the new tool interface."""


def count_query(table, name="n"):
    return {
        "base_table": table,
        "metrics": [{"name": name, "metric": f"{table}.__count"}],
    }


def holdings_query(name="value", customer=False, date=False):
    table, key = ("sw_holdings", "customer_id") if customer else ("sw_amcs", "amc_name")
    query = {
        "base_table": "sw_holdings",
        "dimensions": [{"name": key, "field": {"table": table, "column": key}}],
        "metrics": [{"name": name, "metric": "holding_value"}],
        "order_by": [{"name": key}],
    }
    if date:
        query["metrics"].append({"name": "nav_date", "metric": "valuation_date"})
    return query
