"""
Generates an HTML validation report from ValidationResult objects.
Output is a self-contained HTML file (no external dependencies).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.validation.schema_validator import ValidationResult

_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Data Validation Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
    h1 {{ color: #1a1a2e; }}
    .summary {{ display: flex; gap: 2rem; margin: 1.5rem 0; }}
    .stat {{ background: #f4f4f4; border-radius: 8px; padding: 1rem 2rem; text-align: center; }}
    .stat .num {{ font-size: 2rem; font-weight: bold; }}
    .critical {{ color: #c0392b; }} .warning {{ color: #e67e22; }} .info {{ color: #27ae60; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th {{ background: #1a1a2e; color: white; padding: 0.6rem 1rem; text-align: left; }}
    td {{ padding: 0.5rem 1rem; border-bottom: 1px solid #ddd; }}
    tr.critical-row td {{ background: #fdecea; }}
    tr.warning-row td {{ background: #fef9e7; }}
    .badge {{ border-radius: 4px; padding: 2px 8px; font-size: 0.8rem; font-weight: bold; color: white; }}
    .badge-critical {{ background: #c0392b; }}
    .badge-warning {{ background: #e67e22; }}
    .badge-info {{ background: #27ae60; }}
  </style>
</head>
<body>
  <h1>Operational Risk Dashboard — Data Validation Report</h1>
  <p>Generated: {timestamp} | Pipeline run</p>

  <div class="summary">
    <div class="stat"><div class="num">{total}</div><div>Total Checks</div></div>
    <div class="stat"><div class="num critical">{n_critical}</div><div>Critical</div></div>
    <div class="stat"><div class="num warning">{n_warning}</div><div>Warnings</div></div>
    <div class="stat"><div class="num info">{n_passed}</div><div>Passed</div></div>
  </div>

  <table>
    <thead>
      <tr>
        <th>Level</th><th>Table</th><th>Check</th><th>Detail</th><th>Value</th><th>Threshold</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>
"""

_ROW = """\
<tr class="{row_class}">
  <td><span class="badge badge-{level_lower}">{level}</span></td>
  <td>{table}</td>
  <td><code>{check}</code></td>
  <td>{detail}</td>
  <td>{value}</td>
  <td>{threshold}</td>
</tr>"""


def generate(results: list[ValidationResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_critical = sum(1 for r in results if r.level == "CRITICAL")
    n_warning = sum(1 for r in results if r.level == "WARNING")
    n_info = sum(1 for r in results if r.level == "INFO")

    rows_html = "\n".join(
        _ROW.format(
            row_class=f"{r.level.lower()}-row" if r.level != "INFO" else "",
            level=r.level,
            level_lower=r.level.lower(),
            table=r.table,
            check=r.check,
            detail=r.detail,
            value=r.value if r.value is not None else "—",
            threshold=r.threshold if r.threshold is not None else "—",
        )
        for r in sorted(results, key=lambda x: {"CRITICAL": 0, "WARNING": 1, "INFO": 2}[x.level])
    )

    html = _TEMPLATE.format(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        total=len(results),
        n_critical=n_critical,
        n_warning=n_warning,
        n_passed=n_info,
        rows=rows_html,
    )

    out_path.write_text(html)
    print(f"Validation report written to: {out_path}")
