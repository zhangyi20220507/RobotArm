"""Compatibility entrypoint; prefer robotarm modeling pi0-training-report."""
from robotarm.modeling.reports.pi0_training import main, points, render, summarize_predictions
from robotarm.modeling.reports.html import chart

if __name__ == '__main__':
    raise SystemExit(main())
