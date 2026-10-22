"""Compatibility entrypoint; prefer robotarm modeling act-training-report."""
from robotarm.modeling.reports.act_training import main

if __name__ == '__main__':
    raise SystemExit(main())
