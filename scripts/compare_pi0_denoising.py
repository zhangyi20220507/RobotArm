"""Compatibility entrypoint; prefer robotarm modeling pi0-denoising-report."""
from robotarm.modeling.reports.pi0_denoising import (
    main, audit, outside, select_indices, summarize, paired_difference, render, conclusions,
)
from robotarm.modeling.reports.html import table

if __name__ == '__main__':
    raise SystemExit(main())
