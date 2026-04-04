# ML5

This folder contains the Milestone 5 batch-evaluation layer requested for
Mayank Dode's contribution area. It is intentionally separate from the core
`src/` pipeline code and reuses the existing `CMSVSPipeline` implementation.

## What is included

- `configuration.py`: YAML manifest parser for batch jobs
- `discovery.py`: SBC and FUNSD pair collection / preprocessing helpers
- `pipeline.py`: batch runner that executes the existing pipeline and writes
  per-pair system outputs plus a batch summary manifest
- `cli.py`: lightweight entry point for batch execution
- `configs/sbc_batch.yaml`: default manifest for all 8 SBC evaluation pairs
- `configs/funsd_batch.yaml`: default manifest for FUNSD image-pair execution

## Output behavior

Each batch run writes:

- one JSON report per discovered document pair
- `batch_summary.json` describing which pairs ran, output locations, and any
  failures

The default manifests use `report_format: groundtruth` so the outputs match the
existing M2/M5 evaluation JSON shape already produced by `ReportGenerator`.

## Usage

```bash
python ML5/cli.py --manifest ML5/configs/sbc_batch.yaml
python ML5/cli.py --manifest ML5/configs/funsd_batch.yaml
```

The runner expects `NVIDIA_API_KEY` to be set, because it delegates execution
to the existing CMSVS pipeline.
