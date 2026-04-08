---
title: DataCleanEnv-X
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# DataCleanEnv-X (DCX)

Data cleaning and validation environment for structured datasets natively compliant with strict OpenEnv metrics!

## Task Environments

The `DCX` environment natively supports three distinct configurations evaluated under robust heuristics tracking dataset limits explicitly:

- **Easy**: Fix missing values and handle straightforward type coercing anomalies.
- **Medium**: Deduplicate fields, normalize explicit structures natively scaling strings predictably.
- **Hard**: Full pipeline executing constraints across outliers explicitly handling invalid row metrics organically dynamically bounding constraints natively!

## Observation & Action Space

### Observation Space
Observations map directly into schema specifications matching natively bounded formats:
- `sample_rows`: Glimpse evaluating explicitly up to 20 native slices.
- `column_profiles`: Dynamic dictionary formatting counts natively.
- `detected_issues`: High, Medium, Low severity alerts tracing explicit loops correctly handling anomaly ratios natively!
- `progress` / `step_budget_remaining`: Tracks exact metrics evaluated securely mapping limits directly ensuring bounded lengths!

### Action Space
Your agents natively execute actions under the `ActionModel` bounding mappings directly to:
`type(ActionType)`, `column(str | None)`, and `parameters(dict)`.

Supported actions:
- `fill_value`, `cast_type`, `normalize_field`
- `drop_row`, `flag_invalid`, `deduplicate`
- `handle_outliers`, `escalate`, `finish`

## Quick Start
You can run this project out of the box leveraging pre-configured API bounds seamlessly natively securely!

### Requirements
```bash
pip install -r requirements.txt
```

### Docker Deployments (Huggingface Ready)
```bash
docker build -t dcx .
docker run -p 7860:7860 dcx
```
Check status:
```bash
curl http://localhost:7860/reset
```

### Inference Tests 

Ensure your `$OPENAI_API_KEY`, `$MODEL_NAME`, and `$API_BASE_URL` are strictly exported inside your native local paths or environment container securely executing:

```bash
python inference.py
```

### Expected Flow
You will receive JSON bound streams trailing sequentially cleanly evaluating outputs smoothly tracking scores limits scaling bounded smoothly:
```
[START]
task=easy
[STEP]
step=1
action_type=fill_value
column=age
reward=0.1
[END]
task=easy
score=0.8
```
