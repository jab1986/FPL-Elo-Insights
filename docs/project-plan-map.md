# FPL-Elo-Insights Delivery Plan & System Map

## 1. Programme Goals
- Restore a runnable, reproducible ML pipeline that supports ridge baselines, position-specific ensembles, and the deep stack without leaking labels or requiring optional dependencies.
- Ship stakeholder-ready evaluation artefacts (advanced diagnostics, Tableau exports) from the automated run so dashboards and alerts consume the same payloads.
- Harden the production rails (monitoring, retraining, documentation, dependency toggles) to support a small team operating the system over a Premier League season.

## 2. Current State Snapshot
- **Pipeline import blocked**: `_train_model_by_type` contains an empty `else` branch, yielding an `IndentationError` and preventing `python -m ml.pipeline` from executing.
- **Evaluation unwired**: `_run_enhanced_evaluation` is never invoked, so `PipelineResult.evaluation` stays `None` and Tableau exports miss advanced statistics.
- **Deep ensemble mis-specified**: the meta-learner ingests the ground-truth target during training and sees zeros at inference; base Keras layers mutate `input_dim` post-construction.
- **Reporting fragile**: formatted exports rely on malformed f-strings and classifier-style calibration metrics that are not valid for regression outputs.
- **SBSS claims aspirational**: documentation references SBSS, but no ingest, features, or configuration exist in the repo.

## 3. Architecture Map
| Layer | Key Components | Owner Teams | Current Gaps |
| --- | --- | --- | --- |
| **Data & Features** | `ml/features.py`, rolling/lag pipelines, Elo differentials | Data Science | SBSS feature hooks missing; need verification for NaN guards |
| **Model Training** | `PointsPredictionPipeline`, `PositionSpecificModel`, `DeepEnsembleModel` | ML Engineering | Import failure, label leakage, heavy training defaults |
| **Evaluation & Monitoring** | `AdvancedEvaluator`, `evaluation_export.py`, drift monitors | MLOps | Advanced evaluation not attached, calibration metrics misused |
| **Serving & Automation** | `backend/app/`, Typer CLI, retraining scripts | Platform | CLI blocked by pipeline import, optional deps not guarded |
| **BI & Stakeholders** | Tableau dashboards, docs in `/docs` | Analytics | Exports missing metrics, SBSS status unclear |

## 4. Delivery Roadmap
### Phase 0 – Triage (Day 0)
1. Reinstate the missing `raise ValueError(f"Unknown model type: {model_type}")` inside `_train_model_by_type` and ensure `_run_enhanced_evaluation` is defined at module scope.
2. Add a regression test invoking `python -m ml.pipeline` (via `subprocess`) to catch syntax regressions.
3. Document dependency expectations in `ml/README.md` and `docs/quick-start.md`.

### Phase 1 – Pipeline Restoration (Days 1–3)
1. Call `_run_enhanced_evaluation` inside `PointsPredictionPipeline.run` and assign results to `PipelineResult.evaluation`.
2. Correct export/report formatting (`{value:.4f}`) and wrap Tableau exporters so they skip gracefully when evaluation data is missing.
3. Introduce configuration flags for LightGBM/XGBoost/TensorFlow with clear fallbacks.

### Phase 2 – Ensemble Rebuild (Days 4–7)
1. Regenerate deep ensemble meta-features using out-of-fold predictions; remove target leakage from the meta-network inputs.
2. Define explicit `Input` layers or `input_shape` arguments during Keras model construction; replace manual `input_dim` mutations.
3. Tune epoch/patience defaults to align with CPU-only runs (e.g., 50 epochs, 5-patience) and expose overrides via pipeline config.

### Phase 3 – Evaluation & Monitoring (Days 8–10)
1. Replace classifier calibration with regression-aware diagnostics (residual histograms, prediction interval coverage).
2. Ensure Tableau/export helpers consume the new evaluator API; add unit tests covering CSV generation and formatting.
3. Wire monitoring scaffolding into an integration test that verifies PSI/KS computation and alert trigger paths using mock data.

### Phase 4 – SBSS & Roadmap Alignment (Days 11–14)
1. Draft SBSS ingestion design (source systems, refresh cadence, schema) and capture it in `docs/SBSS_plan.md`.
2. Prototype feature hooks for SBSS metrics behind a feature flag; validate against historical data before enabling by default.
3. Update stakeholder docs and dashboards to reflect actual capabilities; remove or annotate aspirational claims.

## 5. Cross-Cutting Workstreams
- **Testing Strategy**: expand `pytest` coverage for pipeline import, evaluation attachment, monitoring flows, and exporters; add smoke CLI invocation to CI.
- **Documentation**: refresh README, quick-start, and Tableau guides as behaviour changes; log verification commands in `CURRENT_WORK_STATUS.md` when deviating from defaults.
- **Operational Readiness**: configure alert channels (email/Slack) in infrastructure-as-code, validate retraining triggers, and script rollback procedures.
- **Team Enablement**: run knowledge-sharing sessions on the new evaluator and ensemble design; provide runbooks for support rotations.

## 6. Milestones & Checkpoints
| Milestone | Target Date | Exit Criteria |
| --- | --- | --- |
| **M1 – Pipeline Bootstrapped** | Day 3 | CLI run succeeds; evaluation stored on `PipelineResult`; docs updated with dependency guidance |
| **M2 – Ensemble Stabilised** | Day 7 | Deep ensemble trains without leakage, uses explicit inputs, completes within baseline runtime |
| **M3 – Evaluation Operational** | Day 10 | Regression diagnostics live, Tableau export verified in CI, monitoring integration test passes |
| **M4 – SBSS Plan Ready** | Day 14 | SBSS design document published, feature hooks prototyped and flagged |

## 7. Open Questions
- What production environment constraints (CPU vs GPU) should drive default training budgets?
- Which Tableau dashboards are actively consumed, and what data contracts must we honour?
- Are there regulatory or privacy considerations when ingesting new SBSS data sources?
- How will optional dependency failures be surfaced to on-call engineers (logs, alerts, dashboards)?

## 8. Next Actions for This Sprint
1. Fix `_train_model_by_type` and unblock the pipeline import.
2. Attach advanced evaluation output to the pipeline result and patch export formatting.
3. Produce a spike branch for the deep ensemble rebuild to validate OOF meta-features and new Keras inputs.
