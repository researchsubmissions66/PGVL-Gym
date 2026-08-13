---
hide:
  - toc
---

<section class="pgvl-hero" aria-labelledby="pgvl-title">
  <div class="pgvl-hero__content">
    <h1 id="pgvl-title">PGVL-Gym</h1>
    <p class="pgvl-hero__lede">
      A configuration-first framework for systematic, reproducible evaluation
      of whole-slide pathology vision-language models.
    </p>
    <div class="pgvl-actions">
      <a class="md-button pgvl-button pgvl-button--primary" href="getting-started/">Run your first benchmark <span aria-hidden="true">→</span></a>
      <a class="md-button pgvl-button pgvl-button--primary" href="architecture/">Explore the architecture</a>
    </div>
  </div>
  <figure class="pgvl-hero__visual">
    <img src="assets/logo_gym.png" alt="PGVL-Gym pathology vision-language benchmark artwork" width="1536" height="1024">
  </figure>
</section>

<div class="pgvl-statbar" aria-label="Framework coverage">
  <div><strong>20</strong><span>experiment variants</span></div>
  <div><strong>13+</strong><span>model families</span></div>
  <div><strong>5</strong><span>cohort protocols</span></div>
  <div><strong>∞</strong><span>feature backbones</span></div>
</div>

## The benchmark contract

PGVL-Gym turns every experiment into the same explicit contract: a method
adapter, a dataset protocol, and traceable feature provenance. The result is a
comparison you can explain, reproduce, and extend.

<div class="pgvl-card-grid">
  <article class="pgvl-card">
    <span class="pgvl-card__number">01</span>
    <h3>Configure the task</h3>
    <p>Labels, prompts, folds, shots, resolutions, and paths stay in readable protocol files—not hidden in training code.</p>
    <a href="configuration/">Configuration guide <span aria-hidden="true">→</span></a>
  </article>
  <article class="pgvl-card">
    <span class="pgvl-card__number">02</span>
    <h3>Declare provenance</h3>
    <p>Every cached tensor records its encoder, feature space, level, and resolution so incompatible assets fail early.</p>
    <a href="architecture/">Feature architecture <span aria-hidden="true">→</span></a>
  </article>
  <article class="pgvl-card">
    <span class="pgvl-card__number">03</span>
    <h3>Compare fairly</h3>
    <p>Shared seeds, patient-disjoint folds, identical shot counts, and aligned reporting keep the scoreboard honest.</p>
    <a href="benchmarks/">Benchmark protocols <span aria-hidden="true">→</span></a>
  </article>
</div>

## From feature store to fair result

<div class="pgvl-flow" role="list" aria-label="Benchmark workflow">
  <div class="pgvl-flow__step" role="listitem"><span>01</span><strong>Features</strong><small>patch or slide level</small></div>
  <div class="pgvl-flow__connector" aria-hidden="true">→</div>
  <div class="pgvl-flow__step" role="listitem"><span>02</span><strong>Protocol</strong><small>task, fold, shots</small></div>
  <div class="pgvl-flow__connector" aria-hidden="true">→</div>
  <div class="pgvl-flow__step" role="listitem"><span>03</span><strong>Adapter</strong><small>declared capability</small></div>
  <div class="pgvl-flow__connector" aria-hidden="true">→</div>
  <div class="pgvl-flow__step" role="listitem"><span>04</span><strong>Report</strong><small>comparable metrics</small></div>
</div>

FOCUS, MAPLE, MSCPT, MUSE, PathPT, ViLa-MIL, CoD-MIL, TOP, SLIP, WSI-FiVE,
ConVLM, SLDPC, and composite variants all enter through the same registry. The
generated protocols currently cover TCGA-NSCLC, TCGA-BRCA, TCGA-RCC,
CAMELYON16, and UBC-OCEAN.

!!! note "Model fidelity is part of fairness"

    The shared interface standardizes loading, validation, and orchestration.
    It never silently redesigns a paper architecture. Fixed and allowlisted
    encoder boundaries remain explicit—even when another encoder has the same
    output dimension.

<div class="pgvl-next">
  <div>
    <h2>Start with a validated configuration.</h2>
    <p>Build a smoke-tested run before committing expensive GPU hours.</p>
  </div>
  <a class="md-button pgvl-button pgvl-button--primary" href="getting-started/">Get started <span aria-hidden="true">→</span></a>
</div>
