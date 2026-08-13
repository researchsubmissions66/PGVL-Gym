const methods = [
  "FOCUS",
  "MAPLE",
  "MSCPT",
  "MUSE",
  "PathPT",
  "ViLa-MIL",
  "CoD-MIL",
  "TOP",
  "SLIP",
  "WSI-FiVE",
  "ConVLM",
  "SLDPC",
];

const datasets = [
  ["TCGA-NSCLC", "LUAD / LUSC", "Patient-disjoint few-shot classification"],
  ["TCGA-BRCA", "Breast carcinoma", "Patient-disjoint few-shot classification"],
  ["TCGA-RCC", "KIRC / KIRP / KICH", "Patient-disjoint multiclass classification"],
  ["CAMELYON16", "Lymph-node metastasis", "Binary slide classification"],
  ["UBC-OCEAN", "Ovarian carcinoma", "Multiclass slide classification"],
];

function Arrow() {
  return <span aria-hidden="true">↗</span>;
}

export default function Home() {
  return (
    <main>
      <div className="progress" aria-hidden="true" />
      <nav className="site-nav" aria-label="Primary navigation">
        <div className="nav-inner">
          <a className="wordmark" href="#top" aria-label="PGVL-Gym home">
            PGVL<span>-Gym</span>
          </a>
          <div className="nav-links">
            <a href="#framework">Framework</a>
            <a href="#coverage">Coverage</a>
            <a href="#protocols">Protocols</a>
            <a href="https://researchsubmissions66.github.io/PGVL-Gym/">Documentation</a>
            <a href="https://github.com/researchsubmissions66/PGVL-Gym">GitHub</a>
          </div>
        </div>
      </nav>

      <header className="hero" id="top">
        <div className="constellation" aria-hidden="true">
          <span /><span /><span /><span /><span /><span />
        </div>
        <img
          className="hero-logo"
          src="/logo-gym.png"
          alt="PGVL-Gym: prompt-guided vision-language model benchmarking in computational pathology"
          width="1536"
          height="1024"
        />
        <p className="kicker">Generalized pathology VLM evaluation</p>
        <h1>A fair, configuration-first gym for whole-slide vision-language models.</h1>
        <p className="hero-copy">
          One framework for reproducible few-shot and zero-shot evaluation across
          model families, cohorts, feature spaces, resolutions, and slide encoders.
        </p>
        <div className="hero-actions">
          <a className="button button-primary" href="#framework">Explore the framework</a>
          <a className="button button-secondary" href="https://researchsubmissions66.github.io/PGVL-Gym/getting-started/">
            Read the documentation <Arrow />
          </a>
          <a className="button button-secondary" href="https://github.com/researchsubmissions66/PGVL-Gym">
            View source <Arrow />
          </a>
        </div>
      </header>

      <section className="metrics" aria-label="Framework coverage">
        <article><strong>13+</strong><span>model families</span></article>
        <article><strong>5</strong><span>cohort protocols</span></article>
        <article><strong>20</strong><span>experiment variants</span></article>
        <article><strong>∞</strong><span>feature backbones</span></article>
      </section>

      <section className="section" id="framework">
        <div className="section-heading">
          <p className="eyebrow">Framework overview</p>
          <h2>Standardize the experiment, not the model.</h2>
          <p>
            PGVL-Gym keeps each paper&apos;s architecture intact while giving every
            method the same explicit experiment contract. Comparisons become
            traceable without flattening meaningful model differences.
          </p>
        </div>

        <div className="feature-grid">
          <article className="feature-card">
            <span className="feature-number">01</span>
            <h3>Protocol as configuration</h3>
            <p>Labels, prompts, folds, shots, resolutions, and paths live in readable protocol files.</p>
          </article>
          <article className="feature-card">
            <span className="feature-number">02</span>
            <h3>Explicit provenance</h3>
            <p>Patch and slide tensors declare encoder, feature space, level, dimension, and magnification.</p>
          </article>
          <article className="feature-card">
            <span className="feature-number">03</span>
            <h3>Fair by construction</h3>
            <p>Shared seeds, identical shots, aligned folds, and patient-disjoint splits support fair comparison.</p>
          </article>
        </div>
      </section>

      <section className="section section-tinted" id="coverage">
        <div className="section-heading compact">
          <p className="eyebrow">Model coverage</p>
          <h2>One registry. Distinct model families.</h2>
          <p>
            Each adapter declares what it needs: patch bags, dual-resolution
            features, slide embeddings, prompt encoders, or paired slide-text towers.
          </p>
        </div>
        <div className="method-cloud" aria-label="Supported model families">
          {methods.map((method, index) => (
            <span key={method}><small>{String(index + 1).padStart(2, "0")}</small>{method}</span>
          ))}
          <span><small>+</small>Composite variants</span>
        </div>

        <div className="capability-grid">
          <article>
            <h3>Patch-level paths</h3>
            <p>Static feature bags, variable dimensions, and separately declared low/high resolutions.</p>
            <strong>Single-scale · dual-scale · dynamic encoders</strong>
          </article>
          <article>
            <h3>Slide-level paths</h3>
            <p>Slide-encoder-agnostic loading with explicit provenance and projection contracts.</p>
            <strong>TITAN and compatible registered encoders</strong>
          </article>
          <article>
            <h3>Prompt paths</h3>
            <p>Dataset labels, templates, learned context, and method-owned text towers remain configurable.</p>
            <strong>Zero-shot · few-shot · prompt learning</strong>
          </article>
        </div>
      </section>

      <section className="section" id="protocols">
        <div className="section-heading">
          <p className="eyebrow">Benchmark protocols</p>
          <h2>Five cohorts, one comparison contract.</h2>
          <p>
            Cohorts plug into the framework through labels, prompts, manifests,
            patient groups, feature sources, and fold definitions—without changing training code.
          </p>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Dataset</th><th>Clinical target</th><th>Evaluation task</th></tr>
            </thead>
            <tbody>
              {datasets.map(([name, target, task]) => (
                <tr key={name}><td><strong>{name}</strong></td><td>{target}</td><td>{task}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="section workflow-section">
        <div className="section-heading compact">
          <p className="eyebrow">Run lifecycle</p>
          <h2>From feature store to comparable report.</h2>
        </div>
        <ol className="workflow">
          <li><span>01</span><strong>Declare</strong><small>task, paths, labels</small></li>
          <li><span>02</span><strong>Validate</strong><small>contract and provenance</small></li>
          <li><span>03</span><strong>Train</strong><small>shared folds and shots</small></li>
          <li><span>04</span><strong>Evaluate</strong><small>held-out patients</small></li>
          <li><span>05</span><strong>Report</strong><small>aligned metrics</small></li>
        </ol>
      </section>

      <section className="final-cta">
        <p className="eyebrow">Ready to benchmark?</p>
        <h2>Start from a validated configuration.</h2>
        <p>Build a smoke-tested run before committing expensive GPU hours.</p>
        <div className="hero-actions">
          <a className="button button-light" href="https://researchsubmissions66.github.io/PGVL-Gym/getting-started/">Get started <Arrow /></a>
          <a className="button button-outline" href="https://github.com/researchsubmissions66/PGVL-Gym">Browse the repository <Arrow /></a>
        </div>
      </section>

      <footer>
        <a className="wordmark footer-wordmark" href="#top">PGVL<span>-Gym</span></a>
        <p>Prompt-guided vision-language model benchmarking in computational pathology.</p>
        <div><a href="https://researchsubmissions66.github.io/PGVL-Gym/">Documentation</a><a href="https://github.com/researchsubmissions66/PGVL-Gym">GitHub</a></div>
        <small>Submitted for anonymous review.</small>
      </footer>
    </main>
  );
}
