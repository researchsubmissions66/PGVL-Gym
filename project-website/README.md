# PGVL-Gym project website

This directory contains the standalone PGVL-Gym research-project website. It
shares content and branding across two deployment targets:

- a static GitHub Pages build at `/PGVL-Gym/project/`;
- the optional private Sites deployment.

## Local development

```bash
npm ci
npm run dev
```

## Validate both deployment targets

```bash
npm test
PAGES_BASE_PATH=/PGVL-Gym/project/ npm run build:pages
```

- `npm run build` produces the Sites-compatible build in `dist/`.
- `npm run build:pages` produces the static build in `dist-pages/`.
- `npm run preview:pages` previews the last static build.

See [DEPLOYMENT.md](DEPLOYMENT.md) for GitHub Pages setup, automatic
deployment, fork instructions, and troubleshooting.
