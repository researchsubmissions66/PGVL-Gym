# Project website deployment

PGVL-Gym keeps its documentation and project website separate without requiring
a second repository:

| Surface | GitHub Pages URL |
| --- | --- |
| Documentation | `https://researchsubmissions66.github.io/PGVL-Gym/` |
| Project website | `https://researchsubmissions66.github.io/PGVL-Gym/project/` |

The project website is a static build, so GitHub Pages needs no server, runtime
variables, or deployment secrets.

## Enable GitHub Pages

For the original repository or a fork:

1. Open **Settings → Pages** in GitHub.
2. Set **Build and deployment → Source** to **GitHub Actions**.
3. Open **Actions → Documentation**.
4. Choose **Run workflow**, or push a commit to `main`.

The included workflow builds both surfaces, places the project website below
`/project/`, uploads one Pages artifact, and deploys it. This prevents the
project website from overwriting the documentation.

## Automatic deployment

Every push to `main` runs the deployment automatically:

```bash
git add .
git commit -m "Update project website"
git push origin main
```

Pull requests build and validate both sites but do not publish.

## Build locally

```bash
cd project-website
npm ci
PAGES_BASE_PATH=/PGVL-Gym/project/ npm run build:pages
npm run preview:pages
```

The generated static site is written to `project-website/dist-pages/`. Use
`npm run build` for the private Sites target and `npm run build:pages` for
GitHub Pages.

## Deploy from a fork

The workflow calculates the base path from the repository name. No source edit
is needed. The fork's URL will be:

```text
https://<owner>.github.io/<repository-name>/project/
```

Repository names and URL paths are case-sensitive.

## Troubleshooting

### Styling or images are missing

Confirm that Pages uses **GitHub Actions**, not a branch folder. Then check that
the URL contains the exact repository name and ends in `/project/`.

### The project route returns 404

Open **Actions → Documentation** and confirm both the build and deploy jobs
succeeded. Pages may take a short time to refresh after the workflow completes.

### The documentation disappeared

Do not upload `project-website/dist-pages/` directly as the Pages root. Restore
and run the included combined workflow, which merges the static site into
`site/project/` after MkDocs finishes.
