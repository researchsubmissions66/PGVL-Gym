# Deploy the project website

The repository publishes two independent surfaces in one GitHub Pages artifact:

- documentation: `https://<owner>.github.io/<repository>/`
- project website: `https://<owner>.github.io/<repository>/project/`

For this repository, the project website URL is:

```text
https://researchsubmissions66.github.io/PGVL-Gym/project/
```

## First-time repository setup

1. Open the repository on GitHub.
2. Select **Settings → Pages**.
3. Under **Build and deployment**, choose **GitHub Actions** as the source.
4. Open **Actions → Documentation** and select **Run workflow**, or push to
   `main`.

The workflow builds MkDocs, builds the static project website with its
repository-relative base path, merges the outputs, and deploys one Pages
artifact. It requires no deployment secret.

## Automatic deployment

Any push to `main` that passes the workflow updates both sites:

```bash
git add .
git commit -m "Update project website"
git push origin main
```

Pull requests run both builds without publishing. The deployment job runs only
after a push to the default branch.

## Build the Pages version locally

From the repository root:

```bash
cd project-website
npm ci
PAGES_BASE_PATH=/PGVL-Gym/project/ npm run build:pages
npm run preview:pages
```

The static output is written to `project-website/dist-pages/`. The normal
`npm run build` command remains the deployment build for the private Sites
version; `npm run build:pages` is the static GitHub Pages build.

## Deploy from a fork

GitHub project sites are served below the repository name. Replace the base path
with the fork's exact, case-sensitive repository name:

```bash
PAGES_BASE_PATH=/<repository-name>/project/ npm run build:pages
```

The included workflow derives that path automatically from
`github.event.repository.name`, so manual configuration is normally
unnecessary.

## Troubleshooting

- **The site has no styling or images:** confirm that the Pages source is
  **GitHub Actions** and that the repository name matches the URL exactly.
- **The documentation disappeared:** use the included combined workflow; do not
  deploy `dist-pages/` directly over the Pages root.
- **The project route returns 404:** confirm the latest **Documentation**
  workflow completed successfully and open the URL with the trailing slash.
- **A fork does not deploy:** enable Pages in the fork and allow GitHub Actions
  under the repository's Actions settings.
