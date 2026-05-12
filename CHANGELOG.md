# Changelog

Changes to frankenbote.

---
## [unreleased]

### Miscellaneous Chores

- **(actions)** Add branch name and PR title validation steps to CI workflow (#13) - ([8523de8](http://192.168.2.134:3000/intipunku/frankenbote/commit/8523de86185640817a41f45bc25abac7d3c25323)) - kura_andi - 

---
## [0.1.3-beta](http://192.168.2.134:3000/intipunku/frankenbote/compare/v0.1.2-beta..v0.1.3-beta) - 2026-05-11

### Changed

- Merge pull request '#7 fixes wrong build-backend and added version injection for local builds which can't access .git' (#12) from ci_chain into main - ([c37179b](http://192.168.2.134:3000/intipunku/frankenbote/commit/c37179b99f7c2d613b50d870487f29e216a7d203)) - kura_andi - v0.1.3-beta
- #7 fixes wrong build-backend and added version injection for local builds which can't access .git - ([8987567](http://192.168.2.134:3000/intipunku/frankenbote/commit/8987567e1aa99539a926293bcbb7dc48e15a6764)) - kura_andi - v0.1.3-beta
- #7 added automated versioning on release, added release gitea action for distributing tags creating release and package - ([c937e25](http://192.168.2.134:3000/intipunku/frankenbote/commit/c937e252aa2b8609e1ca48c6dc3e02efe281a741)) - kura_andi - v0.1.3-beta
- #7 added docker stages to allow tests running inside docker, fixed failing test by removing exception handling from helper - ([5366d0e](http://192.168.2.134:3000/intipunku/frankenbote/commit/5366d0e945c5b129f24d1a1236558e7c1c5dbcd2)) - kura_andi - v0.1.3-beta
- #6 Omit files and methods which are non-deterministic (LLM calls) or require special set-up (SFTP upload). Updated to pytest-watcher for live change tests. Updated README to cover tests. - ([703ac44](http://192.168.2.134:3000/intipunku/frankenbote/commit/703ac44913119293a71f117c72440a56b624a728)) - kura_andi - v0.1.3-beta
- #6 added missing {__init__.py} for tests and considered coverage reports in {.gitignore} - ([342f91f](http://192.168.2.134:3000/intipunku/frankenbote/commit/342f91fe0c780f583154ab888c0d4ebd0f880181)) - kura_andi - v0.1.3-beta
- Add '{demo.yaml}' for testing Gitea actions. - ([b5cd2b1](http://192.168.2.134:3000/intipunku/frankenbote/commit/b5cd2b19323f6cc98ae1caab01d7c094adde0b8e)) - kura_andi - v0.1.3-beta
- #6 Setup initial test suite and a gitea action to run tests with each commit or pull request as requested for #7 - ([fdb08e8](http://192.168.2.134:3000/intipunku/frankenbote/commit/fdb08e8247f1a4bc87e1c468aacfc86fb5b8a29e)) - kura_andi - v0.1.3-beta

---
## [0.1.2-beta](http://192.168.2.134:3000/intipunku/frankenbote/compare/v0.1.0-beta..v0.1.2-beta) - 2026-05-07

### Changed

- Updated to latest version number - ([72e1216](http://192.168.2.134:3000/intipunku/frankenbote/commit/72e12160f77d3bb84a81ab05b613d73a9feba194)) - kura_andi - v0.1.2-beta
- Updated README and license information - ([f0014dd](http://192.168.2.134:3000/intipunku/frankenbote/commit/f0014dda0cd4b44c553f7ce969474f069830a6d8)) - kura_andi - v0.1.2-beta
- Added templates and assets folders to docker package - ([76c4aa4](http://192.168.2.134:3000/intipunku/frankenbote/commit/76c4aa43ee918c156d7ce8af2eca245ee56593ad)) - kura_andi - v0.1.2-beta

---
## [0.1.0-beta] - 2026-05-05

### Changed

- Add SFTP publisher with provider-agnostic config - ([9ed3272](http://192.168.2.134:3000/intipunku/frankenbote/commit/9ed32720f34be9a2473a812d2dacb4ad4b27eae9)) - kura_andi - v0.1.0-beta
- Add curator guidance: drop video-only items, welcome service journalism - ([9cdf370](http://192.168.2.134:3000/intipunku/frankenbote/commit/9cdf3702b037bc133b00037f0aa7cf96481245b3)) - kura_andi - v0.1.0-beta
- Added defensive methods: handle JSON arrays provided as strings from API and extended prompt to avoid German double quotes; simplified summary length - ([fb44ea0](http://192.168.2.134:3000/intipunku/frankenbote/commit/fb44ea0bbfb8e568f885bd3dc19d651f0c62124c)) - kura_andi - v0.1.0-beta
- Add summarizer; convert curator and summarizer to tool use; add debug-save on failure - ([627d67f](http://192.168.2.134:3000/intipunku/frankenbote/commit/627d67f292f8fed6c1fef84042f3f772f62a729d)) - kura_andi - v0.1.0-beta
- Add AI summarizer: clean German summaries in Spiegel voice - ([00a5ecc](http://192.168.2.134:3000/intipunku/frankenbote/commit/00a5eccd9d532114a4b1a4b24a430a6dda7e8099)) - kura_andi - v0.1.0-beta
- Added a lead article identifier for each section - ([80891fa](http://192.168.2.134:3000/intipunku/frankenbote/commit/80891faa1a2e4ef6e0bb1b9338b47f01b5c8a51a)) - kura_andi - v0.1.0-beta
- Link back edition pages to index.html, use same red in svg as for the page, show red borders for header and colophon - ([0781b07](http://192.168.2.134:3000/intipunku/frankenbote/commit/0781b077b34cd801a593ab9a3fef5c43763414ff)) - kura_andi - v0.1.0-beta
- Graphical improvements - ([d8186bd](http://192.168.2.134:3000/intipunku/frankenbote/commit/d8186bdbfcec26259f3eafa0409b2f5d191f59df)) - kura_andi - v0.1.0-beta
- Added renderer generating html - ([56c96d6](http://192.168.2.134:3000/intipunku/frankenbote/commit/56c96d6c99c15c31870711bc4e6252fa860bf42c)) - kura_andi - v0.1.0-beta
- Editorial changes for better selection results - ([4d468e4](http://192.168.2.134:3000/intipunku/frankenbote/commit/4d468e495a6075786314c028b1185f5b08218646)) - kura_andi - v0.1.0-beta
- Add selector: build final edition from curated articles - ([1e82c65](http://192.168.2.134:3000/intipunku/frankenbote/commit/1e82c65bd367a3cf16422d6a1d345ecef2ed4ff4)) - kura_andi - v0.1.0-beta
- Edited rss sources to cover a broader spectrum and reduce duplications of nationwide newspaters - ([e617d7a](http://192.168.2.134:3000/intipunku/frankenbote/commit/e617d7a9296bd892a8f1403dfcc531975962e12b)) - kura_andi - v0.1.0-beta
- Using streaming for API calls - ([7a55fad](http://192.168.2.134:3000/intipunku/frankenbote/commit/7a55fadce90fc27a8ad2c477a49d78f4caa2eaca)) - kura_andi - v0.1.0-beta
- Add AI curator: classify candidates with Claude Sonnet 4.6 - ([7ae48bd](http://192.168.2.134:3000/intipunku/frankenbote/commit/7ae48bdf49f49d2f83a43622dbd1ccc1c5dd4064)) - kura_andi - v0.1.0-beta
- Add filter stage with timezone-validated window. - ([e0093d6](http://192.168.2.134:3000/intipunku/frankenbote/commit/e0093d6600c871f7444e9d04ce00cdf81ec3f0f6)) - schmaeche - v0.1.0-beta
- Updated news sources: added real RSS links for BR24 and Nordbayern, all other sources currently disabled - ([703cbaa](http://192.168.2.134:3000/intipunku/frankenbote/commit/703cbaaa6eda1d321bf201f85b351c5d6acdfe32)) - schmaeche - v0.1.0-beta
- Add fetcher: pull articles from configured RSS feeds - ([ab4f838](http://192.168.2.134:3000/intipunku/frankenbote/commit/ab4f838be63ea6bb35a2be2c9f0d9fce0b4e4c56)) - schmaeche - v0.1.0-beta
- Initial project skeleton: Docker, CLI, hello command - ([71b1818](http://192.168.2.134:3000/intipunku/frankenbote/commit/71b1818df1ea65e66427e7ad1ccf6b8f17c2360c)) - schmaeche - v0.1.0-beta
- Updated description in README.md - ([938ea2f](http://192.168.2.134:3000/intipunku/frankenbote/commit/938ea2fbdfe58a8715cc5681024e4d774cd81c4f)) - schmaeche - v0.1.0-beta
- Initial commit - ([2ea1d45](http://192.168.2.134:3000/intipunku/frankenbote/commit/2ea1d452e1a48479e076747112c58fa18ee12288)) - kura_andi - v0.1.0-beta

<!-- generated by git-cliff -->
