# Changelog

Changes to frankenbote.

---
## [0.4.0](http://192.168.2.134:3000/intipunku/frankenbote/compare/v0.3.0..v0.4.0) - 2026-06-12

### Changed

- Add debug for understanding wrong call during pushes [no ci] - ([246fd4b](http://192.168.2.134:3000/intipunku/frankenbote/commit/246fd4b6cd64d4670967b3a56a75a686ba486e23)) - kura_andi
- Retrigger - ([1aeefd3](http://192.168.2.134:3000/intipunku/frankenbote/commit/1aeefd342e784c36c53c04c845b6328ed4fd6d17)) - kura_andi

### Documentation

- **(changelog)** Update changelog with recent feature additions and enhancements [no ci] - ([bdd8f66](http://192.168.2.134:3000/intipunku/frankenbote/commit/bdd8f665b51c169e061b51da3364a42869f4db90)) - kura_andi
- **(changelog)** Add commit preprocessor to enforce blank line before footer in conventional commits - ([d2569ab](http://192.168.2.134:3000/intipunku/frankenbote/commit/d2569abf22eb00e1e32abbb1edd2ee1877744aef)) - kura_andi
- **(readme)** Update README to include paywall detection module and its components [no ci] - ([631c5eb](http://192.168.2.134:3000/intipunku/frankenbote/commit/631c5ebd4fbba6d2b23c20035329995f8a34a7c9)) - kura_andi

### Features

- **(fetcher)** Update image extraction logic to prioritize content:encoded. Fixes #23 - ([c87c3d9](http://192.168.2.134:3000/intipunku/frankenbote/commit/c87c3d9495b6bc14f4e7851f81fd009f21cbe63d)) - kura_andi
- **(fetcher)** Enhance image extraction to support metaplus mp:image schema. Fixes #23 - ([c22f8b2](http://192.168.2.134:3000/intipunku/frankenbote/commit/c22f8b2cb93642e6742d4eb4d3e875f651894e09)) - kura_andi
- **(fetcher)** Enhance image extraction logic to prioritize enclosures and update tests - ([f227f92](http://192.168.2.134:3000/intipunku/frankenbote/commit/f227f92edbbdcfc66c611b1a8ebe9c938d4634ca)) - kura_andi
- **(fetcher)** Extract and handle image URLs from feed entries - ([023939b](http://192.168.2.134:3000/intipunku/frankenbote/commit/023939bbf6144708c89bf5377e91841c7cb6948b)) - kura_andi
- **(models)** Add image_url field to Article model - ([a0657c8](http://192.168.2.134:3000/intipunku/frankenbote/commit/a0657c85b3b20ee7cd79cc34ef2be67c2be85bfb)) - kura_andi
- **(paywall)** Add ContentLengthStrategy for paywall detection and update detector logic - ([686051b](http://192.168.2.134:3000/intipunku/frankenbote/commit/686051bf6dcbbc7ab531d6c6c4fa6f785e32d5ab)) - kura_andi
- **(paywall)** Implement paywall detection and selection logic, add tests for paywalled articles - ([ccf439e](http://192.168.2.134:3000/intipunku/frankenbote/commit/ccf439ea8020739b6d99952ef09bdd20bb459bc2)) - kura_andi
- **(paywall)** Implement paywall detection strategies and detector logic - ([386bed2](http://192.168.2.134:3000/intipunku/frankenbote/commit/386bed2b38d6fbb0582c718a186813f39eecf6fe)) - kura_andi
- **(renderer)** Add image display for articles with hero and thumbnail styles - ([2de057f](http://192.168.2.134:3000/intipunku/frankenbote/commit/2de057f90e814a187a17d769335fd8ab7f6d46cb)) - kura_andi
- **(styles)** Add wrap-up details section with toggle functionality in article template - ([4aa60ce](http://192.168.2.134:3000/intipunku/frankenbote/commit/4aa60cea9b8e109e6e7bfe156fa14a0c22544d70)) - kura_andi
- **(summarizer)** Modify selection filter to allow wrap-ups for all articles. fixes #28 - ([0e73800](http://192.168.2.134:3000/intipunku/frankenbote/commit/0e738000076e173a35923b65d5a9bd4773527c91)) - kura_andi
- **(tests)** Add test for rejecting unparseable URLs in image safety check - ([1fdc0a4](http://192.168.2.134:3000/intipunku/frankenbote/commit/1fdc0a4273d21efbbdb7d02ec2df4d640b9fd91b)) - kura_andi
- **(tests)** Add image extraction tests and update renderer tests for article images - ([9037ef2](http://192.168.2.134:3000/intipunku/frankenbote/commit/9037ef20b1ccc350f4a76cba890c1c601fc0fb91)) - kura_andi

### Miscellaneous Chores

- **(tests)** Add test for handling malformed HTML and JSON in paywall detection - ([84c78e3](http://192.168.2.134:3000/intipunku/frankenbote/commit/84c78e3a36e23744448ab0f6f5c6de08924d6655)) - kura_andi

### Refactoring

- **(summarizer)** Enhance logging for article body availability in batch wrap-up generation - ([cdd14e5](http://192.168.2.134:3000/intipunku/frankenbote/commit/cdd14e5145457d690545e8a364c94ea92974da69)) - kura_andi

---
## [0.3.0](http://192.168.2.134:3000/intipunku/frankenbote/compare/v0.3.0-beta3..v0.3.0) - 2026-05-28

### Documentation

- **(readme)** Add --batch-off option for synchronous API calls in pipeline commands. Documentation for #30 changes. - ([e3eb596](http://192.168.2.134:3000/intipunku/frankenbote/commit/e3eb5966a241739024ffce6d8011c4eea8aee1b9)) - kura_andi

### Features

- **(config)** Adjust target distribution and refine editorial guidance for curator. Fixes #17 - ([01b21e2](http://192.168.2.134:3000/intipunku/frankenbote/commit/01b21e2bc6643479cc491c540ec4a25db6db8f3a)) - kura_andi
- **(curator)** Implement batch processing for LLM classification and add tests for result extraction, Fixes #30 - ([842ca3b](http://192.168.2.134:3000/intipunku/frankenbote/commit/842ca3b92921d2fd353178c8293dc9a60ee527d6)) - kura_andi
- **(summarizer)** Add batch processing support for wrap-up generation and enhance tests for result extraction; Fixes #30 - ([4f2b907](http://192.168.2.134:3000/intipunku/frankenbote/commit/4f2b907998979129993f69bfb91a72a55269c2bb)) - kura_andi

### Miscellaneous Chores

- **(tests)** Enhance load_curator_config tests for error handling and valid YAML structure - ([a152206](http://192.168.2.134:3000/intipunku/frankenbote/commit/a152206fa77c79fafa28b24a1311d90713464aa8)) - kura_andi

---
## [0.3.0-beta2](http://192.168.2.134:3000/intipunku/frankenbote/compare/v0.3.0-beta1..v0.3.0-beta2) - 2026-05-26

### Miscellaneous Chores

- **(workflow)** Update registry variable references from 'variables' to 'vars' (#29) - ([106675d](http://192.168.2.134:3000/intipunku/frankenbote/commit/106675d3ddab854cfef0f56141efe16af834cff8)) - kura_andi

---
## [0.3.0-beta1](http://192.168.2.134:3000/intipunku/frankenbote/compare/v0.2.0..v0.3.0-beta1) - 2026-05-26

### Miscellaneous Chores

- **(workflow)** Replace hardcoded registry URL with workflow variables #29 [no ci] - ([e55064a](http://192.168.2.134:3000/intipunku/frankenbote/commit/e55064a3249395c900b68ef5193a758f8628a289)) - kura_andi

---
## [0.2.0](http://192.168.2.134:3000/intipunku/frankenbote/compare/v0.1.4-beta..v0.2.0) - 2026-05-20

### Bug Fixes

- **(styles)** Adjust font sizes for lead article title and masthead in responsive design (#20) - ([4bea8e6](http://192.168.2.134:3000/intipunku/frankenbote/commit/4bea8e648f1eeb23e3c7686754dc720e3b31ee34)) - kura_andi

### Documentation

- Added section for production deployment, reworked tests section, fixed broken link to CHANGELOG.md, removed version duplicates from changelog (#18) - ([0b1c69a](http://192.168.2.134:3000/intipunku/frankenbote/commit/0b1c69a018635e7fce72583392a4fb1b8e9b5ae3)) - kura_andi

### Features

- **(curator)** Add deduplication guidelines (#21) - ([0d40b97](http://192.168.2.134:3000/intipunku/frankenbote/commit/0d40b97a934d76695ab8a93b83666f6bf6d53af2)) - kura_andi
- **(renderer)** Remove rational display (#22) - ([8a437d7](http://192.168.2.134:3000/intipunku/frankenbote/commit/8a437d766f204c14c61124b18571a9d4b1268a6c)) - kura_andi
- **(selector)** Added external priority config #11 (#19) - ([eaa7b15](http://192.168.2.134:3000/intipunku/frankenbote/commit/eaa7b155a4f8bf5399d8de0659555ff7e8807d1d)) - kura_andi
- **(summarizer)** Implement generation of longer wrap-ups for lead articles and add body fetching functionality #3 (#26) - ([d0ca734](http://192.168.2.134:3000/intipunku/frankenbote/commit/d0ca7341768360648b3b11b4a0295d0d0375b1b8)) - kura_andi
- **(summarizer)** Add summarizer LLM configuration and loading functionality [no ci] (#25) - ([59c4daa](http://192.168.2.134:3000/intipunku/frankenbote/commit/59c4daa1de5bed141a9f4a907ba856ccd05774f2)) - kura_andi
- **(summarizer)** Add summarizer LLM configuration and loading functionality (#24) - ([ea2f336](http://192.168.2.134:3000/intipunku/frankenbote/commit/ea2f336275f642db501c331accec3fd5802f554e)) - kura_andi

---
## [0.1.4-beta](http://192.168.2.134:3000/intipunku/frankenbote/compare/v0.1.3-beta..v0.1.4-beta) - 2026-05-12

### Miscellaneous Chores

- **(actions)** Added changelog generator to release action workflow (#14) - ([d26ec17](http://192.168.2.134:3000/intipunku/frankenbote/commit/d26ec17ce55f7c9672f016b8dc761e4377ac9419)) - kura_andi
- **(actions)** Add branch name and PR title validation steps to CI workflow (#13) - ([8523de8](http://192.168.2.134:3000/intipunku/frankenbote/commit/8523de86185640817a41f45bc25abac7d3c25323)) - kura_andi

---
## [0.1.3-beta](http://192.168.2.134:3000/intipunku/frankenbote/compare/v0.1.2-beta..v0.1.3-beta) - 2026-05-11

### Changed

- Merge pull request '#7 fixes wrong build-backend and added version injection for local builds which can't access .git' (#12) from ci_chain into main - ([c37179b](http://192.168.2.134:3000/intipunku/frankenbote/commit/c37179b99f7c2d613b50d870487f29e216a7d203)) - kura_andi
- #7 fixes wrong build-backend and added version injection for local builds which can't access .git - ([8987567](http://192.168.2.134:3000/intipunku/frankenbote/commit/8987567e1aa99539a926293bcbb7dc48e15a6764)) - kura_andi
- #7 added automated versioning on release, added release gitea action for distributing tags creating release and package - ([c937e25](http://192.168.2.134:3000/intipunku/frankenbote/commit/c937e252aa2b8609e1ca48c6dc3e02efe281a741)) - kura_andi
- #7 added docker stages to allow tests running inside docker, fixed failing test by removing exception handling from helper - ([5366d0e](http://192.168.2.134:3000/intipunku/frankenbote/commit/5366d0e945c5b129f24d1a1236558e7c1c5dbcd2)) - kura_andi
- #6 Omit files and methods which are non-deterministic (LLM calls) or require special set-up (SFTP upload). Updated to pytest-watcher for live change tests. Updated README to cover tests. - ([703ac44](http://192.168.2.134:3000/intipunku/frankenbote/commit/703ac44913119293a71f117c72440a56b624a728)) - kura_andi
- #6 added missing {__init__.py} for tests and considered coverage reports in {.gitignore} - ([342f91f](http://192.168.2.134:3000/intipunku/frankenbote/commit/342f91fe0c780f583154ab888c0d4ebd0f880181)) - kura_andi
- Add '{demo.yaml}' for testing Gitea actions. - ([b5cd2b1](http://192.168.2.134:3000/intipunku/frankenbote/commit/b5cd2b19323f6cc98ae1caab01d7c094adde0b8e)) - kura_andi
- #6 Setup initial test suite and a gitea action to run tests with each commit or pull request as requested for #7 - ([fdb08e8](http://192.168.2.134:3000/intipunku/frankenbote/commit/fdb08e8247f1a4bc87e1c468aacfc86fb5b8a29e)) - kura_andi

---
## [0.1.2-beta](http://192.168.2.134:3000/intipunku/frankenbote/compare/v0.1.0-beta..v0.1.2-beta) - 2026-05-07

### Changed

- Updated to latest version number - ([72e1216](http://192.168.2.134:3000/intipunku/frankenbote/commit/72e12160f77d3bb84a81ab05b613d73a9feba194)) - kura_andi
- Updated README and license information - ([f0014dd](http://192.168.2.134:3000/intipunku/frankenbote/commit/f0014dda0cd4b44c553f7ce969474f069830a6d8)) - kura_andi
- Added templates and assets folders to docker package - ([76c4aa4](http://192.168.2.134:3000/intipunku/frankenbote/commit/76c4aa43ee918c156d7ce8af2eca245ee56593ad)) - kura_andi

---
## [0.1.0-beta] - 2026-05-05

### Changed

- Add SFTP publisher with provider-agnostic config - ([9ed3272](http://192.168.2.134:3000/intipunku/frankenbote/commit/9ed32720f34be9a2473a812d2dacb4ad4b27eae9)) - kura_andi
- Add curator guidance: drop video-only items, welcome service journalism - ([9cdf370](http://192.168.2.134:3000/intipunku/frankenbote/commit/9cdf3702b037bc133b00037f0aa7cf96481245b3)) - kura_andi
- Added defensive methods: handle JSON arrays provided as strings from API and extended prompt to avoid German double quotes; simplified summary length - ([fb44ea0](http://192.168.2.134:3000/intipunku/frankenbote/commit/fb44ea0bbfb8e568f885bd3dc19d651f0c62124c)) - kura_andi
- Add summarizer; convert curator and summarizer to tool use; add debug-save on failure - ([627d67f](http://192.168.2.134:3000/intipunku/frankenbote/commit/627d67f292f8fed6c1fef84042f3f772f62a729d)) - kura_andi
- Add AI summarizer: clean German summaries in Spiegel voice - ([00a5ecc](http://192.168.2.134:3000/intipunku/frankenbote/commit/00a5eccd9d532114a4b1a4b24a430a6dda7e8099)) - kura_andi
- Added a lead article identifier for each section - ([80891fa](http://192.168.2.134:3000/intipunku/frankenbote/commit/80891faa1a2e4ef6e0bb1b9338b47f01b5c8a51a)) - kura_andi
- Link back edition pages to index.html, use same red in svg as for the page, show red borders for header and colophon - ([0781b07](http://192.168.2.134:3000/intipunku/frankenbote/commit/0781b077b34cd801a593ab9a3fef5c43763414ff)) - kura_andi
- Graphical improvements - ([d8186bd](http://192.168.2.134:3000/intipunku/frankenbote/commit/d8186bdbfcec26259f3eafa0409b2f5d191f59df)) - kura_andi
- Added renderer generating html - ([56c96d6](http://192.168.2.134:3000/intipunku/frankenbote/commit/56c96d6c99c15c31870711bc4e6252fa860bf42c)) - kura_andi
- Editorial changes for better selection results - ([4d468e4](http://192.168.2.134:3000/intipunku/frankenbote/commit/4d468e495a6075786314c028b1185f5b08218646)) - kura_andi
- Add selector: build final edition from curated articles - ([1e82c65](http://192.168.2.134:3000/intipunku/frankenbote/commit/1e82c65bd367a3cf16422d6a1d345ecef2ed4ff4)) - kura_andi
- Edited rss sources to cover a broader spectrum and reduce duplications of nationwide newspaters - ([e617d7a](http://192.168.2.134:3000/intipunku/frankenbote/commit/e617d7a9296bd892a8f1403dfcc531975962e12b)) - kura_andi
- Using streaming for API calls - ([7a55fad](http://192.168.2.134:3000/intipunku/frankenbote/commit/7a55fadce90fc27a8ad2c477a49d78f4caa2eaca)) - kura_andi
- Add AI curator: classify candidates with Claude Sonnet 4.6 - ([7ae48bd](http://192.168.2.134:3000/intipunku/frankenbote/commit/7ae48bdf49f49d2f83a43622dbd1ccc1c5dd4064)) - kura_andi
- Add filter stage with timezone-validated window. - ([e0093d6](http://192.168.2.134:3000/intipunku/frankenbote/commit/e0093d6600c871f7444e9d04ce00cdf81ec3f0f6)) - schmaeche
- Updated news sources: added real RSS links for BR24 and Nordbayern, all other sources currently disabled - ([703cbaa](http://192.168.2.134:3000/intipunku/frankenbote/commit/703cbaaa6eda1d321bf201f85b351c5d6acdfe32)) - schmaeche
- Add fetcher: pull articles from configured RSS feeds - ([ab4f838](http://192.168.2.134:3000/intipunku/frankenbote/commit/ab4f838be63ea6bb35a2be2c9f0d9fce0b4e4c56)) - schmaeche
- Initial project skeleton: Docker, CLI, hello command - ([71b1818](http://192.168.2.134:3000/intipunku/frankenbote/commit/71b1818df1ea65e66427e7ad1ccf6b8f17c2360c)) - schmaeche
- Updated description in README.md - ([938ea2f](http://192.168.2.134:3000/intipunku/frankenbote/commit/938ea2fbdfe58a8715cc5681024e4d774cd81c4f)) - schmaeche
- Initial commit - ([2ea1d45](http://192.168.2.134:3000/intipunku/frankenbote/commit/2ea1d452e1a48479e076747112c58fa18ee12288)) - kura_andi

<!-- generated by git-cliff -->
