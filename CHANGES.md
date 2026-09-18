# Reliability and packaging revision

Based on uploaded snapshot 24fd968abc60c94297b008985ca34b4a39f441e7.
Version remains 0.1.6: this is an unpublished source revision.

## Changes

- Added pyproject.toml with runtime dependencies and notebook/materials/dev extras.
- Quiet package import: no scientific dependency probing or scratch creation.
- Empty, malformed, unreadable API-key files cannot crash package import.
- print_defaults() redacts the entire API key.
- Lazy GSAS-II loader supports installed packages and explicit legacy directories.
- setup_gsas2_refiner() honors its path argument and fails before clearing results.
- Failed legacy imports restore sys.path. GPX loading/reloading uses the loader.
- Scratch directories are created when phase/refinement operations need them.
- Added JSON diagnostics, updated installation instructions, regression tests,
  and a Python 3.11–3.14 startup/build GitHub Actions workflow.

## Compatibility notes

Existing exrd method signatures are preserved. Scientific algorithms and plotting
behavior were not changed. Import no longer prints status: use print_defaults()
or python -m easyxrd.diagnostics. gsasii_lib_path remains 'not found' unless set
by the caller; discovery happens at operation time. Legacy installations formerly
found through scratch or g2full paths must provide an explicit GSASII directory
or expose it through the environment. Backend failures raise exceptions instead
of requesting interactive input. Restart Python before switching GSAS-II
installations because imported modules are cached. Notebook and Materials Project
dependencies are now optional extras.

## Validation

12 unittest tests passed on Python 3.12.14/Linux. GSAS-II tests use mocked
backends; one test compiles the real setup method in isolation to verify that
backend failure preserves existing refinement state. All package files compiled.
A wheel built successfully with pip wheel --no-deps --no-build-isolation.
The diagnostic command ran successfully, including NumPy header discovery.

The full scientific stack and GSAS-II are not installed here. End-to-end
integration, refinement, plotting, NetCDF round trips, full dependency resolution,
and cross-version CI were not run. CI covers startup and packaging only.
Dependencies are not locked and no tested scientific compatibility matrix is claimed.

## Next priorities

1. Reference-image integration and known-phase refinement regression tests.
2. NetCDF round-trip tests before restructuring core.py.
3. Validate input ambiguity, radial ranges, mask shapes, and NaNs in load_xrd_data.
4. Replace broad exception handlers in export/refinement with targeted handling.
5. Review global warning suppression and plotting settings for notebook compatibility.
6. Separate processing, plotting, and refinement gradually; benchmark before adding
   concurrency. HiddenPrints remains process-wide and is not thread-safe.
