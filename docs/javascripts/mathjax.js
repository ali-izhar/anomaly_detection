// MathJax 3 configuration for mkdocs-material + pymdownx.arithmatex (generic mode).
// Reference: https://squidfunk.github.io/mkdocs-material/reference/math/
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    // Only process elements arithmatex wrapped; ignore everything else
    // (e.g., code blocks with literal `$` or `\(` characters).
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// If Material's instant-navigation (`navigation.instant`) is enabled, re-typeset
// on each client-side page transition. Guarded because `document$` is only
// defined when the instant feature is active; without the guard this throws.
if (typeof document$ !== "undefined") {
  document$.subscribe(() => {
    MathJax.startup?.output?.clearCache && MathJax.startup.output.clearCache();
    MathJax.typesetClear && MathJax.typesetClear();
    MathJax.texReset && MathJax.texReset();
    MathJax.typesetPromise && MathJax.typesetPromise();
  });
}
