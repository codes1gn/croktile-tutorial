function highlightCode() {
  document.querySelectorAll('pre code[class*="language-"]').forEach(function(el) {
    delete el.dataset.highlighted;
    hljs.highlightElement(el);
  });
}

document.addEventListener('DOMContentLoaded', highlightCode);

document$.subscribe(highlightCode);
