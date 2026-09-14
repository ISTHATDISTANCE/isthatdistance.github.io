// Native disclosure controls and publication links work without JavaScript.
// Enhance the page with topic filtering and citation copying when available.
(() => {
  "use strict";

  function initializeResearch() {
    document.querySelectorAll("[data-research]").forEach((section) => {
      const publications = Array.from(section.querySelectorAll(".research-publication"));
      const filters = Array.from(section.querySelectorAll("[data-research-filter]"));
      const filterGroup = section.querySelector(".research-filters");
      const count = section.querySelector(".research-count");
      const empty = section.querySelector(".research-empty");

      function filterPublications(activeButton) {
        const topic = activeButton.dataset.researchFilter;
        let visibleCount = 0;

        publications.forEach((publication) => {
          const topics = (publication.dataset.topic || "").split(/\s+/);
          const visible = topic === "all" || topics.includes(topic);
          publication.hidden = !visible;
          const listItem = publication.closest("li");
          if (listItem) listItem.hidden = !visible;
          if (visible) visibleCount += 1;
        });

        // Jekyll Scholar may render a separate list and heading for each year.
        section.querySelectorAll("ol.bibliography").forEach((list) => {
          const hasVisiblePublication = Array.from(list.querySelectorAll(".research-publication")).some((publication) => !publication.hidden);
          list.hidden = !hasVisiblePublication;
          const heading = list.previousElementSibling;
          if (heading && /^H[2-6]$/.test(heading.tagName)) heading.hidden = !hasVisiblePublication;
        });

        filters.forEach((button) => button.setAttribute("aria-pressed", String(button === activeButton)));
        if (count) {
          const label = visibleCount === 1 ? "publication" : "publications";
          const topicLabel = topic === "all" ? "" : ` in ${activeButton.textContent.trim().toLowerCase()}`;
          count.textContent = `${visibleCount} ${label}${topicLabel}`;
        }
        if (empty) empty.hidden = visibleCount > 0;
      }

      if (publications.length && filters.length) {
        filters.forEach((button) => button.addEventListener("click", () => filterPublications(button)));
        filterPublications(filters.find((button) => button.dataset.researchFilter === "all") || filters[0]);
        if (filterGroup) filterGroup.hidden = false;
      }
    });

    document.querySelectorAll(".research-citation").forEach((citation) => {
      const button = citation.querySelector(".research-copy");
      const code = citation.querySelector("pre code");
      const status = citation.querySelector(".research-copy-status");
      const controls = citation.querySelector(".research-copy-controls");
      if (!button || !code || !status) return;
      if (controls) controls.hidden = false;

      button.addEventListener("click", async () => {
        button.disabled = true;
        status.textContent = "";
        try {
          const text = code.textContent.trim();
          let copied = false;
          if (navigator.clipboard && window.isSecureContext) {
            try {
              await navigator.clipboard.writeText(text);
              copied = true;
            } catch (_error) {
              // Clipboard permission can be denied even on a secure page.
            }
          }
          if (!copied) copied = copyWithSelection(text);
          if (!copied) throw new Error("Clipboard unavailable");
          status.textContent = "Citation copied.";
        } catch (_error) {
          status.textContent = "Could not copy automatically. Select and copy the citation below.";
        } finally {
          button.disabled = false;
        }
      });
    });
  }

  function copyWithSelection(text) {
    const activeElement = document.activeElement;
    const selection = window.getSelection();
    const ranges = selection ? Array.from({ length: selection.rangeCount }, (_, index) => selection.getRangeAt(index).cloneRange()) : [];
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.setAttribute("aria-label", "Citation to copy");
    textarea.style.cssText = "position:fixed;top:0;left:-9999px;opacity:0;";
    document.body.appendChild(textarea);
    textarea.select();
    let copied = false;
    try {
      copied = document.execCommand("copy");
    } finally {
      textarea.remove();
      if (activeElement && typeof activeElement.focus === "function") activeElement.focus({ preventScroll: true });
      if (selection) {
        selection.removeAllRanges();
        ranges.forEach((range) => selection.addRange(range));
      }
    }
    return copied;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeResearch, { once: true });
  } else {
    initializeResearch();
  }
})();
