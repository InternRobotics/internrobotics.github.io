(() => {
  'use strict';

  // Add real external destinations here. Empty values remain non-interactive text.
  const RESOURCE_LINKS = Object.freeze({
    paper: '',
    code: '',
    data: '',
    models: '',
    video: '',
  });

  const mobileViewport = window.matchMedia('(max-width: 900px)');

  function configureResources() {
    document.querySelectorAll('[data-resource]').forEach((placeholder) => {
      const key = placeholder.dataset.resource;
      const value = RESOURCE_LINKS[key]?.trim();
      if (!value) return;

      let url;
      try {
        url = new URL(value, window.location.href);
      } catch {
        return;
      }
      if (!['http:', 'https:'].includes(url.protocol)) return;

      const link = document.createElement('a');
      link.className = placeholder.className.replace('is-pending', 'is-available');
      link.dataset.resource = key;
      link.href = url.href;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.innerHTML = placeholder.innerHTML;
      const state = link.querySelector('small');
      if (state) state.textContent = 'External link';
      else if (placeholder.closest('.resource-row')) link.textContent = 'Open link';
      placeholder.replaceWith(link);
    });
  }

  function configureNavigation() {
    const toggle = document.querySelector('.menu-toggle');
    const navigation = document.getElementById('mobile-nav');

    function setOpen(open, returnFocus = false) {
      toggle.setAttribute('aria-expanded', String(open));
      toggle.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
      navigation.hidden = !open;
      if (returnFocus) toggle.focus({ preventScroll: true });
    }

    setOpen(false);
    toggle.addEventListener('click', () => {
      setOpen(toggle.getAttribute('aria-expanded') !== 'true');
    });
    navigation.addEventListener('click', (event) => {
      if (event.target.closest('a')) setOpen(false);
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && toggle.getAttribute('aria-expanded') === 'true') {
        event.preventDefault();
        setOpen(false, true);
      }
    });
    mobileViewport.addEventListener('change', (event) => {
      if (!event.matches) setOpen(false);
    });
  }

  function configureResultTabs() {
    const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
    if (!tabs.length) return;

    function selectTab(selected, moveFocus = false) {
      tabs.forEach((tab) => {
        const active = tab === selected;
        tab.setAttribute('aria-selected', String(active));
        tab.tabIndex = active ? 0 : -1;
        document.getElementById(tab.getAttribute('aria-controls')).hidden = !active;
      });
      if (moveFocus) selected.focus({ preventScroll: true });
    }

    tabs.forEach((tab) => tab.addEventListener('click', () => selectTab(tab)));
    document.querySelector('[role="tablist"]').addEventListener('keydown', (event) => {
      const current = tabs.indexOf(document.activeElement);
      if (current < 0) return;
      let target = null;
      if (event.key === 'ArrowRight') target = (current + 1) % tabs.length;
      if (event.key === 'ArrowLeft') target = (current - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') target = 0;
      if (event.key === 'End') target = tabs.length - 1;
      if (target === null) return;
      event.preventDefault();
      selectTab(tabs[target], true);
    });
    selectTab(tabs[0]);
  }

  function configureFigureViewer() {
    const dialog = document.getElementById('figure-dialog');
    const viewport = document.getElementById('figure-viewport');
    const zoom = document.getElementById('figure-zoom');
    const close = dialog.querySelector('.dialog-close');
    let trigger = null;

    function setZoomed(zoomed) {
      viewport.classList.toggle('is-zoomed', zoomed);
      zoom.setAttribute('aria-pressed', String(zoomed));
      zoom.textContent = zoomed ? 'Fit to window' : 'Original size';
    }

    document.querySelectorAll('[data-figure]').forEach((link) => {
      link.addEventListener('click', (event) => {
        event.preventDefault();
        if (dialog.open) return;
        trigger = link;
        document.getElementById('dialog-title').textContent = link.dataset.title;
        document.getElementById('dialog-caption').textContent = link.dataset.caption;
        document.getElementById('figure-source').href = link.dataset.source;
        const image = document.createElement('img');
        image.src = link.getAttribute('href');
        image.alt = link.querySelector('img').alt;
        viewport.replaceChildren(image);
        setZoomed(false);
        dialog.showModal();
        document.body.classList.add('dialog-open');
        close.focus();
      });
    });

    close.addEventListener('click', () => dialog.close());
    zoom.addEventListener('click', () => {
      setZoomed(!viewport.classList.contains('is-zoomed'));
      viewport.scrollTo({ top: 0, left: 0, behavior: 'auto' });
    });
    dialog.addEventListener('click', (event) => {
      if (event.target !== dialog) return;
      const bounds = dialog.getBoundingClientRect();
      const outside = event.clientX < bounds.left || event.clientX > bounds.right
        || event.clientY < bounds.top || event.clientY > bounds.bottom;
      if (outside) dialog.close();
    });
    dialog.addEventListener('close', () => {
      document.body.classList.remove('dialog-open');
      setZoomed(false);
      trigger?.focus({ preventScroll: true });
    });
  }

  configureResources();
  configureNavigation();
  configureResultTabs();
  configureFigureViewer();
  document.documentElement.classList.add('js');
})();
