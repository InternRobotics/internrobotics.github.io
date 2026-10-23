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

  const VIDEO_SEGMENTS = Object.freeze({
    nesting: {
      duration: 18,
      segments: [
        { skill: 'Pick', action: 'Picks up the first matryoshka doll.', start: 0, end: 2.0 },
        { skill: 'Place', action: 'Places the first matryoshka doll on the table.', start: 2.04, end: 3.48 },
        { skill: 'Pick', action: 'Picks up the second matryoshka doll.', start: 3.52, end: 5.16 },
        { skill: 'Place', action: 'Places the second matryoshka doll on the table.', start: 5.2, end: 6.96 },
        { skill: 'Pick', action: 'Picks up the third matryoshka doll.', start: 7.0, end: 8.76 },
        { skill: 'Place', action: 'Places the third matryoshka doll on the table.', start: 8.8, end: 10.8 },
        { skill: 'Pick', action: 'Picks up the fourth matryoshka doll.', start: 10.84, end: 12.52 },
        { skill: 'Place', action: 'Places the fourth matryoshka doll on the table.', start: 12.56, end: 13.8 },
        { skill: 'Pick', action: 'Picks up the fifth matryoshka doll.', start: 13.84, end: 15.64 },
        { skill: 'Place', action: 'Places the fifth matryoshka doll on the table.', start: 15.68, end: 17.96 },
      ],
    },
    sandwich: {
      duration: 53,
      segments: [
        { skill: 'Pick', action: 'Picks up the first bread slice and the meat patty.', start: 0, end: 12.6 },
        { skill: 'Place', action: 'Places the first bread slice onto the plate.', start: 12.67, end: 22.47 },
        { skill: 'Place', action: 'Places the meat patty on the first bread slice.', start: 22.53, end: 27.4 },
        { skill: 'Pick', action: 'Picks up the lettuce leaf.', start: 27.47, end: 30.93 },
        { skill: 'Place', action: 'Places the lettuce leaf on the meat patty.', start: 31.0, end: 37.93 },
        { skill: 'Pick', action: 'Picks up the second bread slice.', start: 38.0, end: 42.87 },
        { skill: 'Place', action: 'Places the second bread slice on top of the lettuce leaf.', start: 42.93, end: 52.87 },
      ],
    },
    dishwasher: {
      duration: 35,
      segments: [
        { skill: 'Open', action: 'Opens the dishwasher door.', start: 0, end: 26.27 },
        { skill: 'Move', action: 'Moves to the dishwasher.', start: 26.33, end: 29.13 },
        { skill: 'Place', action: 'Places the dark bowl into the dishwasher.', start: 29.2, end: 34.87 },
      ],
    },
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

  function configureVideoCards() {
    document.querySelectorAll('[data-video-card]').forEach((card) => {
      const config = VIDEO_SEGMENTS[card.dataset.video];
      if (!config) return;
      const video = card.querySelector('video');
      video.preload = 'auto';
      video.load();
      const timeline = card.querySelector('[data-timeline]');
      const activeSkill = card.querySelector('[data-active-skill]');
      const activeAction = card.querySelector('[data-active-action]');
      const buttons = config.segments.map((segment, index) => {
        const button = document.createElement('button');
        const start = (segment.start / config.duration) * 100;
        const width = ((segment.end - segment.start) / config.duration) * 100;
        button.type = 'button';
        button.className = 'timeline-segment';
        button.dataset.skill = segment.skill;
        button.style.setProperty('--segment-start', `${start}%`);
        button.style.setProperty('--segment-width', `${width}%`);
        button.setAttribute('aria-pressed', 'false');
        button.setAttribute('aria-label', `${segment.skill}: ${segment.action} Jump to ${segment.start.toFixed(1)} seconds.`);
        button.title = `${segment.skill}: ${segment.action}`;
        button.addEventListener('click', () => {
          video.currentTime = Math.min(segment.start + 0.01, Math.max(0, config.duration - 0.05));
          updateActiveSegment();
        });
        button.dataset.index = String(index);
        return button;
      });
      timeline.replaceChildren(...buttons);
      card.classList.add('has-timeline');

      const updateActiveSegment = () => {
        const current = video.currentTime;
        let activeIndex = config.segments.findIndex((segment) => current < segment.end);
        if (activeIndex < 0) activeIndex = config.segments.length - 1;
        const active = config.segments[activeIndex];
        activeSkill.textContent = active.skill;
        activeAction.textContent = active.action;
        buttons.forEach((button, index) => button.setAttribute('aria-pressed', String(index === activeIndex)));
      };

      video.addEventListener('loadedmetadata', updateActiveSegment);
      video.addEventListener('timeupdate', updateActiveSegment);
      video.addEventListener('seeked', updateActiveSegment);
      updateActiveSegment();
    });
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
  configureVideoCards();
  configureFigureViewer();
  document.documentElement.classList.add('js');
})();
