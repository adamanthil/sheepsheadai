// Minimal standalone runtime for the decision-review prototypes, so each
// page opens straight from disk. The pages were authored on a design canvas
// whose templating is reproduced here: {{path}} interpolation, <sc-for>,
// <sc-if>, onClick bindings, and a Component class whose renderVals()
// returns the values the template reads.
(function () {
  'use strict';

  function lookup(scope, path) {
    return path.split('.').reduce(function (value, key) {
      return value == null ? undefined : value[key];
    }, scope);
  }

  function whole(text) {
    var match = /^\s*\{\{\s*([\w.$]+)\s*\}\}\s*$/.exec(text);
    return match ? match[1] : null;
  }

  function fill(text, scope) {
    return text.replace(/\{\{\s*([\w.$]+)\s*\}\}/g, function (_, path) {
      var value = lookup(scope, path);
      return value == null ? '' : String(value);
    });
  }

  function extend(scope, name, value) {
    var child = Object.create(scope);
    child[name] = value;
    return child;
  }

  function renderChildren(node, scope, into) {
    node.childNodes.forEach(function (child) { render(child, scope, into); });
  }

  function render(node, scope, into) {
    if (node.nodeType === Node.TEXT_NODE) {
      into.appendChild(document.createTextNode(fill(node.nodeValue, scope)));
      return;
    }
    if (node.nodeType !== Node.ELEMENT_NODE) return;
    var tag = node.localName;
    if (tag === 'helmet') return;
    if (tag === 'sc-for') {
      var list = lookup(scope, whole(node.getAttribute('list'))) || [];
      list.forEach(function (item) {
        renderChildren(node, extend(scope, node.getAttribute('as'), item), into);
      });
      return;
    }
    if (tag === 'sc-if') {
      if (lookup(scope, whole(node.getAttribute('value')))) renderChildren(node, scope, into);
      return;
    }
    var copy = document.createElementNS(node.namespaceURI, tag);
    Array.prototype.forEach.call(node.attributes, function (attr) {
      if (attr.name === 'onclick') {
        var handler = lookup(scope, whole(attr.value));
        if (typeof handler === 'function') copy.addEventListener('click', handler);
        return;
      }
      copy.setAttribute(attr.name, fill(attr.value, scope));
    });
    renderChildren(node, scope, copy);
    into.appendChild(copy);
  }

  document.addEventListener('DOMContentLoaded', function () {
    var template = document.querySelector('x-dc');
    var source = document.querySelector('script[data-dc-script]');
    var helmet = template.querySelector('helmet');
    if (helmet) while (helmet.firstChild) document.head.appendChild(helmet.firstChild);
    template.remove();

    var root = document.createElement('div');
    document.body.appendChild(root);

    function DCLogic() { this.state = {}; }
    DCLogic.prototype.setState = function (patch) {
      Object.assign(this.state, patch);
      paint();
    };
    var Component = new Function('DCLogic', source.textContent + '\nreturn Component;')(DCLogic);
    var component = new Component();

    function paint() {
      var next = document.createDocumentFragment();
      renderChildren(template, component.renderVals(), next);
      root.replaceChildren(next);
    }
    paint();
  });
})();
