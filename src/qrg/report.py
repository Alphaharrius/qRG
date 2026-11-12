"""Self-contained interactive HTML report for a renormalization tree."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .tree import OPERATOR, Node

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --paper: #f7f3ea;
    --ink: #243447;
    --muted: #5f6c7b;
    --line: #6b7a8f;
    --border: #e4dccb;
    --card: #fffdf8;
    --gold: #edc948;
    --gold-edge: #8c6d1f;
    --blue: #d6ebf5;
    --blue-edge: #355070;
    --red: #d1495b;
    --red-edge: #7f1d1d;
    --badge-bg: rgba(36, 52, 71, 0.08);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--paper);
    color: var(--ink);
    font: 14.5px/1.5 "Segoe UI", system-ui, -apple-system, sans-serif;
  }
  main {
    display: grid;
    grid-template-columns: minmax(380px, 1.15fr) minmax(420px, 1fr);
    min-height: 100vh;
    align-items: start;
  }
  @media (max-width: 980px) {
    main { grid-template-columns: 1fr; }
    #tree { position: static; max-height: none; }
  }
  #tree {
    min-width: 0;
    align-self: start;
    position: sticky;
    top: 0;
    max-height: 100vh;
    padding: 24px 20px 72px;
    overflow: auto;
  }
  #record {
    padding: 28px 36px 56px;
    background: var(--card);
    border-left: 1px solid var(--border);
    min-width: 0;
  }
  h1 { font-size: 18px; margin: 0 0 6px; letter-spacing: -0.01em; }
  .subtitle { color: var(--muted); font-size: 12.5px; margin: 0 0 18px; }
  h2 { font-size: 21px; margin: 0 0 4px; }
  .path-line {
    font-family: ui-monospace, Consolas, monospace;
    font-size: 12.5px;
    color: var(--muted);
    margin-bottom: 18px;
  }
  h3 {
    font-size: 14.5px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
    margin: 26px 0 8px;
  }
  .node {
    display: inline-block;
    min-width: 155px;
    max-width: 280px;
    margin: 0;
    padding: 8px 12px;
    font: inherit;
    text-align: left;
    overflow-wrap: anywhere;
    vertical-align: top;
    border-radius: 10px;
    border: 2px solid var(--blue-edge);
    background: var(--blue);
    color: var(--ink);
    cursor: pointer;
    transition: transform 0.08s ease, box-shadow 0.08s ease;
  }
  .node:hover { transform: translateY(-1px); box-shadow: 0 3px 8px rgba(36, 52, 71, 0.12); }
  .node.root-node { background: var(--gold); border-color: var(--gold-edge); color: #1f1a0a; }
  .node.frozen-node { background: #e8eef4; border-style: dashed; }
  .node.error-node {
    background: #fff1f2;
    border-color: #be123c;
    border-left-width: 6px;
    color: #881337;
  }
  .node.error-node small { color: #9f1239; }
  .node.error-node .pill { background: #ffe4e6; color: #9f1239; font-weight: 700; }
  .node .title { display: block; font-weight: 700; font-size: 13.5px; }
  .node small {
    display: block;
    margin-top: 2px;
    font-weight: 500;
    font-size: 11.5px;
    font-family: ui-monospace, Consolas, monospace;
    color: #355070;
  }
  .node .pill {
    display: inline-block;
    margin-top: 4px;
    padding: 1px 6px;
    border-radius: 999px;
    font-size: 11px;
    font-family: ui-monospace, Consolas, monospace;
    background: var(--badge-bg);
    color: var(--ink);
  }
  .node.selected {
    background: var(--red);
    border-color: var(--red-edge);
    border-style: solid;
    color: white;
  }
  .node.selected small { color: #ffe4e6; }
  .node.selected .pill { background: rgba(255, 255, 255, 0.22); color: white; }
  .edge {
    margin: 6px 0 6px 16px;
    padding: 2px 0 2px 12px;
    border-left: 2px solid var(--line);
    color: var(--line);
    font-family: ui-monospace, Consolas, monospace;
    font-size: 12px;
    max-width: 280px;
    overflow-wrap: anywhere;
  }
  .branch { margin-left: 6px; }
  .fork-group { margin: 4px 0 4px 4px; }
  .fork-columns {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    margin-left: 12px;
    padding-top: 6px;
    border-top: 2px solid var(--line);
    width: max-content;
  }
  .fork-col { flex: 0 0 auto; min-width: 168px; }
  .fork-col > .edge { margin-left: 6px; margin-top: 0; }
  .table-wrap {
    overflow-x: auto;
    margin: 6px 0 16px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: white;
  }
  table { border-collapse: collapse; width: 100%; }
  th, td {
    border-bottom: 1px solid #efe9dc;
    text-align: left;
    padding: 6px 10px;
    font-family: ui-monospace, Consolas, monospace;
    font-size: 12.5px;
    vertical-align: top;
  }
  tr:last-child th, tr:last-child td { border-bottom: none; }
  thead th {
    background: #f5f0e5;
    color: var(--ink);
    font-weight: 650;
    border-bottom: 1px solid var(--border);
  }
  tbody tr:nth-child(even) td { background: #fcfaf5; }
  th.kv-key { width: 200px; color: var(--muted); font-weight: 600; background: #faf7f0; }
  .plot-card {
    margin: 14px 0 22px;
    padding: 10px 12px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: white;
  }
  .plot-box { width: 100%; min-height: 430px; }
  .plot-box.plot-3d { min-height: 520px; }
  .error-card {
    margin: 10px 0 20px;
    padding: 14px 16px;
    border: 2px solid #be123c;
    border-left-width: 6px;
    border-radius: 10px;
    background: #fff1f2;
    color: #881337;
  }
  .error-card .error-head {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 750;
    font-size: 14.5px;
    margin-bottom: 6px;
  }
  .error-card .error-tag {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 999px;
    background: #be123c;
    color: white;
    font-size: 11.5px;
    font-family: ui-monospace, Consolas, monospace;
    letter-spacing: 0.03em;
  }
  .error-card .error-msg {
    font-family: ui-monospace, Consolas, monospace;
    font-size: 13px;
    background: white;
    border: 1px solid #fecdd3;
    border-radius: 6px;
    padding: 8px 10px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .error-trace {
    margin: 10px 0 0;
    padding: 10px 12px;
    border-radius: 6px;
    background: #1e293b;
    color: #f8fafc;
    font-family: ui-monospace, Consolas, monospace;
    font-size: 12px;
    line-height: 1.45;
    overflow-x: auto;
    max-height: 360px;
    overflow-y: auto;
  }
  img {
    max-width: 100%;
    margin-top: 12px;
    border-radius: 8px;
    background: white;
    border: 1px solid var(--border);
  }
  .empty { color: var(--muted); }
</style>
</head>
<body>
<main>
  <section id="tree">
    <h1>__TITLE__</h1>
    <p class="subtitle">
      Links show RG steps; blocks show operators, tables, and interactive plots.
      Crimson marks the active block.
    </p>
    <div id="nodes"></div>
  </section>
  <section id="record"></section>
</main>
<script>__PLOTLY__</script>
<script id="tree-data" type="application/json">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById("tree-data").textContent);
const byId = Object.fromEntries(data.nodes.map((node) => [node.id, node]));
const children = {};
for (const node of data.nodes) {
  (children[node.parent] ||= []).push(node);
}
const host = document.getElementById("nodes");
const record = document.getElementById("record");
let selected = byId[data.selected] ? data.selected : data.nodes[0].id;

function formatScalar(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    const abs = Math.abs(value);
    if (abs !== 0 && (abs < 1e-4 || abs >= 1e5)) return value.toExponential(4);
    return Number(value.toPrecision(6)).toString();
  }
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    if (value.every((item) => item === null || typeof item !== "object")) {
      return "(" + value.map(formatScalar).join(", ") + ")";
    }
    return JSON.stringify(value);
  }
  if (typeof value === "object") {
    return Object.entries(value)
      .map(([k, v]) => k + ": " + formatScalar(v))
      .join(", ");
  }
  return String(value);
}

function buttonFor(node) {
  const button = document.createElement("button");
  let cls = "node";
  if (node.parent === null) cls += " root-node";
  if (node.role === "frozen" || node.role === "frozen_sibling") cls += " frozen-node";
  if (node.error || node.role === "error") cls += " error-node";
  if (node.id === selected) cls += " selected";
  button.className = cls;
  button.type = "button";

  const title = document.createElement("span");
  title.className = "title";
  title.textContent = node.label || node.name;
  button.appendChild(title);

  if (node.tensors && node.tensors.length) {
    const shapes = document.createElement("small");
    shapes.textContent = node.tensors
      .map((item) => item.key + " " + item.shape.join("×"))
      .join(" · ");
    button.appendChild(shapes);
  } else if (node.error && node.error.message) {
    const errPreview = document.createElement("small");
    const msg = String(node.error.message);
    errPreview.textContent = msg.length > 68 ? msg.slice(0, 65) + "..." : msg;
    button.appendChild(errPreview);
  }
  if (node.badge) {
    const pill = document.createElement("span");
    pill.className = "pill";
    pill.textContent = node.badge;
    button.appendChild(pill);
  }
  button.addEventListener("click", () => {
    selected = node.id;
    host.replaceChildren();
    draw(null, host);
    show(node);
  });
  return button;
}

function draw(parentId, container) {
  const kids = children[parentId] || [];
  for (const node of kids) {
    if (node.role === "frozen_sibling") continue;
    if (node.frozen_id && byId[node.frozen_id]) {
      const frozenNode = byId[node.frozen_id];
      const group = document.createElement("div");
      group.className = "fork-group";
      if (node.step_edge) {
        const headerEdge = document.createElement("div");
        headerEdge.className = "edge";
        headerEdge.textContent = node.step_edge;
        group.appendChild(headerEdge);
      }
      const cols = document.createElement("div");
      cols.className = "fork-columns";

      const leftCol = document.createElement("div");
      leftCol.className = "fork-col";
      const leftEdge = document.createElement("div");
      leftEdge.className = "edge";
      leftEdge.textContent = frozenNode.edge || "frozen isometry";
      leftCol.append(leftEdge, buttonFor(frozenNode));

      const rightCol = document.createElement("div");
      rightCol.className = "fork-col";
      const rightEdge = document.createElement("div");
      rightEdge.className = "edge";
      rightEdge.textContent = node.edge || "courier isometry";
      rightCol.append(rightEdge, buttonFor(node));
      const nested = document.createElement("div");
      nested.className = "branch";
      rightCol.appendChild(nested);
      draw(node.id, nested);

      cols.append(leftCol, rightCol);
      group.appendChild(cols);
      container.appendChild(group);
      continue;
    }

    if (node.edge) {
      const edge = document.createElement("div");
      edge.className = "edge";
      edge.textContent = node.edge;
      container.appendChild(edge);
    }
    container.appendChild(buttonFor(node));
    const nested = document.createElement("div");
    nested.className = "branch";
    container.appendChild(nested);
    draw(node.id, nested);
  }
}

function kvTable(entries) {
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  const tbl = document.createElement("table");
  const tbody = document.createElement("tbody");
  for (const [key, val] of entries) {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.className = "kv-key";
    th.textContent = key;
    const td = document.createElement("td");
    td.textContent = formatScalar(val);
    tr.append(th, td);
    tbody.appendChild(tr);
  }
  tbl.appendChild(tbody);
  wrap.appendChild(tbl);
  return wrap;
}

function isRowTable(val) {
  return (
    Array.isArray(val) &&
    val.length > 0 &&
    val.every((row) => row && typeof row === "object" && !Array.isArray(row))
  );
}

function isDictOfDicts(val) {
  if (!val || typeof val !== "object" || Array.isArray(val)) return false;
  const entries = Object.values(val);
  return (
    entries.length > 0 &&
    entries.every((row) => row && typeof row === "object" && !Array.isArray(row) && !row.image)
  );
}

function matrixTable(rows, rowHeaderName) {
  const cols = [];
  for (const [, rowObj] of rows) {
    for (const k of Object.keys(rowObj)) {
      if (!cols.includes(k)) cols.push(k);
    }
  }
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  const tbl = document.createElement("table");
  const thead = document.createElement("thead");
  const htr = document.createElement("tr");
  if (rowHeaderName !== null) {
    const th0 = document.createElement("th");
    th0.textContent = rowHeaderName;
    htr.appendChild(th0);
  }
  for (const col of cols) {
    const th = document.createElement("th");
    th.textContent = col;
    htr.appendChild(th);
  }
  thead.appendChild(htr);
  tbl.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const [rowKey, rowObj] of rows) {
    const tr = document.createElement("tr");
    if (rowHeaderName !== null) {
      const th = document.createElement("th");
      th.textContent = rowKey;
      tr.appendChild(th);
    }
    for (const col of cols) {
      const td = document.createElement("td");
      td.textContent = col in rowObj ? formatScalar(rowObj[col]) : "—";
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  tbl.appendChild(tbody);
  wrap.appendChild(tbl);
  return wrap;
}

function addSection(titleText, element) {
  const h3 = document.createElement("h3");
  h3.textContent = titleText;
  record.append(h3, element);
}

function has3DTrace(spec) {
  return (spec.data || []).some((t) =>
    ["scatter3d", "surface", "mesh3d", "cone", "isosurface"].includes(t.type)
  );
}

function show(node) {
  record.replaceChildren();
  const title = document.createElement("h2");
  title.textContent = node.label || node.name;
  const pathLine = document.createElement("div");
  pathLine.className = "path-line";
  pathLine.textContent = "path: " + (node.path || node.name) + " · depth " + node.depth;
  record.append(title, pathLine);

  if (node.error) {
    const card = document.createElement("div");
    card.className = "error-card";
    const head = document.createElement("div");
    head.className = "error-head";
    const tag = document.createElement("span");
    tag.className = "error-tag";
    tag.textContent = "ERROR";
    const errType = document.createElement("span");
    const stageSuffix = node.error.stage ? "  (stage: " + node.error.stage + ")" : "";
    errType.textContent = (node.error.type || "Exception") + stageSuffix;
    head.append(tag, errType);
    card.appendChild(head);

    const msg = document.createElement("div");
    msg.className = "error-msg";
    msg.textContent = node.error.message || "Unknown error";
    card.appendChild(msg);

    if (node.error.traceback) {
      const pre = document.createElement("pre");
      pre.className = "error-trace";
      pre.textContent = node.error.traceback;
      card.appendChild(pre);
    }
    record.appendChild(card);
  }

  const overview = [];
  if (node.step) {
    overview.push(["step", node.step.method + " · " + node.step.impl]);
    overview.push(["kind", node.step.kind]);
    overview.push(["parameters", node.step.params]);
  }
  for (const item of node.tensors || []) {
    const spaces = item.dims && item.dims.length ? "  (" + item.dims.join(", ") + ")" : "";
    overview.push(["tensor " + item.key, item.shape.join(" × ") + spaces]);
  }
  if (overview.length) {
    addSection("Block & Tensors", kvTable(overview));
  }

  const scalars = [];
  const tables = [];
  const images = [];
  for (const [key, value] of Object.entries(node.info || {})) {
    if (value && typeof value === "object" && value.image) {
      images.push([key, value]);
    } else if (isRowTable(value)) {
      tables.push([key, matrixTable(value.map((r, i) => [String(i), r]), null)]);
    } else if (isDictOfDicts(value)) {
      tables.push([key, matrixTable(Object.entries(value), "key")]);
    } else if (value && typeof value === "object" && !Array.isArray(value)) {
      tables.push([key, kvTable(Object.entries(value))]);
    } else {
      scalars.push([key, value]);
    }
  }

  if (scalars.length) {
    addSection("Recorded Summary", kvTable(scalars));
  }
  for (const [name, tblElem] of tables) {
    addSection(name.replace(/_/g, " "), tblElem);
  }

  for (const [name, value] of images) {
    const h3 = document.createElement("h3");
    h3.textContent = name.replace(/_/g, " ");
    const image = document.createElement("img");
    image.src = value.image;
    image.alt = value.path || name;
    record.append(h3, image);
  }

  if (node.plots && node.plots.length && window.Plotly) {
    for (const item of node.plots) {
      const h3 = document.createElement("h3");
      h3.textContent = item.title;
      const card = document.createElement("div");
      card.className = "plot-card";
      const box = document.createElement("div");
      box.className = "plot-box" + (has3DTrace(item.spec) ? " plot-3d" : "");
      card.appendChild(box);
      record.append(h3, card);

      const cloned = structuredClone(item.spec);
      const layout = Object.assign(
        {
          paper_bgcolor: "#ffffff",
          plot_bgcolor: "#ffffff",
          margin: { t: 46, r: 20, l: 58, b: 48 },
        },
        cloned.layout || {}
      );
      delete layout.width;
      layout.autosize = true;
      Plotly.newPlot(box, cloned.data || [], layout, { responsive: true });
    }
  }

  if (
    !node.error &&
    !overview.length &&
    !scalars.length &&
    !tables.length &&
    !images.length &&
    !(node.plots && node.plots.length)
  ) {
    const note = document.createElement("p");
    note.className = "empty";
    note.textContent = "No record on this node.";
    record.appendChild(note);
  }
}

draw(null, host);
show(byId[selected]);
</script>
</body>
</html>
"""


def write_report(
    node: Node,
    path: str | Path,
    *,
    selected: Node | str | None = None,
) -> Path:
    """
    Write a self-contained HTML view of the tree rooted at `node`.

    Links show the step that produced each layer, Wannier layers fork into
    ``frozen`` and ``courier`` branches, numerical records are rendered as
    tables, and Plotly figures (both 2D and 3D) are embedded interactively.
    """
    root = node.root()
    nodes = list(_nodes(root))
    if isinstance(selected, Node):
        chosen = _node_id(selected)
    elif isinstance(selected, str) and selected:
        chosen = selected
    else:
        err_entry = next((item["id"] for item in nodes if item.get("error")), None)
        chosen = err_entry or (_latest_leaf(root) if node is root else _node_id(node))
    payload = {"nodes": nodes, "selected": chosen}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload).replace("</", "<\\/")
    title = f"qRG · {root.name}"
    has_plots = any(bool(item.get("plots")) for item in nodes)
    plotly = _plotly_js() if has_plots else ""
    destination.write_text(
        _PAGE.replace("__TITLE__", title).replace("__DATA__", data).replace("__PLOTLY__", plotly),
        encoding="utf-8",
    )
    root._remember(
        {
            "kind": "export",
            "plot": "report",
            "backend": "html",
            "file": str(destination),
            "path": root.path(),
            "depth": root.depth(),
        }
    )
    return destination


def _nodes(root: Node) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def visit(node: Node) -> None:
        step = node.step
        node_id = _node_id(node)
        parent_id = None if node.is_root() else _node_id(node.parent())
        is_wannier_split = "frozen" in node and "courier" in node

        raw_info = dict(node.info)
        error_obj = _normalize_error(raw_info.pop("error", None), step)
        frozen_raw = raw_info.pop("frozen", None) if is_wannier_split else None
        courier_raw = raw_info.pop("courier", None) if is_wannier_split else None

        if is_wannier_split and isinstance(courier_raw, Mapping):
            merged_courier: dict[str, Any] = {**dict(courier_raw), **raw_info}
        else:
            merged_courier = raw_info

        clean_info, plots = _extract_info_and_plots(merged_courier)
        if not plots and error_obj is None and _is_hamiltonian(node):
            fallback = _default_hamiltonian_plot(node.operator(), title=f"{node.name} bands")
            if fallback is not None:
                plots.append(fallback)

        step_dict = (
            None
            if step is None
            else {
                "method": step.method,
                "impl": step.impl,
                "kind": step.kind,
                "params": _plain(dict(step.params)),
            }
        )

        if is_wannier_split:
            frozen_id = f"{node_id}/frozen"
            found.append(_frozen_branch_entry(node, parent_id, frozen_id, frozen_raw))
            courier_tensors = [
                item for item in _tensors(node) if item["key"] in {OPERATOR, "courier"}
            ]
            c_label = f"ERROR · courier ({node.name})" if error_obj else f"courier ({node.name})"
            found.append(
                {
                    "id": node_id,
                    "name": node.name,
                    "label": c_label,
                    "path": node.path() or node.name,
                    "depth": node.depth(),
                    "parent": parent_id,
                    "role": "error" if error_obj else "courier",
                    "frozen_id": frozen_id,
                    "step_edge": None if step is None else _edge_text(step),
                    "edge": _sub_edge("courier", step),
                    "step": step_dict,
                    "tensors": courier_tensors,
                    "badge": _badge(clean_info, error_obj),
                    "error": error_obj,
                    "info": clean_info,
                    "plots": plots,
                }
            )
        else:
            s_label = f"ERROR ({node.name})" if error_obj else node.name
            found.append(
                {
                    "id": node_id,
                    "name": node.name,
                    "label": s_label,
                    "path": node.path() or node.name,
                    "depth": node.depth(),
                    "parent": parent_id,
                    "role": "error" if error_obj else "standard",
                    "frozen_id": None,
                    "step_edge": None,
                    "edge": None if step is None else _edge_text(step),
                    "step": step_dict,
                    "tensors": _tensors(node),
                    "badge": _badge(clean_info, error_obj),
                    "error": error_obj,
                    "info": clean_info,
                    "plots": plots,
                }
            )

        for child in node.children.values():
            visit(child)

    visit(root)
    return found


def _frozen_branch_entry(
    node: Node,
    parent_id: str | None,
    frozen_id: str,
    frozen_raw: Any,
) -> dict[str, Any]:
    step = node.step
    mapping = dict(frozen_raw) if isinstance(frozen_raw, Mapping) else {}
    if frozen_raw is not None and not isinstance(frozen_raw, Mapping):
        mapping["value"] = frozen_raw
    clean_info, plots = _extract_info_and_plots(mapping)

    frozen_op = _frozen_operator(node)
    tensors: list[dict[str, Any]] = []
    if frozen_op is not None:
        op_shape = _shape(frozen_op)
        if op_shape is not None:
            tensors.append({"key": "H_frozen", "shape": op_shape, "dims": _dim_names(frozen_op)})
    tensors.append(
        {
            "key": "W_frozen",
            "shape": _shape(node["frozen"]) or [],
            "dims": _dim_names(node["frozen"]),
        }
    )

    if not plots and frozen_op is not None:
        fallback = _default_hamiltonian_plot(frozen_op, title="Frozen Wannier bands")
        if fallback is not None:
            plots.append(fallback)

    return {
        "id": frozen_id,
        "name": "frozen",
        "label": f"frozen ({node.name})",
        "path": f"{node.path() or node.name}/frozen",
        "depth": node.depth(),
        "parent": parent_id,
        "role": "frozen_sibling",
        "frozen_id": None,
        "step_edge": None,
        "edge": _sub_edge("frozen", step),
        "step": None
        if step is None
        else {
            "method": step.method,
            "impl": step.impl,
            "kind": "frozen subspace",
            "params": _plain(dict(step.params)),
        },
        "tensors": tensors,
        "badge": _badge(clean_info, None),
        "error": None,
        "info": clean_info,
        "plots": plots,
    }


def _normalize_error(raw_err: Any, step: Any) -> dict[str, Any] | None:
    if isinstance(raw_err, Mapping):
        return {
            "type": str(raw_err.get("type", "Exception")),
            "message": str(raw_err.get("message", "")),
            "stage": str(raw_err["stage"]) if raw_err.get("stage") else None,
            "traceback": str(raw_err["traceback"]) if raw_err.get("traceback") else None,
        }
    if raw_err is not None:
        return {
            "type": "Error",
            "message": str(raw_err),
            "stage": None,
            "traceback": None,
        }
    if step is not None and getattr(step, "kind", None) == "error":
        return {
            "type": "StepError",
            "message": f"Step {step.method!r} ({step.impl}) failed.",
            "stage": None,
            "traceback": None,
        }
    return None


def _sub_edge(role: str, step: Any) -> str:
    if step is not None:
        recipe = step.params.get(role)
        if isinstance(recipe, str) and recipe:
            return f"{role} isometry ({recipe})"
    return f"{role} isometry"


def _badge(info: Mapping[str, Any], error_obj: Mapping[str, Any] | None = None) -> str | None:
    if error_obj is not None:
        err_type = error_obj.get("type") or "Exception"
        return f"ERROR · {err_type}"
    parts: list[str] = []
    for key in ("z2", "Z2", "z2_indices"):
        if key in info:
            val = info[key]
            if isinstance(val, list):
                parts.append("Z2 (" + ",".join(str(x) for x in val) + ")")
            else:
                parts.append(f"Z2 {val}")
            break
    gap = info.get("direct_gap")
    if isinstance(gap, (int, float)):
        parts.append(f"gap {gap:.3f}")
    return " · ".join(parts) if parts else None


def _extract_info_and_plots(
    raw: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clean: dict[str, Any] = {}
    plots: list[dict[str, Any]] = []
    for key, value in raw.items():
        spec = _as_plotly_spec(value)
        if spec is not None:
            plots.append({"title": str(key), "spec": spec})
            continue
        if key == "plots" and isinstance(value, Mapping):
            for plot_title, plot_obj in value.items():
                sub_spec = _as_plotly_spec(plot_obj)
                if sub_spec is not None:
                    plots.append({"title": str(plot_title), "spec": sub_spec})
            continue
        clean[str(key)] = _plain(value)
    return clean, plots


def _as_plotly_spec(value: Any) -> dict[str, Any] | None:
    to_json = getattr(value, "to_json", None)
    to_plotly_json = getattr(value, "to_plotly_json", None)
    if callable(to_json) and callable(to_plotly_json):
        loaded = json.loads(str(to_json()))
        if isinstance(loaded, dict):
            return loaded
    if isinstance(value, Mapping) and "__plotly__" in value:
        inner = value["__plotly__"]
        if isinstance(inner, dict):
            return inner
    return None


def _node_id(node: Node) -> str:
    return node.path() or node.name


def _latest_leaf(root: Node) -> str:
    found = root
    while found.children:
        found = list(found.children.values())[-1]
    return _node_id(found)


def _edge_text(step: Any) -> str:
    pieces = [f"{key} {value}" for key, value in step.params.items() if _short(value)]
    detail = ", ".join(pieces)
    if detail:
        return f"{step.impl}  {detail}"
    return str(step.impl)


def _short(value: Any) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(value, bool)


def _tensors(node: Node) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key in node:
        tensor = node[key]
        shape = _shape(tensor)
        items.append(
            {
                "key": key,
                "shape": shape if shape is not None else [],
                "dims": _dim_names(tensor),
            }
        )
    return items


def _dim_names(value: Any) -> list[str]:
    dims = getattr(value, "dims", None)
    if not isinstance(dims, tuple):
        return []
    return [type(dim).__name__ for dim in dims]


def _shape(value: Any) -> list[int] | None:
    data = getattr(value, "data", None)
    shape = getattr(data, "shape", None)
    if shape is None:
        return None
    return [int(size) for size in shape]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return _file(value)
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and candidate.is_file():
            return _file(candidate)
        return value
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    return str(value)


def _is_hamiltonian(node: Node) -> bool:
    try:
        operator = node.operator()
    except Exception:
        return False
    data = getattr(operator, "data", None)
    return getattr(data, "ndim", None) == 3


def _frozen_operator(node: Node) -> Any:
    if node.is_root() or not _is_hamiltonian(node.parent()):
        return None
    try:
        from .rg import project

        return project(node.parent().operator(), node["frozen"])
    except Exception:
        return None


def _default_hamiltonian_plot(operator: Any, *, title: str) -> dict[str, Any] | None:
    plot_fn = getattr(operator, "plot", None)
    if callable(plot_fn):
        try:
            fig = plot_fn("bandstructure", backend="plotly", show=False, title=title)
            update = getattr(fig, "update_layout", None)
            if callable(update):
                update(showlegend=False)
            spec = _as_plotly_spec(fig)
            if spec is not None:
                return {"title": title, "spec": spec}
        except Exception:
            pass
    return _fallback_line_cut(operator, title=title)


def _fallback_line_cut(operator: Any, *, title: str) -> dict[str, Any] | None:
    data = getattr(operator, "data", None)
    dims = getattr(operator, "dims", None)
    if data is None or getattr(data, "ndim", None) != 3 or not dims:
        return None
    try:
        elements = list(dims[0].elements())
    except Exception:
        return None
    coordinates: list[tuple[int, list[float]]] = []
    for index, momentum in enumerate(elements):
        rep = momentum.rep
        fractional: list[float] = []
        for row in range(int(rep.rows)):
            value = float(rep[row, 0]) % 1.0
            fractional.append(0.0 if value > 1.0 - 1e-8 else value)
        coordinates.append((index, fractional))
    if len(coordinates) < 2:
        return None
    dimension = len(coordinates[0][1])

    def on_axis(axis: int) -> list[tuple[int, list[float]]]:
        kept = []
        for index, fractional in coordinates:
            others = (fractional[i] for i in range(dimension) if i != axis)
            if all(min(value, 1.0 - value) < 1e-6 for value in others):
                kept.append((index, fractional))
        kept.sort(key=lambda item: item[1][axis])
        return kept

    axis = max(range(dimension), key=lambda number: len(on_axis(number)))
    chosen = on_axis(axis)
    if len(chosen) < 2:
        return None
    import qten

    spectrum = qten.linalg.eigvalsh(operator).data.real.detach().cpu()
    values = spectrum[[item[0] for item in chosen]]
    names = ("kx", "ky", "kz")
    xlabel = names[axis] if axis < len(names) else f"k{axis}"
    traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": [item[1][axis] for item in chosen],
            "y": band,
            "line": {"color": "#243447", "width": 1.4},
        }
        for band in values.transpose(0, 1).tolist()
    ]
    return {
        "title": title,
        "spec": {
            "data": traces,
            "layout": {
                "title": {"text": title},
                "xaxis": {"title": {"text": xlabel}},
                "yaxis": {"title": {"text": "Energy"}},
                "showlegend": False,
            },
        },
    }


def _plotly_js() -> str:
    try:
        import plotly
    except ImportError:
        return ""
    script = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
    if not script.is_file():
        return ""
    return script.read_text(encoding="utf-8")


def _file(path: Path) -> dict[str, str]:
    suffix = path.suffix.lower()
    if suffix == ".png":
        media = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        media = "image/jpeg"
    elif suffix == ".webp":
        media = "image/webp"
    else:
        return {"path": str(path)}
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"path": str(path), "image": f"data:{media};base64,{encoded}"}
