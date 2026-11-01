import { app } from '../../../scripts/app.js'
import { api } from '../../../scripts/api.js'

const ASPECT_PRESETS = {
    "1:1":  [1024, 1024],
    "16:9": [1824, 1024],
    "9:16": [1024, 1824],
    "4:3":  [1360, 1024],
    "3:4":  [1024, 1360],
    "3:2":  [1536, 1024],
    "2:3":  [1024, 1536],
    "21:9": [2048, 878],
};

const RES_PRESETS = {
    "1K": 1024,
    "2K": 2048,
    "4K": 4096,
};

const COLOURS = [
    "#FF4444","#44AAFF","#44FF88","#FFAA44",
    "#FF44FF","#44FFFF","#FFFF44","#AA44FF",
];

const HANDLE_SIZE = 8; // px, half-size of resize handle hit area

// MOSS layout vocabulary (public Layout draft schema)
const MOSS_LABELS = ["object", "character", "text"];
const MOSS_CANVAS = ["single", "multi_panel", "collage"];
const MOSS_MIN_BOX = 16; // px, smallest box a drag can produce

const clamp  = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const snap16 = (v) => Math.max(16, Math.round(v / 16) * 16);
// Round down to a multiple of 16 (never below 16). Python's _floor16 is the same rule.
const floor16 = (v) => Math.max(16, Math.floor(v / 16) * 16);
// Which of the fixed aspects is a size closest to? Linked sizes are rounded to 8/16/32/64 by whatever
// made them, so they never match exactly; compare on a log scale (ratios are multiplicative) and take the nearest.
const nearestAspect = (w, h) => {
    const r = Math.log(w / h);
    let best = "1:1", bestD = Infinity;
    for (const [key, [aw, ah]] of Object.entries(ASPECT_PRESETS)) {
        const d = Math.abs(r - Math.log(aw / ah));
        if (d < bestD) { bestD = d; best = key; }
    }
    return best;
};
const splitList = (v) => (Array.isArray(v) ? v : String(v ?? "").split(","))
    .map(s => String(s).trim()).filter(Boolean);
const nextIdFrom = (els) => els.reduce((m, e) => {
    const n = parseInt(String(e.id).match(/(\d+)$/)?.[1] ?? "0", 10);
    return Math.max(m, n);
}, 0) + 1;

const FIELD_CSS = `background:#111;color:#eee;border:1px solid #444;border-radius:3px;
    padding:3px;font-family:monospace;font-size:11px;box-sizing:border-box;`;

function makeBtn(label, bg) {
    const b = document.createElement("button");
    b.textContent = label;
    b.style.cssText = `background:${bg||"#2a2a2a"};color:#eee;border:1px solid #555;
        border-radius:4px;padding:3px 8px;cursor:pointer;font-size:11px;white-space:nowrap;`;
    return b;
}

function makeSelect(options, defaultVal) {
    const s = document.createElement("select");
    s.style.cssText = `background:#222;color:#eee;border:1px solid #555;
        border-radius:4px;padding:2px 4px;font-size:11px;`;
    options.forEach(o => {
        const opt = document.createElement("option");
        opt.value = opt.textContent = o;
        if (o === defaultVal) opt.selected = true;
        s.appendChild(opt);
    });
    s.addEventListener("mousedown", e => e.stopPropagation());
    return s;
}

function makeField(placeholder) {
    const i = document.createElement("input");
    i.type = "text";
    i.placeholder = placeholder || "";
    i.style.cssText = FIELD_CSS + "width:100%;";
    i.addEventListener("mousedown", e => e.stopPropagation());
    return i;
}

function makeNum(min, max, step, width) {
    const i = document.createElement("input");
    i.type = "number";
    i.min = min; i.max = max; i.step = step;
    i.style.cssText = FIELD_CSS + `width:${width}px;`;
    i.addEventListener("mousedown", e => e.stopPropagation());
    return i;
}

function hitHandle(handles, mx, my) {
    for (const h of handles) {
        if (Math.abs(mx - h.x) <= HANDLE_SIZE && Math.abs(my - h.y) <= HANDLE_SIZE) return h;
    }
    return null;
}

app.registerExtension({
    name: "BoyoNodes.BoyoBBOX",

    // The Python node announces the size it actually used (a linked Resolution Selector
    // is invisible to the browser until the graph runs), so the editor can follow it.
    async setup() {
        api.addEventListener("boyo_bbox_size", (ev) => {
            const d = ev.detail || {};
            const id = Number(String(d.node).split(":").pop());
            const node = app.graph?.getNodeById?.(id);
            node?._boyoSetLinkedSize?.(d.width, d.height, d.raw_width, d.raw_height);
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "BoyoBBOX") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        const onConfigure   = nodeType.prototype.onConfigure;
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;

        nodeType.prototype.onConnectionsChange = function () {
            const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
            try { this._boyoRefreshLinked?.(); } catch (e) { /* node still being built */ }
            return r;
        };

        // Saved workflows / pasted nodes: put the saved bbox_data back into the editor.
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            try { this._boyoHydrate?.(); }
            catch (e) { console.error("BoyoBBOX: could not restore saved state", e); }
            return r;
        };

        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            // ── State ────────────────────────────────────────────────────
            // Each format keeps its own stores, so flipping never mangles the other.
            const newFlux = () => ({
                width: 1024, height: 1024, aspect: "1:1", res: "1K",
                caption: "", elements: [], nextId: 1, selectedEl: null,
            });
            const newMoss = () => ({
                width: 1024, height: 1024, aspect: "1:1", res: "1K",
                caption: "", meta: "", canvas: "single", palette: "",
                scale: true,    // boxes keep their place when the canvas size changes
                elements: [], nextId: 1, selectedEl: null,
            });

            const S = {
                format: "flux3",        // "flux3" | "moss"
                mode: "generate",       // Flux only: "generate" | "edit"
                flux: newFlux(),
                moss: newMoss(),
                imageData: null,

                // interaction
                dragMode: null,     // null | "draw" | "move" | "resize"
                dragHandle: null,   // handle name if resizing
                dragStartMx: 0, dragStartMy: 0,
                dragStartBbox: null,
                drawStart: null,    // {x,y} canvas px for new box draw
                currentDraw: null,  // [x1,y1,x2,y2] canvas px live preview

                // ctrl+click cycling
                ctrlCyclePoint: null,   // {mx,my} last ctrl click point
                ctrlCycleIndex: 0,      // current index into candidates
            };
            const store = () => (S.format === "moss" ? S.moss : S.flux);
            // S.width / S.height / S.elements / S.nextId / S.selectedEl always mean "the active format's"
            for (const key of ["width", "height", "elements", "nextId", "selectedEl"]) {
                Object.defineProperty(S, key, {
                    get: () => store()[key],
                    set: (v) => { store()[key] = v; },
                    enumerable: true,
                });
            }

            // ── Panel ────────────────────────────────────────────────────
            const panel = document.createElement("div");
            panel.style.cssText = `width:100%;background:#1a1a1a;border-radius:6px;
                overflow:hidden;display:flex;flex-direction:column;font-family:monospace;
                font-size:12px;color:#ddd;box-sizing:border-box;`;

            // ── Toolbar ──────────────────────────────────────────────────
            const toolbar = document.createElement("div");
            toolbar.style.cssText = `display:flex;gap:5px;padding:5px 7px;background:#111;
                align-items:center;flex-wrap:wrap;border-bottom:1px solid #333;`;

            const formatBtn  = makeBtn("FLUX 3", "#2a4a7a");
            formatBtn.style.cssText += "font-weight:bold;font-size:12px;padding:3px 12px;";
            formatBtn.title = "Switch between Flux 3 and MOSS-Image layouts (each keeps its own boxes and text)";
            const modeBtn    = makeBtn("⚡ Generate", "#2a5a2a");
            const aspectSel  = makeSelect(Object.keys(ASPECT_PRESETS), "1:1");
            const resSel     = makeSelect(Object.keys(RES_PRESETS), "1K");
            const sizeWrap   = document.createElement("span");
            sizeWrap.style.cssText = "display:none;align-items:center;gap:3px;color:#888;font-size:11px;";
            const widthIn    = makeNum(16, 8192, 16, 62);
            const heightIn   = makeNum(16, 8192, 16, 62);
            widthIn.title = heightIn.title = "Canvas pixels (multiples of 16). Changing size never moves boxes.";
            sizeWrap.append(widthIn, "×", heightIn, "px");
            const scaleBtn   = makeBtn("⇔ Scale boxes", "#2a4a2a");
            scaleBtn.title = "On: boxes keep their place when the canvas size changes (this node's size boxes or a linked size).\n" +
                             "Off: boxes stay on the same pixels; anything overhanging is flagged red and clipped at run time.";
            scaleBtn.style.display = "none";
            const linkLbl    = document.createElement("span");
            linkLbl.textContent = "🔗 size from linked input (syncs on run)";
            linkLbl.style.cssText = "display:none;color:#e0a0c0;font-size:10px;";
            const loadImgBtn = makeBtn("📥 Load Image", "#2a2a5a");
            loadImgBtn.style.display = "none";
            loadImgBtn.title = "Load connected image onto canvas";
            const addBoxBtn  = makeBtn("➕ Box", "#3a3a1a");
            const clearBtn   = makeBtn("🗑 Clear", "#4a1a1a");
            clearBtn.title = "Remove all boxes in the current format";
            const hintLbl    = document.createElement("span");
            hintLbl.style.cssText = "color:#555;font-size:10px;margin-left:4px;";

            toolbar.append(formatBtn, modeBtn, aspectSel, resSel, sizeWrap, scaleBtn, linkLbl, loadImgBtn,
                           addBoxBtn, clearBtn, hintLbl);

            // ── Canvas ───────────────────────────────────────────────────
            const canvasWrap = document.createElement("div");
            canvasWrap.style.cssText = `width:100%;background:#0d0d0d;display:flex;
                align-items:center;justify-content:center;overflow:hidden;`;

            const canvas = document.createElement("canvas");
            canvas.width = 512; canvas.height = 512;
            canvas.style.cssText = "cursor:crosshair;display:block;max-width:100%;";
            canvasWrap.appendChild(canvas);
            const ctx = canvas.getContext("2d");

            // ── Caption ──────────────────────────────────────────────────
            const captionWrap = document.createElement("div");
            captionWrap.style.cssText = `padding:5px 7px;border-top:1px solid #333;`;
            const captionLbl = document.createElement("div");
            captionLbl.style.cssText = "margin-bottom:3px;color:#aaa;font-size:11px;";
            const captionTA = document.createElement("textarea");
            captionTA.style.cssText = `width:100%;height:52px;background:#111;color:#eee;
                border:1px solid #444;border-radius:4px;padding:4px;box-sizing:border-box;
                resize:vertical;font-family:monospace;font-size:11px;`;
            captionTA.addEventListener("mousedown", e => e.stopPropagation());

            // MOSS summary extras (meta / canvas / palette) — hidden in Flux format
            const mossSummary = document.createElement("div");
            mossSummary.style.cssText = "display:none;flex-direction:column;gap:4px;margin-top:4px;";
            const metaIn = makeField("meta tags, comma-separated (e.g. product campaign, studio photography)");
            const canvasRow = document.createElement("div");
            canvasRow.style.cssText = "display:flex;gap:6px;align-items:center;color:#aaa;font-size:11px;";
            const canvasSel = makeSelect(MOSS_CANVAS, "single");
            const palIn = makeField("palette (optional), comma-separated");
            palIn.style.cssText += "flex:1;";
            canvasRow.append("canvas", canvasSel, palIn);
            const capWarn = document.createElement("div");
            capWarn.style.cssText = "color:#f66;font-size:10px;min-height:0;";
            mossSummary.append(metaIn, canvasRow);
            captionWrap.append(captionLbl, captionTA, mossSummary, capWarn);

            // ── Element list ─────────────────────────────────────────────
            const elWrap = document.createElement("div");
            elWrap.style.cssText = `max-height:180px;overflow-y:auto;border-top:1px solid #333;
                padding:5px 7px;`;
            const elLbl = document.createElement("div");
            elLbl.style.cssText = "margin-bottom:4px;color:#aaa;font-size:11px;";
            const elList = document.createElement("div");
            elList.style.cssText = "display:flex;flex-direction:column;gap:4px;";
            elWrap.append(elLbl, elList);

            panel.append(toolbar, canvasWrap, captionWrap, elWrap);

            // ── Core helpers ─────────────────────────────────────────────
            // BFL coordinate system is rotated 90°: their X axis = our Y axis, their Y axis = our X axis
            // So we swap x and y when converting to/from BFL normalised space
            const toNorm = (px, py) => {
                const nx = Math.max(0, Math.min(1000, Math.round((px / canvas.width)  * 1000)));
                const ny = Math.max(0, Math.min(1000, Math.round((py / canvas.height) * 1000)));
                return [ny, nx]; // swap: BFL bbox[0]=their_x=our_y, bbox[1]=their_y=our_x
            };

            // Element box -> canvas pixels [x1,y1,x2,y2], per format.
            //  flux3: bbox = [bx1,by1,bx2,by2], normalised 0-1000, bx = our y, by = our x
            //  moss : bbox = [x1,y1,x2,y2] in real image pixels (top-left origin)
            const boxToCanvas = (el) => {
                if (S.format === "moss") {
                    const sx = canvas.width / S.width, sy = canvas.height / S.height;
                    return [el.bbox[0]*sx, el.bbox[1]*sy, el.bbox[2]*sx, el.bbox[3]*sy];
                }
                return [
                    (el.bbox[1]/1000)*canvas.width,  (el.bbox[0]/1000)*canvas.height,
                    (el.bbox[3]/1000)*canvas.width,  (el.bbox[2]/1000)*canvas.height,
                ];
            };

            // Canvas pixels (ordered x1<x2, y1<y2) -> bbox in the active format
            const canvasToBox = (x1, y1, x2, y2) => {
                if (S.format === "moss") {
                    const W = S.width, H = S.height;
                    return [
                        clamp(Math.round(x1 * W / canvas.width),  0, W),
                        clamp(Math.round(y1 * H / canvas.height), 0, H),
                        clamp(Math.round(x2 * W / canvas.width),  0, W),
                        clamp(Math.round(y2 * H / canvas.height), 0, H),
                    ];
                }
                const [nx1,ny1] = toNorm(x1,y1);
                const [nx2,ny2] = toNorm(x2,y2);
                return [nx1,ny1,nx2,ny2];
            };

            const getHandles = (el) => {
                const [px1,py1,px2,py2] = boxToCanvas(el);
                return [
                    { name:"nw", x:px1, y:py1 },
                    { name:"ne", x:px2, y:py1 },
                    { name:"se", x:px2, y:py2 },
                    { name:"sw", x:px1, y:py2 },
                ];
            };

            const pointInBox = (el, mx, my) => {
                const [px1,py1,px2,py2] = boxToCanvas(el);
                return mx >= px1 && mx <= px2 && my >= py1 && my <= py2;
            };

            // All elements whose box contains mx,my — reversed so the topmost (last drawn) is first
            const boxesAtPoint = (elements, mx, my) =>
                elements.filter(el => pointInBox(el, mx, my)).reverse();

            const selectedEl = () => S.elements.find(e => e.id === S.selectedEl) || null;

            const makeEl = (bbox) => {
                const col = COLOURS[S.elements.length % COLOURS.length];
                if (S.format === "moss") {
                    return { id: `item_${S.nextId++}`, label: "object", bbox,
                             desc: "", text: "", tags: "", palette: "", colour: col };
                }
                return { id: `element_${S.nextId++}`, bbox, desc: "", colour: col };
            };

            const syncWidget = () => {
                const w = this.widgets?.find(w => w.name === "bbox_data");
                if (!w) return;
                w.value = JSON.stringify({
                    format: S.format,
                    // Flux keys stay exactly as they always were (top level)
                    mode: S.mode, width: S.flux.width, height: S.flux.height,
                    caption: S.flux.caption,
                    elements: S.flux.elements.map(e => ({ id:e.id, bbox:e.bbox, desc:e.desc })),
                    flux_ui: { aspect: S.flux.aspect, res: S.flux.res },
                    // MOSS draft state
                    moss: {
                        width: S.moss.width, height: S.moss.height,
                        aspect: S.moss.aspect, res: S.moss.res,
                        scale_boxes: S.moss.scale,
                        summary: {
                            caption: S.moss.caption,
                            meta: splitList(S.moss.meta),
                            canvas: S.moss.canvas,
                            palette: splitList(S.moss.palette),
                        },
                        elements: S.moss.elements.map(e => ({
                            id: e.id, label: e.label, bbox: e.bbox, desc: e.desc,
                            text: e.text, tags: splitList(e.tags), palette: splitList(e.palette),
                        })),
                    },
                });
            };

            // ── MOSS validity (display only — the Python compiler is the authority) ──
            const boxProblem = (el) => {
                const [x1,y1,x2,y2] = el.bbox;
                if (!(x1 < x2 && y1 < y2)) return "box has no area";
                if (x1 < 0 || y1 < 0 || x2 > S.width || y2 > S.height)
                    return `box exceeds the ${S.width}×${S.height} canvas`;
                return null;
            };
            const mossIssues = (el) => {
                const out = [];
                if (!String(el.desc || "").trim()) out.push("caption is empty");
                if (el.label === "text" && !String(el.text || "").trim())
                    out.push("text items need their exact text");
                const bp = boxProblem(el);
                if (bp) out.push(bp);
                return out;
            };

            // ── Redraw ───────────────────────────────────────────────────
            const redraw = () => {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                const moss = S.format === "moss";

                // Background
                if (!moss && S.mode === "edit" && S.imageData) {
                    ctx.drawImage(S.imageData, 0, 0, canvas.width, canvas.height);
                } else {
                    ctx.fillStyle = "#1e1e2e";
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    ctx.strokeStyle = "#2a2a3a"; ctx.lineWidth = 1;
                    for (let i = 0; i <= 10; i++) {
                        const x = (i/10)*canvas.width, y = (i/10)*canvas.height;
                        ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,canvas.height); ctx.stroke();
                        ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(canvas.width,y); ctx.stroke();
                    }
                    if (moss) {
                        ctx.font = "10px monospace";
                        ctx.fillStyle = "#6a6a8a";
                        ctx.fillText(`${S.width}×${S.height} px`, 6, canvas.height - 6);
                    }
                }

                // Elements
                S.elements.forEach((el, idx) => {
                    const [x1,y1,x2,y2] = boxToCanvas(el);
                    const isSel = el.id === S.selectedEl;
                    const bad = moss && !!boxProblem(el);
                    const edge = bad ? "#ff3333" : el.colour;

                    // Fill tint
                    ctx.fillStyle = el.colour + (isSel ? "33" : "18");
                    ctx.fillRect(x1, y1, x2-x1, y2-y1);

                    // Border
                    ctx.strokeStyle = edge;
                    ctx.lineWidth = isSel ? 2.5 : 1.5;
                    ctx.setLineDash(bad ? [2,3] : (isSel ? [] : [6,3]));
                    ctx.strokeRect(x1, y1, x2-x1, y2-y1);
                    ctx.setLineDash([]);

                    // Label pill
                    const pill = moss ? `#${idx} ${el.label}${bad ? " ⚠" : ""}` : el.id;
                    ctx.font = "bold 11px monospace";
                    const tw = ctx.measureText(pill).width;
                    ctx.fillStyle = edge + "cc";
                    ctx.beginPath();
                    ctx.roundRect(x1+3, y1+3, tw+8, 16, 3);
                    ctx.fill();
                    ctx.fillStyle = "#000";
                    ctx.fillText(pill, x1+7, y1+15);

                    // Resize handles (selected only)
                    if (isSel) {
                        const handles = getHandles(el);
                        handles.forEach(h => {
                            ctx.fillStyle = "#fff";
                            ctx.strokeStyle = el.colour;
                            ctx.lineWidth = 2;
                            ctx.beginPath();
                            ctx.rect(h.x - HANDLE_SIZE/2, h.y - HANDLE_SIZE/2, HANDLE_SIZE, HANDLE_SIZE);
                            ctx.fill();
                            ctx.stroke();
                        });
                    }
                });

                // Live draw preview
                if (S.dragMode === "draw" && S.currentDraw) {
                    const [x1,y1,x2,y2] = S.currentDraw;
                    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5;
                    ctx.setLineDash([5,5]);
                    ctx.strokeRect(x1, y1, x2-x1, y2-y1);
                    ctx.setLineDash([]);
                    ctx.fillStyle = "#ffffff11";
                    ctx.fillRect(x1, y1, x2-x1, y2-y1);
                }
            };

            // ── Element list builder ─────────────────────────────────────
            const rowEls   = new Map();  // el.id -> row element
            const boxInputs = new Map(); // el.id -> [x1,y1,x2,y2] number inputs (MOSS)
            const issueEls = new Map();  // el.id -> issue line (MOSS)

            const applyRowSelection = () => {
                rowEls.forEach((row, id) => {
                    const el = S.elements.find(e => e.id === id);
                    row.style.outline = (el && id === S.selectedEl) ? `1px solid ${el.colour}66` : "none";
                });
            };

            const updateBoxInputs = (el) => {
                const ins = boxInputs.get(el.id);
                if (ins) el.bbox.forEach((v, i) => { ins[i].value = v; });
            };

            const refreshIssues = () => {
                if (S.format !== "moss") { capWarn.textContent = ""; return; }
                S.moss.elements.forEach(el => {
                    const d = issueEls.get(el.id);
                    if (!d) return;
                    const iss = mossIssues(el);
                    d.textContent = iss.length ? "⚠ " + iss.join(" · ") : "";
                });
                const sw = [];
                if (!S.moss.caption.trim()) sw.push("summary caption is empty");
                if (/<[A-Za-z_][\w-]*>/.test(S.moss.caption))
                    sw.push("inline <tags> are Flux-only; MOSS describes relationships in prose");
                if (!S.moss.elements.length) sw.push("add at least one item");
                if (S.width % 16 || S.height % 16)
                    sw.push(`size ${S.width}×${S.height} is not a multiple of 16 — MOSS will refuse it`);
                capWarn.textContent = sw.length ? "⚠ " + sw.join(" · ") : "";
            };

            const moveEl = (idx, dir) => {
                const a = S.elements, j = idx + dir;
                if (j < 0 || j >= a.length) return;
                [a[idx], a[j]] = [a[j], a[idx]];
                redraw(); buildElList(); syncWidget();
            };

            const fluxRow = (el) => {
                const row = document.createElement("div");
                const isSel = el.id === S.selectedEl;
                row.style.cssText = `display:flex;gap:4px;align-items:flex-start;
                    background:#1f1f1f;border-radius:4px;padding:4px;
                    border-left:3px solid ${el.colour};
                    ${isSel ? "outline:1px solid "+el.colour+"66;" : ""}`;

                const idLbl = document.createElement("div");
                idLbl.textContent = el.id;
                idLbl.style.cssText = `color:${el.colour};min-width:85px;font-weight:bold;
                    cursor:pointer;padding-top:2px;font-size:11px;user-select:none;`;
                idLbl.title = "Click to select";
                idLbl.onclick = () => { S.selectedEl = el.id; redraw(); buildElList(); };

                const ta = document.createElement("textarea");
                ta.value = el.desc;
                ta.placeholder = "Describe this element...";
                ta.style.cssText = `flex:1;height:40px;background:#111;color:#eee;
                    border:1px solid #444;border-radius:3px;padding:3px;
                    font-family:monospace;font-size:11px;resize:vertical;`;
                ta.oninput = () => { el.desc = ta.value; syncWidget(); };

                // Stop textarea mouse events bubbling to canvas
                ta.addEventListener("mousedown", e => e.stopPropagation());

                const del = document.createElement("button");
                del.textContent = "✕";
                del.style.cssText = `background:#4a1a1a;border:none;color:#f88;
                    cursor:pointer;border-radius:3px;padding:2px 5px;align-self:flex-start;`;
                del.onclick = () => {
                    S.elements = S.elements.filter(e => e.id !== el.id);
                    if (S.selectedEl === el.id) S.selectedEl = null;
                    redraw(); buildElList(); syncWidget();
                };

                row.append(idLbl, ta, del);
                return row;
            };

            const mossRow = (el, idx) => {
                const row = document.createElement("div");
                const isSel = el.id === S.selectedEl;
                row.style.cssText = `display:flex;flex-direction:column;gap:4px;
                    background:#1f1f1f;border-radius:4px;padding:4px;
                    border-left:3px solid ${el.colour};
                    ${isSel ? "outline:1px solid "+el.colour+"66;" : ""}`;
                rowEls.set(el.id, row);

                const select = () => {
                    if (S.selectedEl === el.id) return;
                    S.selectedEl = el.id; redraw(); applyRowSelection();
                };

                // head: index · label · reorder · delete
                const head = document.createElement("div");
                head.style.cssText = "display:flex;gap:4px;align-items:center;";
                const idLbl = document.createElement("div");
                idLbl.textContent = `#${idx}`;
                idLbl.style.cssText = `color:${el.colour};min-width:28px;font-weight:bold;
                    cursor:pointer;font-size:11px;user-select:none;`;
                idLbl.title = "Click to select (numbering matches compiler error messages)";
                idLbl.onclick = () => { S.selectedEl = el.id; redraw(); buildElList(); };

                const labelSel = makeSelect(MOSS_LABELS, el.label);
                labelSel.onchange = () => {
                    el.label = labelSel.value;
                    redraw(); buildElList(); syncWidget();
                };

                const spacer = document.createElement("div");
                spacer.style.flex = "1";
                const mkSmall = (txt, bg, fn, title) => {
                    const b = document.createElement("button");
                    b.textContent = txt; b.title = title || "";
                    b.style.cssText = `background:${bg};border:none;color:#ddd;cursor:pointer;
                        border-radius:3px;padding:2px 6px;`;
                    b.onclick = fn;
                    return b;
                };
                const up  = mkSmall("▲", "#2a2a2a", () => moveEl(idx, -1), "Earlier in the items array");
                const dn  = mkSmall("▼", "#2a2a2a", () => moveEl(idx,  1), "Later in the items array");
                const del = mkSmall("✕", "#4a1a1a", () => {
                    S.elements = S.elements.filter(e => e.id !== el.id);
                    if (S.selectedEl === el.id) S.selectedEl = null;
                    redraw(); buildElList(); syncWidget();
                }, "Delete item");
                del.style.color = "#f88";
                head.append(idLbl, labelSel, spacer, up, dn, del);

                // caption
                const ta = document.createElement("textarea");
                ta.value = el.desc;
                ta.placeholder = "Caption: what this item looks like (appearance, material, light" +
                                 (el.label === "text" ? "; typography and surface treatment for text" : "") + ")";
                ta.style.cssText = `${FIELD_CSS}width:100%;height:46px;resize:vertical;`;
                ta.oninput = () => { el.desc = ta.value; syncWidget(); refreshIssues(); };
                ta.addEventListener("mousedown", e => e.stopPropagation());
                ta.addEventListener("focus", select);
                row.append(head, ta);

                // literal text (text items only)
                if (el.label === "text") {
                    const tt = document.createElement("textarea");
                    tt.value = el.text || "";
                    tt.placeholder = "Exact visible characters. Enter = line break; nothing here is rewritten.";
                    tt.style.cssText = `${FIELD_CSS}width:100%;height:36px;resize:vertical;border-color:#6a3a5a;`;
                    tt.oninput = () => { el.text = tt.value; syncWidget(); refreshIssues(); };
                    tt.addEventListener("mousedown", e => e.stopPropagation());
                    tt.addEventListener("focus", select);
                    row.append(tt);
                }

                // exact box (real pixels)
                const boxRow = document.createElement("div");
                boxRow.style.cssText = "display:flex;gap:4px;align-items:center;color:#888;font-size:10px;";
                const ins = ["x1","y1","x2","y2"].map((name, i) => {
                    const inp = makeNum(0, 8192, 1, 58);
                    inp.value = el.bbox[i];
                    inp.title = name;
                    inp.onfocus = select;
                    inp.onchange = () => {
                        const v = Math.round(Number(inp.value));
                        if (!Number.isFinite(v)) { updateBoxInputs(el); return; }
                        const b = [...el.bbox];
                        b[i] = clamp(v, 0, i % 2 === 0 ? S.width : S.height);
                        if (b[0] < b[2] && b[1] < b[3]) el.bbox = b;
                        updateBoxInputs(el); redraw(); refreshIssues(); syncWidget();
                    };
                    return inp;
                });
                boxInputs.set(el.id, ins);
                boxRow.append("x1", ins[0], "y1", ins[1], "x2", ins[2], "y2", ins[3]);
                row.append(boxRow);

                // optional tags / palette
                const optRow = document.createElement("div");
                optRow.style.cssText = "display:flex;gap:4px;";
                const tagsIn = makeField("tags (optional), comma-separated");
                tagsIn.value = el.tags || "";
                tagsIn.oninput = () => { el.tags = tagsIn.value; syncWidget(); };
                tagsIn.addEventListener("focus", select);
                const palIn2 = makeField("palette (optional)");
                palIn2.value = el.palette || "";
                palIn2.oninput = () => { el.palette = palIn2.value; syncWidget(); };
                palIn2.addEventListener("focus", select);
                optRow.append(tagsIn, palIn2);
                row.append(optRow);

                const issue = document.createElement("div");
                issue.style.cssText = "color:#f66;font-size:10px;";
                issueEls.set(el.id, issue);
                row.append(issue);

                return row;
            };

            const buildElList = () => {
                elList.innerHTML = "";
                rowEls.clear(); boxInputs.clear(); issueEls.clear();
                S.elements.forEach((el, idx) => {
                    elList.appendChild(S.format === "moss" ? mossRow(el, idx) : fluxRow(el));
                });
                refreshIssues();
            };

            // ── Canvas sizing ─────────────────────────────────────────────
            const setCanvasAspect = () => {
                const maxW = 620;
                const ratio = S.width / S.height;
                canvas.width  = maxW;
                canvas.height = Math.round(maxW / ratio);
                redraw();
            };

            const syncSizeInputs = () => { widthIn.value = S.width; heightIn.value = S.height; };

            // Boxes are proportional to the canvas they were drawn on: when the canvas changes
            // size, scale by the width and height ratios so the layout follows the new aspect.
            const scaleMossBoxes = (ow, oh, nw, nh) => {
                const rx = nw / ow, ry = nh / oh;
                S.moss.elements.forEach(el => {
                    let [x1, y1, x2, y2] = el.bbox.map(Number);
                    x1 = clamp(Math.round(x1 * rx), 0, nw); x2 = clamp(Math.round(x2 * rx), 0, nw);
                    y1 = clamp(Math.round(y1 * ry), 0, nh); y2 = clamp(Math.round(y2 * ry), 0, nh);
                    if (x2 <= x1) { if (x1 >= nw) x1 = nw - 1; x2 = x1 + 1; }   // never collapse a box
                    if (y2 <= y1) { if (y1 >= nh) y1 = nh - 1; y2 = y1 + 1; }
                    el.bbox = [x1, y1, x2, y2];
                });
            };
            const setMossSize = (w, h) => {
                const ow = S.moss.width, oh = S.moss.height;
                S.moss.width = w; S.moss.height = h;
                if (S.moss.scale && ow > 0 && oh > 0 && (ow !== w || oh !== h)) scaleMossBoxes(ow, oh, w, h);
            };

            const applyGenMode = () => {
                const [rw, rh] = ASPECT_PRESETS[aspectSel.value];
                const base = RES_PRESETS[resSel.value];
                const scale = base / Math.max(rw, rh);
                store().aspect = aspectSel.value;
                store().res = resSel.value;
                if (S.format === "moss") {
                    // MOSS needs multiples of 16; boxes scale with the canvas (if enabled)
                    setMossSize(snap16(rw * scale), snap16(rh * scale));
                } else {
                    S.width  = Math.round(rw * scale);
                    S.height = Math.round(rh * scale);
                    S.imageData = null;
                }
                syncSizeInputs();
                setCanvasAspect();
                if (S.format === "moss") buildElList();
                syncWidget();
            };

            const onCustomSize = () => {
                setMossSize(snap16(Number(widthIn.value) || S.width),
                            snap16(Number(heightIn.value) || S.height));
                syncSizeInputs();
                setCanvasAspect();
                buildElList();
                syncWidget();
            };
            widthIn.onchange = onCustomSize;
            heightIn.onchange = onCustomSize;

            // ── Format / chrome ───────────────────────────────────────────
            const refreshChrome = () => {
                const moss = S.format === "moss";
                formatBtn.textContent = moss ? "MOSS" : "FLUX 3";
                formatBtn.style.background = moss ? "#c2185b" : "#2a4a7a";

                modeBtn.style.display = moss ? "none" : "";
                modeBtn.textContent = S.mode === "generate" ? "⚡ Generate" : "✏️ Edit";
                modeBtn.style.background = S.mode === "generate" ? "#2a5a2a" : "#5a2a5a";

                const showGen = moss || S.mode === "generate";
                aspectSel.style.display = showGen ? "" : "none";
                resSel.style.display    = showGen ? "" : "none";
                sizeWrap.style.display  = moss ? "flex" : "none";
                scaleBtn.style.display  = moss ? "" : "none";
                scaleBtn.textContent    = S.moss.scale ? "⇔ Scale boxes: on" : "⇔ Scale boxes: off";
                scaleBtn.style.background = S.moss.scale ? "#2a5a2a" : "#4a3a2a";
                loadImgBtn.style.display = (!moss && S.mode === "edit") ? "" : "none";
                aspectSel.value = store().aspect;
                resSel.value = store().res;
                syncSizeInputs();

                hintLbl.textContent = moss
                    ? "Draw · Drag · Ctrl+Click to cycle · boxes are real pixels"
                    : "Draw · Drag · Ctrl+Click to cycle stack";

                captionLbl.textContent = moss
                    ? "Summary caption (global description of the whole image — no <id> tags):"
                    : "Scene caption (include <element_id> tags inline):";
                captionTA.placeholder = moss
                    ? "Medium, lighting, palette, mood, composition. Items below describe the individual pieces."
                    : "A dramatic scene with a woman <woman_1> standing under a lamp <lamp_1>...";
                captionTA.value = store().caption;
                mossSummary.style.display = moss ? "flex" : "none";
                metaIn.value = S.moss.meta;
                canvasSel.value = S.moss.canvas;
                palIn.value = S.moss.palette;

                elLbl.textContent = moss
                    ? "Items — flat list, in order. Click # to select on canvas:"
                    : "Elements — click ID to select on canvas:";
                elWrap.style.maxHeight = moss ? "360px" : "180px";

                // Same four output slots in both formats; only the label on slot 2 changes
                if (this.outputs?.[1]) this.outputs[1].label = moss ? "draft_json" : "bbox_json";
                this._boyoRefreshLinked?.();
                this.setDirtyCanvas?.(true, true);
            };

            const switchFormat = () => {
                S.dragMode = null; S.dragHandle = null; S.currentDraw = null; S.dragStartBbox = null;
                S.ctrlCyclePoint = null;
                S.format = S.format === "moss" ? "flux3" : "moss";
                refreshChrome();
                setCanvasAspect();
                buildElList();
                syncWidget();
            };
            formatBtn.onclick = switchFormat;

            // ── Canvas cursor helper ──────────────────────────────────────
            const updateCursor = (mx, my) => {
                const sel = selectedEl();
                if (sel) {
                    const h = hitHandle(getHandles(sel), mx, my);
                    if (h) {
                        const cursors = { nw:"nw-resize", ne:"ne-resize", se:"se-resize", sw:"sw-resize" };
                        canvas.style.cursor = cursors[h.name];
                        return;
                    }
                    if (pointInBox(sel, mx, my)) {
                        canvas.style.cursor = "move";
                        return;
                    }
                }
                // Any other box?
                const hit = boxesAtPoint(S.elements, mx, my);
                canvas.style.cursor = hit.length ? "pointer" : "crosshair";
            };

            // ── Mouse events ──────────────────────────────────────────────
            const cpos = (e) => {
                const r = canvas.getBoundingClientRect();
                // Scale mouse to canvas internal coords
                const scaleX = canvas.width  / r.width;
                const scaleY = canvas.height / r.height;
                return {
                    x: (e.clientX - r.left) * scaleX,
                    y: (e.clientY - r.top)  * scaleY,
                };
            };

            canvas.addEventListener("mousedown", (e) => {
                if (e.button !== 0) return;
                e.stopPropagation();
                e.preventDefault();

                const { x: mx, y: my } = cpos(e);

                // ── Ctrl+click: cycle stacked boxes ──────────────────────
                if (e.ctrlKey) {
                    const candidates = boxesAtPoint(S.elements, mx, my);
                    if (candidates.length === 0) return;

                    // If same point as last ctrl click, advance index
                    const samePoint = S.ctrlCyclePoint &&
                        Math.abs(mx - S.ctrlCyclePoint.x) < 10 &&
                        Math.abs(my - S.ctrlCyclePoint.y) < 10;

                    if (samePoint) {
                        S.ctrlCycleIndex = (S.ctrlCycleIndex + 1) % candidates.length;
                    } else {
                        S.ctrlCyclePoint = { x: mx, y: my };
                        S.ctrlCycleIndex = 0;
                    }

                    S.selectedEl = candidates[S.ctrlCycleIndex].id;
                    redraw(); buildElList();
                    return;
                }

                // Reset ctrl cycle state on normal click
                S.ctrlCyclePoint = null;

                // ── Check selected box handles first ──────────────────────
                const sel = selectedEl();
                if (sel) {
                    const h = hitHandle(getHandles(sel), mx, my);
                    if (h) {
                        S.dragMode = "resize";
                        S.dragHandle = h.name;
                        S.dragStartMx = mx; S.dragStartMy = my;
                        S.dragStartBbox = [...sel.bbox];
                        return;
                    }
                    // Click inside selected box → move
                    if (pointInBox(sel, mx, my)) {
                        S.dragMode = "move";
                        S.dragStartMx = mx; S.dragStartMy = my;
                        S.dragStartBbox = [...sel.bbox];
                        return;
                    }
                }

                // ── Click unselected box → select it ─────────────────────
                const hits = boxesAtPoint(S.elements, mx, my);
                if (hits.length > 0) {
                    // Select topmost that isn't already selected
                    const pick = hits.find(e => e.id !== S.selectedEl) || hits[0];
                    S.selectedEl = pick.id;
                    // Then move it
                    S.dragMode = "move";
                    S.dragStartMx = mx; S.dragStartMy = my;
                    S.dragStartBbox = [...pick.bbox];
                    redraw(); buildElList();
                    return;
                }

                // ── Empty space → draw new box ────────────────────────────
                S.selectedEl = null;
                S.dragMode = "draw";
                S.drawStart = { x: mx, y: my };
                S.currentDraw = [mx, my, mx, my];
                redraw(); buildElList();
            });

            canvas.addEventListener("mousemove", (e) => {
                const { x: mx, y: my } = cpos(e);

                if (!S.dragMode) {
                    updateCursor(mx, my);
                    return;
                }

                if (S.dragMode === "draw") {
                    S.currentDraw = [
                        Math.min(S.drawStart.x, mx), Math.min(S.drawStart.y, my),
                        Math.max(S.drawStart.x, mx), Math.max(S.drawStart.y, my),
                    ];
                    redraw();
                    return;
                }

                const sel = selectedEl();
                if (!sel) return;

                const dx = mx - S.dragStartMx;
                const dy = my - S.dragStartMy;
                const ob = S.dragStartBbox; // original bbox

                // ── MOSS: real pixels, [x1,y1,x2,y2] ─────────────────────
                if (S.format === "moss") {
                    const W = S.width, H = S.height;
                    const dX = Math.round(dx * W / canvas.width);
                    const dY = Math.round(dy * H / canvas.height);
                    const [ox1,oy1,ox2,oy2] = ob;

                    if (S.dragMode === "move") {
                        const w = ox2 - ox1, h = oy2 - oy1;
                        const x1 = clamp(ox1 + dX, 0, Math.max(0, W - w));
                        const y1 = clamp(oy1 + dY, 0, Math.max(0, H - h));
                        sel.bbox = [x1, y1, x1 + w, y1 + h];
                    } else {
                        let [x1,y1,x2,y2] = ob;
                        if (S.dragHandle.includes("n")) y1 = clamp(oy1 + dY, 0, y2 - MOSS_MIN_BOX);
                        if (S.dragHandle.includes("s")) y2 = clamp(oy2 + dY, y1 + MOSS_MIN_BOX, H);
                        if (S.dragHandle.includes("w")) x1 = clamp(ox1 + dX, 0, x2 - MOSS_MIN_BOX);
                        if (S.dragHandle.includes("e")) x2 = clamp(ox2 + dX, x1 + MOSS_MIN_BOX, W);
                        sel.bbox = [x1, y1, x2, y2];
                    }
                    updateBoxInputs(sel);
                    syncWidget();
                    redraw();
                    refreshIssues();
                    return;
                }

                // ── Flux 3: BFL normalised space ─────────────────────────
                // BFL x=our_y, BFL y=our_x — so canvas dx maps to BFL dy and canvas dy maps to BFL dx
                const dnBFLx = Math.round((dy / canvas.height) * 1000); // canvas dy → BFL x delta
                const dnBFLy = Math.round((dx / canvas.width)  * 1000); // canvas dx → BFL y delta

                if (S.dragMode === "move") {
                    const w = ob[2] - ob[0], h = ob[3] - ob[1];
                    let bx1 = Math.max(0, Math.min(1000-w, ob[0] + dnBFLx));
                    let by1 = Math.max(0, Math.min(1000-h, ob[1] + dnBFLy));
                    sel.bbox = [bx1, by1, bx1+w, by1+h];
                    syncWidget();
                    redraw();
                    return;
                }

                if (S.dragMode === "resize") {
                    // BFL bbox [bx1,by1,bx2,by2]
                    // handles: n/s affect BFL x (our y), w/e affect BFL y (our x)
                    let [bx1,by1,bx2,by2] = ob;
                    if (S.dragHandle.includes("n")) bx1 = Math.max(0, Math.min(bx2-20, ob[0]+dnBFLx));
                    if (S.dragHandle.includes("s")) bx2 = Math.min(1000, Math.max(bx1+20, ob[2]+dnBFLx));
                    if (S.dragHandle.includes("w")) by1 = Math.max(0, Math.min(by2-20, ob[1]+dnBFLy));
                    if (S.dragHandle.includes("e")) by2 = Math.min(1000, Math.max(by1+20, ob[3]+dnBFLy));
                    sel.bbox = [bx1,by1,bx2,by2];
                    syncWidget();
                    redraw();
                }
            });

            const endDrag = (e) => {
                if (S.dragMode === "draw") {
                    const { x: mx, y: my } = cpos(e);
                    const x1=Math.min(S.drawStart.x,mx), y1=Math.min(S.drawStart.y,my);
                    const x2=Math.max(S.drawStart.x,mx), y2=Math.max(S.drawStart.y,my);
                    if (Math.abs(x2-x1) > 8 && Math.abs(y2-y1) > 8) {
                        const bbox = canvasToBox(x1, y1, x2, y2);
                        const valid = S.format === "moss" ? (bbox[0] < bbox[2] && bbox[1] < bbox[3]) : true;
                        if (valid) {
                            const el = makeEl(bbox);
                            S.elements.push(el);
                            S.selectedEl = el.id;
                            buildElList(); syncWidget();
                        }
                    }
                    S.currentDraw = null;
                }
                S.dragMode = null;
                S.dragHandle = null;
                S.dragStartBbox = null;
                redraw();
            };

            canvas.addEventListener("mouseup",    endDrag);
            canvas.addEventListener("mouseleave", endDrag);

            // ── Toolbar actions ───────────────────────────────────────────
            modeBtn.onclick = () => {
                S.mode = S.mode === "generate" ? "edit" : "generate";
                refreshChrome();
                if (S.mode === "generate") applyGenMode();
                else { S.imageData = null; redraw(); }
                syncWidget();
            };

            // ── Load image button — grabs from connected node's output cache ─
            // (Flux edit mode only; the button is hidden in MOSS format)
            loadImgBtn.onclick = async () => {
                loadImgBtn.textContent = "Loading...";
                loadImgBtn.disabled = true;
                try {
                    // Find what's connected to our image input (slot 0 of optional inputs)
                    // image is the first optional input, inputs[0] on this node
                    const imageInputIndex = this.inputs?.findIndex(i => i.name === "image");
                    if (imageInputIndex === -1 || imageInputIndex === undefined) {
                        alert("No image input found on node."); return;
                    }
                    const link = this.inputs[imageInputIndex]?.link;
                    if (!link) {
                        alert("Nothing connected to the image input."); return;
                    }

                    // Get the link info to find source node + slot
                    const linkInfo = app.graph.links[link];
                    if (!linkInfo) { alert("Could not resolve link."); return; }

                    const srcNode = app.graph.getNodeById(linkInfo.origin_id);
                    if (!srcNode) { alert("Source node not found."); return; }

                    // Check if source node has cached output images
                    const imgs = srcNode.imgs;
                    if (imgs && imgs.length > 0) {
                        // Use the first image from the source node's output slot
                        const srcImg = imgs[linkInfo.origin_slot] || imgs[0];
                        // srcImg is already an HTMLImageElement ComfyUI loaded
                        S.imageData = srcImg;
                        S.width  = srcImg.naturalWidth;
                        S.height = srcImg.naturalHeight;
                        setCanvasAspect();
                        syncWidget();
                        return;
                    }

                    // Fallback: try to get output from history API
                    const history = await api.fetchApi("/history?max_items=1");
                    const histData = await history.json();
                    const lastRun = Object.values(histData)[0];
                    if (!lastRun) { alert("No cached output found — run the workflow once first."); return; }

                    const nodeOutputs = lastRun?.outputs?.[String(srcNode.id)];
                    const images = nodeOutputs?.images;
                    if (!images || images.length === 0) {
                        alert("Source node has no cached image output.\nRun the workflow once to cache it.");
                        return;
                    }

                    const imgInfo = images[linkInfo.origin_slot] || images[0];
                    const url = api.apiURL(`/view?filename=${encodeURIComponent(imgInfo.filename)}&type=${imgInfo.type}&subfolder=${encodeURIComponent(imgInfo.subfolder || "")}`);

                    const img = new Image();
                    img.onload = () => {
                        S.imageData = img;
                        S.width  = img.naturalWidth;
                        S.height = img.naturalHeight;
                        setCanvasAspect();
                        syncWidget();
                    };
                    img.onerror = () => alert("Failed to load image from cache.");
                    img.src = url + "&t=" + Date.now();

                } catch(e) {
                    alert("Error loading image: " + e.message);
                    console.error(e);
                } finally {
                    loadImgBtn.textContent = "📥 Load Image";
                    loadImgBtn.disabled = false;
                }
            };

            aspectSel.onchange = applyGenMode;
            resSel.onchange    = applyGenMode;

            addBoxBtn.onclick = () => {
                let bbox;
                if (S.format === "moss") {
                    bbox = [Math.round(S.width * 0.1), Math.round(S.height * 0.1),
                            Math.round(S.width * 0.9), Math.round(S.height * 0.9)];
                } else {
                    bbox = [100, 100, 900, 900];
                }
                const el = makeEl(bbox);
                S.elements.push(el);
                S.selectedEl = el.id;
                redraw(); buildElList(); syncWidget();
            };

            clearBtn.onclick = () => {
                S.elements = []; S.nextId = 1; S.selectedEl = null;
                redraw(); buildElList(); syncWidget();
            };

            captionTA.oninput = () => {
                store().caption = captionTA.value;
                syncWidget();
                refreshIssues();
            };
            scaleBtn.onclick = () => { S.moss.scale = !S.moss.scale; refreshChrome(); syncWidget(); };
            metaIn.oninput    = () => { S.moss.meta = metaIn.value; syncWidget(); };
            palIn.oninput     = () => { S.moss.palette = palIn.value; syncWidget(); };
            canvasSel.onchange = () => { S.moss.canvas = canvasSel.value; syncWidget(); };

            // ── Linked size (width_in / height_in sockets) ───────────────
            const isLinked = () => ["width_in", "height_in"]
                .some(nm => this.inputs?.find(i => i.name === nm)?.link != null);

            let lastLink = null;   // { w, h: linked size; mw, mh: size MOSS uses after rounding down }

            const refreshLinked = () => {
                const linked = isLinked();
                for (const el of [aspectSel, resSel, widthIn, heightIn]) {
                    el.disabled = linked;
                    el.style.opacity = linked ? "0.45" : "1";
                }
                // The resolution preset means nothing when the size comes from a link; the label says it all
                const showGen = S.format === "moss" || S.mode === "generate";
                resSel.style.display = (showGen && !linked) ? "" : "none";
                linkLbl.style.display = linked ? "" : "none";
                if (linked) {
                    const l = lastLink || { w: S.width, h: S.height, mw: S.width, mh: S.height };
                    const asp = nearestAspect(l.w, l.h);
                    aspectSel.value = asp;            // the aspect comes from the incoming size
                    const rounded = S.format === "moss" && (l.mw !== l.w || l.mh !== l.h);
                    linkLbl.textContent = `🔗 ${l.w}×${l.h}${rounded ? ` → ${l.mw}×${l.mh}` : ""} · ≈${asp} · ${(l.w * l.h / 1e6).toFixed(1)} MP`;
                }
                syncSizeInputs();
            };
            this._boyoRefreshLinked = refreshLinked;

            // Called with the size Python really used (it scaled/clipped the boxes the same way).
            this._boyoSetLinkedSize = (w, h, rw, rh) => {
                // rw/rh = the size that was actually linked (before MOSS rounded it down); older events only carry w/h
                const W = Math.round(Number(rw ?? w)), H = Math.round(Number(rh ?? h));
                if (!(W > 0 && H > 0)) return;
                const mw = floor16(W), mh = floor16(H);
                lastLink = { w: W, h: H, mw, mh };
                const asp = nearestAspect(W, H);
                S.moss.aspect = asp;
                if (S.mode === "generate") {                          // Flux edit mode keeps its image size
                    S.flux.aspect = asp; S.flux.width = W; S.flux.height = H;
                }
                setMossSize(mw, mh);                                  // MOSS is generation-only, multiples of 16
                aspectSel.value = store().aspect;
                refreshLinked();
                setCanvasAspect();
                buildElList();
                syncWidget();
            };

            // ── Restore saved state (workflow load, node paste) ───────────
            this._boyoHydrate = () => {
                const w = this.widgets?.find(w => w.name === "bbox_data");
                let d;
                try { d = JSON.parse(w?.value || "{}"); } catch (e) { return; }
                if (!d || typeof d !== "object" || Array.isArray(d)) return;

                const num = (v, fb) => (Number.isFinite(Number(v)) && Number(v) > 0 ? Number(v) : fb);
                const box4 = (b, fb) => (Array.isArray(b) && b.length === 4 && b.every(v => Number.isFinite(Number(v)))
                    ? b.map(Number) : fb);
                const preset = (a, r) => ({
                    aspect: ASPECT_PRESETS[a] ? a : "1:1",
                    res:    RES_PRESETS[r] ? r : "1K",
                });

                // Flux (also reads workflows saved before the format toggle existed)
                S.mode = d.mode === "edit" ? "edit" : "generate";
                S.flux.width  = num(d.width, 1024);
                S.flux.height = num(d.height, 1024);
                Object.assign(S.flux, preset(d.flux_ui?.aspect, d.flux_ui?.res));
                S.flux.caption = String(d.caption ?? "");
                S.flux.elements = (Array.isArray(d.elements) ? d.elements : []).map((e, i) => ({
                    id: String(e.id ?? `element_${i + 1}`),
                    bbox: box4(e.bbox, [100, 100, 900, 900]),
                    desc: String(e.desc ?? ""),
                    colour: COLOURS[i % COLOURS.length],
                }));
                S.flux.nextId = nextIdFrom(S.flux.elements);
                S.flux.selectedEl = null;

                // MOSS
                const m = d.moss;
                if (m && typeof m === "object") {
                    S.moss.width  = snap16(num(m.width, 1024));
                    S.moss.height = snap16(num(m.height, 1024));
                    Object.assign(S.moss, preset(m.aspect, m.res));
                    const sm = m.summary || {};
                    S.moss.caption = String(sm.caption ?? "");
                    S.moss.meta    = splitList(sm.meta).join(", ");
                    S.moss.scale   = m.scale_boxes !== false;
                    S.moss.canvas  = MOSS_CANVAS.includes(sm.canvas) ? sm.canvas : "single";
                    S.moss.palette = splitList(sm.palette).join(", ");
                    S.moss.elements = (Array.isArray(m.elements) ? m.elements : []).map((e, i) => ({
                        id: String(e.id ?? `item_${i + 1}`),
                        label: MOSS_LABELS.includes(e.label) ? e.label : "object",
                        bbox: box4(e.bbox, [0, 0, S.moss.width, S.moss.height]),
                        desc: String(e.desc ?? ""),
                        text: String(e.text ?? ""),
                        tags: splitList(e.tags).join(", "),
                        palette: splitList(e.palette).join(", "),
                        colour: COLOURS[i % COLOURS.length],
                    }));
                    S.moss.nextId = nextIdFrom(S.moss.elements);
                    S.moss.selectedEl = null;
                }

                S.format = d.format === "moss" ? "moss" : "flux3";
                S.imageData = null; // re-load via the Load Image button in Flux edit mode
                refreshChrome();
                setCanvasAspect();
                buildElList();
                refreshLinked();
                syncWidget();
            };

            // ── Mount widget ──────────────────────────────────────────────
            this.addDOMWidget("bbox_panel", "preview", panel, {
                serialize: false,
                hideOnZoom: false,
            });

            // Initial render
            refreshChrome();
            applyGenMode();
            refreshLinked();

            return r;
        };
    },
});
