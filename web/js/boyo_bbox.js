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
    return s;
}

// Returns canvas pixel coords of the 4 corner handles for a box
// BFL bbox = [bx1,by1,bx2,by2] where bx=our_y, by=our_x
// So canvas px = (by/1000)*cw, canvas py = (bx/1000)*ch
function getHandles(el, cw, ch) {
    const px1 = (el.bbox[1]/1000)*cw, py1 = (el.bbox[0]/1000)*ch;
    const px2 = (el.bbox[3]/1000)*cw, py2 = (el.bbox[2]/1000)*ch;
    return [
        { name:"nw", x:px1, y:py1 },
        { name:"ne", x:px2, y:py1 },
        { name:"se", x:px2, y:py2 },
        { name:"sw", x:px1, y:py2 },
    ];
}

function hitHandle(handles, mx, my) {
    for (const h of handles) {
        if (Math.abs(mx - h.x) <= HANDLE_SIZE && Math.abs(my - h.y) <= HANDLE_SIZE) return h;
    }
    return null;
}

function pointInBox(el, mx, my, cw, ch) {
    const px1 = (el.bbox[1]/1000)*cw, py1 = (el.bbox[0]/1000)*ch;
    const px2 = (el.bbox[3]/1000)*cw, py2 = (el.bbox[2]/1000)*ch;
    return mx >= px1 && mx <= px2 && my >= py1 && my <= py2;
}

// All elements whose bbox contains mx,my — returns them in reverse order (topmost last drawn = first)
function boxesAtPoint(elements, mx, my, cw, ch) {
    return elements.filter(el => pointInBox(el, mx, my, cw, ch)).reverse();
}

app.registerExtension({
    name: "BoyoNodes.BoyoBBOX",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "BoyoBBOX") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            // ── State ────────────────────────────────────────────────────
            const S = {
                mode: "generate",
                width: 1024, height: 1024,
                elements: [], nextId: 1,
                selectedEl: null,
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

            // ── Panel ────────────────────────────────────────────────────
            const panel = document.createElement("div");
            panel.style.cssText = `width:100%;background:#1a1a1a;border-radius:6px;
                overflow:hidden;display:flex;flex-direction:column;font-family:monospace;
                font-size:12px;color:#ddd;box-sizing:border-box;`;

            // ── Toolbar ──────────────────────────────────────────────────
            const toolbar = document.createElement("div");
            toolbar.style.cssText = `display:flex;gap:5px;padding:5px 7px;background:#111;
                align-items:center;flex-wrap:wrap;border-bottom:1px solid #333;`;

            const modeBtn    = makeBtn("⚡ Generate", "#2a5a2a");
            const aspectSel  = makeSelect(Object.keys(ASPECT_PRESETS), "1:1");
            const resSel     = makeSelect(Object.keys(RES_PRESETS), "1K");
            const loadImgBtn = makeBtn("📥 Load Image", "#2a2a5a");
            loadImgBtn.style.display = "none";
            loadImgBtn.title = "Load connected image onto canvas";
            const addBoxBtn  = makeBtn("➕ Box", "#3a3a1a");
            const clearBtn   = makeBtn("🗑 Clear", "#4a1a1a");
            const hintLbl    = document.createElement("span");
            hintLbl.textContent = "Draw · Drag · Ctrl+Click to cycle stack";
            hintLbl.style.cssText = "color:#555;font-size:10px;margin-left:4px;";

            toolbar.append(modeBtn, aspectSel, resSel, loadImgBtn,
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
            captionLbl.textContent = "Scene caption (include <element_id> tags inline):";
            captionLbl.style.cssText = "margin-bottom:3px;color:#aaa;font-size:11px;";
            const captionTA = document.createElement("textarea");
            captionTA.placeholder = "A dramatic scene with a woman <woman_1> standing under a lamp <lamp_1>...";
            captionTA.style.cssText = `width:100%;height:52px;background:#111;color:#eee;
                border:1px solid #444;border-radius:4px;padding:4px;box-sizing:border-box;
                resize:vertical;font-family:monospace;font-size:11px;`;
            captionWrap.append(captionLbl, captionTA);

            // ── Element list ─────────────────────────────────────────────
            const elWrap = document.createElement("div");
            elWrap.style.cssText = `max-height:180px;overflow-y:auto;border-top:1px solid #333;
                padding:5px 7px;`;
            const elLbl = document.createElement("div");
            elLbl.textContent = "Elements — click ID to select on canvas:";
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

            // Inverse: BFL norm [bx, by] → canvas pixels [px, py]
            // BFL bx = our ny, BFL by = our nx → canvas px = (by/1000)*cw, canvas py = (bx/1000)*ch
            const toCanvasPx = (bx, by) => [
                (by / 1000) * canvas.width,   // BFL by → our x
                (bx / 1000) * canvas.height,  // BFL bx → our y
            ];

            const selectedEl = () => S.elements.find(e => e.id === S.selectedEl) || null;

            const syncWidget = () => {
                const w = this.widgets?.find(w => w.name === "bbox_data");
                if (w) w.value = JSON.stringify({
                    mode: S.mode, width: S.width, height: S.height,
                    caption: captionTA.value,
                    elements: S.elements.map(e => ({ id:e.id, bbox:e.bbox, desc:e.desc })),
                });
            };

            // ── Redraw ───────────────────────────────────────────────────
            const redraw = () => {
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Background
                if (S.mode === "edit" && S.imageData) {
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
                }

                // Elements
                S.elements.forEach(el => {
                    const [x1,y1] = toCanvasPx(el.bbox[0], el.bbox[1]);
                    const [x2,y2] = toCanvasPx(el.bbox[2], el.bbox[3]);
                    const isSel = el.id === S.selectedEl;

                    // Fill tint
                    ctx.fillStyle = el.colour + (isSel ? "33" : "18");
                    ctx.fillRect(x1, y1, x2-x1, y2-y1);

                    // Border
                    ctx.strokeStyle = el.colour;
                    ctx.lineWidth = isSel ? 2.5 : 1.5;
                    ctx.setLineDash(isSel ? [] : [6,3]);
                    ctx.strokeRect(x1, y1, x2-x1, y2-y1);
                    ctx.setLineDash([]);

                    // Label pill
                    ctx.font = "bold 11px monospace";
                    const tw = ctx.measureText(el.id).width;
                    ctx.fillStyle = el.colour + "cc";
                    ctx.beginPath();
                    ctx.roundRect(x1+3, y1+3, tw+8, 16, 3);
                    ctx.fill();
                    ctx.fillStyle = "#000";
                    ctx.fillText(el.id, x1+7, y1+15);

                    // Resize handles (selected only)
                    if (isSel) {
                        const handles = getHandles(el, canvas.width, canvas.height);
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
            const buildElList = () => {
                elList.innerHTML = "";
                S.elements.forEach(el => {
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
                    elList.appendChild(row);
                });
            };

            // ── Canvas sizing ─────────────────────────────────────────────
            const setCanvasAspect = () => {
                const maxW = 620;
                const ratio = S.width / S.height;
                canvas.width  = maxW;
                canvas.height = Math.round(maxW / ratio);
                redraw();
            };

            const applyGenMode = () => {
                const [rw, rh] = ASPECT_PRESETS[aspectSel.value];
                const base = RES_PRESETS[resSel.value];
                const scale = base / Math.max(rw, rh);
                S.width  = Math.round(rw * scale);
                S.height = Math.round(rh * scale);
                S.imageData = null;
                setCanvasAspect();
                syncWidget();
            };

            // ── Canvas cursor helper ──────────────────────────────────────
            const updateCursor = (mx, my) => {
                const sel = selectedEl();
                if (sel) {
                    const h = hitHandle(getHandles(sel, canvas.width, canvas.height), mx, my);
                    if (h) {
                        const cursors = { nw:"nw-resize", ne:"ne-resize", se:"se-resize", sw:"sw-resize" };
                        canvas.style.cursor = cursors[h.name];
                        return;
                    }
                    if (pointInBox(sel, mx, my, canvas.width, canvas.height)) {
                        canvas.style.cursor = "move";
                        return;
                    }
                }
                // Any other box?
                const hit = boxesAtPoint(S.elements, mx, my, canvas.width, canvas.height);
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
                    const candidates = boxesAtPoint(S.elements, mx, my, canvas.width, canvas.height);
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
                    const h = hitHandle(getHandles(sel, canvas.width, canvas.height), mx, my);
                    if (h) {
                        S.dragMode = "resize";
                        S.dragHandle = h.name;
                        S.dragStartMx = mx; S.dragStartMy = my;
                        S.dragStartBbox = [...sel.bbox];
                        return;
                    }
                    // Click inside selected box → move
                    if (pointInBox(sel, mx, my, canvas.width, canvas.height)) {
                        S.dragMode = "move";
                        S.dragStartMx = mx; S.dragStartMy = my;
                        S.dragStartBbox = [...sel.bbox];
                        return;
                    }
                }

                // ── Click unselected box → select it ─────────────────────
                const hits = boxesAtPoint(S.elements, mx, my, canvas.width, canvas.height);
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
                // BFL x=our_y, BFL y=our_x — so canvas dx maps to BFL dy and canvas dy maps to BFL dx
                const dnBFLx = Math.round((dy / canvas.height) * 1000); // canvas dy → BFL x delta
                const dnBFLy = Math.round((dx / canvas.width)  * 1000); // canvas dx → BFL y delta
                const ob = S.dragStartBbox; // original bbox [bx1,by1,bx2,by2]

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
                        const [nx1,ny1] = toNorm(x1,y1);
                        const [nx2,ny2] = toNorm(x2,y2);
                        const id  = `element_${S.nextId++}`;
                        const col = COLOURS[S.elements.length % COLOURS.length];
                        S.elements.push({ id, bbox:[nx1,ny1,nx2,ny2], desc:"", colour:col });
                        S.selectedEl = id;
                        buildElList(); syncWidget();
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
                modeBtn.textContent      = S.mode === "generate" ? "⚡ Generate" : "✏️ Edit";
                modeBtn.style.background = S.mode === "generate" ? "#2a5a2a" : "#5a2a5a";
                aspectSel.style.display  = S.mode === "generate" ? "" : "none";
                resSel.style.display     = S.mode === "generate" ? "" : "none";
                loadImgBtn.style.display = S.mode === "edit"     ? "" : "none";
                if (S.mode === "generate") applyGenMode();
                else { S.imageData = null; redraw(); }
                syncWidget();
            };

            // ── Load image button — grabs from connected node's output cache ─
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
                const id  = `element_${S.nextId++}`;
                const col = COLOURS[S.elements.length % COLOURS.length];
                S.elements.push({ id, bbox:[100,100,900,900], desc:"", colour:col });
                S.selectedEl = id;
                redraw(); buildElList(); syncWidget();
            };

            clearBtn.onclick = () => {
                S.elements = []; S.nextId = 1; S.selectedEl = null;
                redraw(); buildElList(); syncWidget();
            };

            captionTA.oninput = syncWidget;

            // ── Mount widget ──────────────────────────────────────────────
            this.addDOMWidget("bbox_panel", "preview", panel, {
                serialize: false,
                hideOnZoom: false,
            });

            // Initial render
            applyGenMode();

            return r;
        };
    },
});
