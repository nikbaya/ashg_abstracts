// Poster hall map for Find talks: window.hallMap(board) returns SVG markup for a board id such as "1005W".
// Board numbers are shared across the Wed/Thu/Fri sessions, so the trailing day letter is ignored.
(function () {
  // Overview rectangles [x, y, w, h] trace the convention-center floor plan (1860 px wide); `digit` is the board number's first digit.
  // `gaps` lists boards with a pillar-sized gap above them (the column is that many boards short of its neighbors). `cols` lists one row-pair per column as [first left board, first right board, boards per side]: the left side counts up from the
  // bottom of the column, the right side counts up from the top, as printed on the floor plan. Blocks without `cols` have no board-level detail yet.
  const BLOCKS = [
    { id: "1B", digit: 1, rect: [155, 398, 138, 66], cols: [[1007, 1012, 5], [1029, 1034, 5], [1051, 1056, 5], [1073, 1078, 5], [1095, 1100, 5], [1111, 1116, 5], [1121, 1126, 5], [1131, 1136, 5]] },
    { id: "1A", digit: 1, rect: [158, 568, 84, 58], cols: [[1001, 1017, 6], [1023, 1039, 6], [1045, 1061, 6], [1067, 1083, 6], [1089, 1105, 6]] },
    { id: "2B", digit: 2, rect: [360, 398, 36, 146], cols: [[2008, 2016, 8], [2038, 2046, 8]] },
    { id: "2A", digit: 2, rect: [360, 550, 36, 106], cols: [[2001, 2024, 7], [2031, 2054, 7]] },
    { id: "3", digit: 3, rect: [511, 550, 36, 106], cols: [[3001, 3007, 6], [3013, 3020, 7]] },
    { id: "4B", digit: 4, rect: [867, 398, 36, 146], cols: [[4008, 4016, 8], [4038, 4045, 7]] },
    { id: "4A", digit: 4, rect: [867, 550, 36, 106], cols: [[4001, 4024, 7], [4031, 4052, 7]] },
    { id: "5", digit: 5, rect: [1018, 550, 36, 106], cols: [[5001, 5007, 6], [5013, 5019, 6]] },
    { id: "6B", digit: 6, rect: [1172, 432, 68, 83], cols: [[6007, 6011, 4], [6027, 6032, 5], [6049, 6054, 5], [6071, 6076, 5]] },
    { id: "6A", digit: 6, rect: [1172, 550, 68, 85], cols: [[6001, 6015, 6], [6021, 6037, 6], [6043, 6059, 6], [6065, 6081, 6]] },
    { id: "7", digit: 7, rect: [1462, 432, 50, 205], cols: [[7007, 7013, 6, 0], [7031, 7037, 6, 0], [7055, 7061, 6, 0], [7001, 7019, 6, 1], [7025, 7043, 6, 1], [7049, 7067, 6, 1]] },
    { id: "8B", digit: 8, rect: [1238, 155, 274, 120], gaps: [8285, 8411], cols: [[8251, 8259, 8], [8267, 8275, 8], [8283, 8290, 7], [8297, 8305, 8], [8313, 8321, 8], [8329, 8337, 8], [8345, 8353, 8], [8361, 8369, 8], [8377, 8385, 8], [8393, 8401, 8], [8409, 8416, 7], [8423, 8431, 8], [8439, 8447, 8], [8455, 8463, 8], [8471, null, 8]] },
    { id: "8A", digit: 8, rect: [1238, 293, 274, 120], gaps: [8018, 8046, 8074, 8150], cols: [[8001, 8009, 8], [8017, 8023, 6], [8029, 8037, 8], [8045, 8051, 6], [8057, 8065, 8], [8073, 8079, 6], [8085, 8093, 8], [8101, 8109, 8], [8117, 8125, 8], [8133, 8141, 8], [8149, 8156, 7], [8163, 8171, 8], [8179, 8187, 8], [8195, 8203, 8], [8211, 8219, 8], [8227, 8235, 8], [8243, null, 8]] },
  ];
  const HALL = [[135, 372, 1385, 288], [1238, 150, 282, 270]]; // floor outline: main hall and the 8A/8B annex
  const EXACT = new Map(); // board number -> {block, col, row, side}
  for (const b of BLOCKS) (b.cols || []).forEach(([l, r, n], c) => {
    for (let k = 0; k < n; k++) { EXACT.set(l + k, { b, c, row: n - 1 - k, side: 0 }); if (r != null) EXACT.set(r + k, { b, c, row: k, side: 1 }); }
  });

  const overview = (hit, dot) => { // dot: the board's relative spot {b, fx, fy} inside its block
    const label = (b) => { const [x, y, w, h] = b.rect; return `<text x="${x + w / 2}" y="${y + h / 2}" class="hl-t${hit.has(b) ? " on" : ""}">${b.id}</text>`; };
    return `<svg viewBox="110 130 1430 550" class="hall-ov" role="img" aria-label="Poster hall">` +
      HALL.map(([x, y, w, h]) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" class="hl-hall"/>`).join("") +
      BLOCKS.map((b) => `<rect x="${b.rect[0]}" y="${b.rect[1]}" width="${b.rect[2]}" height="${b.rect[3]}" class="hl-b${hit.has(b) ? " on" : ""}"/>`).join("") +
      BLOCKS.map(label).join("") +
      (dot ? `<circle cx="${dot.b.rect[0] + dot.fx * dot.b.rect[2]}" cy="${dot.b.rect[1] + dot.fy * dot.b.rect[3]}" r="11" class="hl-dot"/>` : "") + `</svg>`;
  };

  // Columns are bottom-aligned, as on the floor plan; a 4th entry in a column puts it in a separate stacked group (block 7).
  const detail = (num, at) => {
    const W = 60, H = 30, S = 18, top = 6, GAP = 16; // S: board width; the spine between the two sides is the poster separator
    const groups = [];
    at.b.cols.forEach((col) => (groups[col[3] || 0] = groups[col[3] || 0] || []).push(col));
    const wide = Math.max(...groups.map((g) => g.length)) * W + 10;
    let y = top, body = "", spot = null;
    for (const g of groups) {
      const rows = Math.max(...g.map((c) => c[2]));
      g.forEach(([l, r, n], c) => {
        const x = 5 + c * W, gv = (at.b.gaps || []).find((v) => (v >= l && v < l + n) || (r != null && v >= r && v < r + n));
        const gr = gv == null ? n : gv < l + n ? n - 1 - (gv - l) : gv - r, gap = gv == null ? 0 : rows - n, y0 = y + (rows - n - gap) * H;
        const at_ = (row) => y0 + (row + (row >= gr ? gap : 0)) * H; // rows from gr down sit below the gap
        for (let k = 0; k < n; k++) for (const [v, side, row] of [[l + k, 0, n - 1 - k], [r + k, 1, k]]) {
          if (r == null && side) continue;
          const on = v === num, bx = x + side * (S + 2), by = at_(row), cx = bx + S / 2, cy = by + (H - 2) / 2;
          if (on) spot = { x: cx, y: cy };
          body += `<rect x="${bx}" y="${by}" width="${S}" height="${H - 2}" class="hl-s${on ? " on" : ""}"/>` +
            `<text x="${cx}" y="${cy}" transform="rotate(-90 ${cx} ${cy})" class="hl-n${on ? " on" : ""}">${v}</text>`;
        }
        if (gap) body += `<rect x="${x}" y="${y0 + gr * H}" width="${2 * S + 2}" height="${gap * H - 2}" class="hl-gap"/>`;
        body += `<line x1="${x + S + 1}" y1="${y0 - 3}" x2="${x + S + 1}" y2="${y0 + (n + gap) * H + 1}" class="hl-sep"/>`;
      });
      y += rows * H + GAP;
    }
    const h = y - GAP + top;
    const html = `<div class="hall-scroll"><svg viewBox="0 0 ${wide} ${h}" width="${wide}" height="${h}" class="hall-detail" role="img" aria-label="Boards in poster block ${at.b.id}">${body}</svg></div>`;
    return { html, fx: spot.x / wide, fy: spot.y / h };
  };

  window.hallMap = function (board) {
    const num = parseInt(board, 10), at = EXACT.get(num);
    if (!num) return "";
    if (at) { const d = detail(num, at);
      return `<p class="sub">Board ${num} is in poster block <b>${at.b.id}</b>, at the red dot and in the highlighted spot below.</p>${overview(new Set([at.b]), { b: at.b, fx: d.fx, fy: d.fy })}${d.html}`; }
    const hit = BLOCKS.filter((b) => b.digit === Math.floor(num / 1000));
    return `<p class="sub">Board ${num} is in the highlighted poster block${hit.length > 1 ? "s (the A and B rows are not split by number yet)" : ""}.</p>${overview(new Set(hit))}`;
  };
})();
