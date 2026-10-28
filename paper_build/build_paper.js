import fs from "node:fs";
import path from "node:path";
import {
  AlignmentType, Document, Footer, Header, HeadingLevel, ImageRun, Packer, PageNumber,
  Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
  convertInchesToTwip,
} from "docx";

const outputPath = process.argv[2];
if (!outputPath) throw new Error("Usage: node build_paper.js /abs/out.docx [doc.json]");

const BASE = "/mnt/agents/output/果蝇生成器/paper";
const data = JSON.parse(fs.readFileSync(process.argv[3] || path.join(BASE, "paper_doc.json"), "utf8"));
const FIGDIR = path.join(BASE, "figs_submission");

const font = { ascii: "Times New Roman", hAnsi: "Times New Roman", cs: "Times New Roman", eastAsia: "SimSun" };

// 图注前缀 → 图片文件与尺寸（宽 600px ≈ 正文通栏，高按原始宽高比）
const FIGS = {
  "Fig. 1 |": ["paper_fig1_schematic.png", 600, 292],
  "Fig. 2 |": ["paper_fig2_degree.png", 600, 196],
  "Fig. 3 |": ["paper_fig4_ring.png", 600, 196],
  "Fig. 4 |": ["paper_fig5_gamma.png", 600, 208],
  "Fig. 5 |": ["paper_fig7_conserve.png", 600, 407],
  "Extended Data Fig. 1 |": ["paper_fig3_space.png", 600, 196],
  "Extended Data Fig. 2 |": ["paper_fig6_male_port.png", 600, 183],
};

function runsWithCites(text, base = {}) {
  const parts = text.split(/(⟨⟨[^⟩]+⟩⟩|\[\[[^\]]+\]\]|\{_[^}]+\})/g).filter(Boolean);
  return parts.map((part) => {
    const m = part.match(/^⟨⟨([^⟩]+)⟩⟩$/);
    if (m) return new TextRun({ text: m[1], font, size: 24, superScript: true, ...base });
    const b = part.match(/^\[\[([^\]]+)\]\]$/);
    if (b) return new TextRun({ text: b[1], font, size: 24, bold: true, ...base });
    const sb = part.match(/^\{_([^}]+)\}$/);
    if (sb) return new TextRun({ text: sb[1], font, size: 24, subScript: true, ...base });
    return new TextRun({ text: part, font, size: 24, ...base });
  });
}

const para = (children, options = {}) => new Paragraph({ spacing: { after: 120 }, ...options, children: Array.isArray(children) ? children : [children] });

const bodyP = (text) => para(runsWithCites(text), {
  spacing: { after: 0, line: 480 }, alignment: AlignmentType.JUSTIFIED,
  indent: { firstLine: convertInchesToTwip(0.35) },
});
const absP = (text) => para(runsWithCites(text, { bold: true }), { spacing: { after: 0, line: 480 }, alignment: AlignmentType.JUSTIFIED });
const methP = (text) => para(runsWithCites(text), {
  spacing: { after: 60, line: 360 }, alignment: AlignmentType.JUSTIFIED,
  indent: { firstLine: convertInchesToTwip(0.3) },
});
const capP = (text) => para(runsWithCites(text, { size: 22 }), { spacing: { after: 240, line: 276 }, alignment: AlignmentType.JUSTIFIED });
const imgP = (file, w, h) => para(new ImageRun({
  type: "png", data: fs.readFileSync(path.join(FIGDIR, file)),
  transformation: { width: w, height: h },
}), { alignment: AlignmentType.CENTER, spacing: { before: 120, after: 120 } });

const titleP = (text) => para(new TextRun({ text, font, size: 40, bold: true }), { alignment: AlignmentType.CENTER, spacing: { after: 240, before: 120 } });
const authorsP = (text) => para(runsWithCites(text, { size: 28 }), { alignment: AlignmentType.CENTER, spacing: { before: 200, after: 120 } });
const metaP = (text) => para(runsWithCites(text, { size: 20 }), { alignment: AlignmentType.CENTER, spacing: { after: 60 } });
const sititleP = (text) => para(new TextRun({ text, font, size: 36, bold: true }), { alignment: AlignmentType.CENTER, spacing: { after: 200, before: 120 } });
const h1 = (text) => para(new TextRun({ text, font, size: 28, bold: true, color: "000000" }), { heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 120 }, pageBreakBefore: true });
const h2 = (text) => para(new TextRun({ text, font, size: 24, bold: true, color: "000000" }), { heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 80 } });
const h3 = (text) => para(new TextRun({ text, font, size: 24, bold: true, color: "000000" }), { heading: HeadingLevel.HEADING_3, spacing: { before: 200, after: 60 } });

// ---- Table 1 ----
const T1_W = [1700, 1150, 1250, 1100, 1250, 1350, 1200];
const tcell = (text, w, opts = {}, size = 18) => new TableCell({
  children: [para(runsWithCites(text, { size }), { spacing: { after: 0, line: 240 } })],
  margins: { top: 60, bottom: 60, left: 100, right: 100 },
  width: { size: w, type: WidthType.DXA },
  ...opts,
});
function buildTable1(rows) {
  const trs = rows.map((row, ri) => {
    const isHead = ri === 0;
    const shade = isHead ? { shading: { type: ShadingType.CLEAR, fill: "EDEFF2" } } : {};
    const cells = [];
    for (let ci = 0; ci < 7; ci++) {
      const val = row[ci];
      if (ri === 9 && ci === 1) {
        cells.push(tcell(val, T1_W[1] + T1_W[2] + T1_W[3] + T1_W[4], { columnSpan: 4, ...shade }));
        ci = 4;
        continue;
      }
      cells.push(tcell(val, T1_W[ci], shade));
    }
    return new TableRow({ children: cells, tableHeader: isHead });
  });
  return new Table({ width: { size: 9000, type: WidthType.DXA }, columnWidths: T1_W, rows: trs });
}

// ---- Extended Data Table 1 ----
const EDT_W = [3400, 1900, 1900, 1800];
function buildEdt1(rows) {
  const trs = rows.map((row, ri) => {
    const isHead = ri === 0;
    const shade = isHead ? { shading: { type: ShadingType.CLEAR, fill: "EDEFF2" } } : {};
    return new TableRow({
      tableHeader: isHead,
      children: row.map((val, ci) => tcell(val, EDT_W[ci], shade)),
    });
  });
  return new Table({ width: { size: 9000, type: WidthType.DXA }, columnWidths: EDT_W, rows: trs });
}

// ---- Extended Data Table 2 ----
const EDT2_W = [700, 3600, 1700, 1500, 1500];
function buildEdt2(rows) {
  const trs = rows.map((row, ri) => {
    const isHead = ri === 0;
    const shade = isHead ? { shading: { type: ShadingType.CLEAR, fill: "EDEFF2" } } : {};
    return new TableRow({
      tableHeader: isHead,
      children: row.map((val, ci) => tcell(val, EDT2_W[ci], shade)),
    });
  });
  return new Table({ width: { size: 9000, type: WidthType.DXA }, columnWidths: EDT2_W, rows: trs });
}

// ---- generic small table (Methods Table 2, SI tables) ----
function buildFlexTable(item) {
  const W = item.widths;
  const total = W.reduce((a, b) => a + b, 0);
  const trs = item.rows.map((row, ri) => {
    const isHead = ri === 0;
    const shade = isHead ? { shading: { type: ShadingType.CLEAR, fill: "EDEFF2" } } : {};
    return new TableRow({
      tableHeader: isHead,
      children: row.map((val, ci) => tcell(val, W[ci], shade, item.kind === "mctable" ? 18 : 16)),
    });
  });
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: W, rows: trs });
}

const children = [];
for (const item of data.body) {
  switch (item.kind) {
    case "title": children.push(titleP(item.text)); break;
    case "sititle": children.push(sititleP(item.text)); break;
    case "authors": children.push(authorsP(item.text)); break;
    case "meta": children.push(metaP(item.text)); break;
    case "abstract":
      children.push(absP(item.text));
      break;
    case "h1": children.push(h1(item.text)); break;
    case "h2": children.push(h2(item.text)); break;
    case "h3": children.push(h3(item.text)); break;
    case "p": children.push(bodyP(item.text)); break;
    case "refs_anchor": {
      if (data.refs && data.refs.length) {
        children.push(h1("References"));
        for (const r of data.refs) {
          children.push(para([
            new TextRun({ text: `${r.n}. `, font, size: 20 }),
            new TextRun({ text: r.text, font, size: 20 }),
          ], { spacing: { after: 40, line: 240 }, indent: { left: 360, hanging: 360 } }));
        }
      }
      break;
    }
    case "mp": children.push(methP(item.text)); break;
    case "caption": {
      const hit = Object.keys(FIGS).find((k) => item.text.startsWith(k));
      if (hit) children.push(imgP(...FIGS[hit]));
      children.push(capP(item.text));
      if (item.text.startsWith("Extended Data Table 1 |")) children.push(buildEdt1(data.edt1));
      if (item.text.startsWith("Extended Data Table 2 |")) children.push(buildEdt2(data.edt2));
      break;
    }
    case "table1": children.push(buildTable1(item.rows)); break;
    case "mctable":
    case "stable": {
      children.push(capP(item.caption));
      children.push(buildFlexTable(item));
      children.push(para(new TextRun({ text: "", font, size: 24 }), { spacing: { after: 200 } }));
      break;
    }
  }
}

const hasAnchor = data.body.some((b) => b.kind === "refs_anchor");
if (data.refs && data.refs.length && !hasAnchor) {
  children.push(h1("References"));
  for (const r of data.refs) {
    children.push(para([
      new TextRun({ text: `${r.n}. `, font, size: 20 }),
      new TextRun({ text: r.text, font, size: 20 }),
    ], { spacing: { after: 40, line: 240 }, indent: { left: 360, hanging: 360 } }));
  }
}

const docx = new Document({
  features: { updateFields: false },
  sections: [{
    properties: { page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [para(new TextRun({ text: "The wiring program of a fly brain", font, size: 18, italics: true }), { alignment: AlignmentType.RIGHT, spacing: { after: 0 } })] }) },
    footers: { default: new Footer({ children: [para(new TextRun({ children: [PageNumber.CURRENT], font, size: 20 }), { alignment: AlignmentType.CENTER, spacing: { after: 0 } })] }) },
    children,
  }],
});

fs.writeFileSync(outputPath, await Packer.toBuffer(docx));
console.log("WROTE", outputPath);
