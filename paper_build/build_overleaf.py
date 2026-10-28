#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper_doc.json / si_doc.json -> Overleaf LaTeX project (pdfLaTeX-compatible)."""
import json, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path('/mnt/agents/output/果蝇生成器')
OUT = ROOT / 'overleaf'
FIGSRC = ROOT / 'paper' / 'figs_submission'

# ---------- text conversion ----------
UNI = {
    '–': '--', '—': '---', '‘': '`', '’': "'", '“': '``', '”': "''",
    '±': r'$\pm$', '≤': r'$\le$', '≥': r'$\ge$', '≈': r'$\approx$',
    '×': r'$\times$', '→': r'$\to$', '↔': r'$\leftrightarrow$',
    '−': '-', '√': r'$\surd$', '∈': r'$\in$', '°': r'$^\circ$',
    '§': r'\S{}', '†': r'\dag{}', '′': r'$^{\prime}$',
    'µ': r'\textmu{}',
    'Δ': r'$\Delta$', 'β': r'$\beta$', 'γ': r'$\gamma$', 'η': r'$\eta$',
    'θ': r'$\theta$', 'κ': r'$\kappa$', 'λ': r'$\lambda$', 'ρ': r'$\rho$', 'φ': r'$\phi$',
    '·': r'$\cdot$', '≡': r'$\equiv$', 'π': r'$\pi$', '∩': r'$\cap$', 'τ': r'$\tau$',
    'ᵢ': r'$_i$', 'ⱼ': r'$_j$', 'ₛ': r'$_s$', 'ₜ': r'$_t$',
    '¹': r'\textsuperscript{1}', '²': r'\textsuperscript{2}',
    '⁵': r'\textsuperscript{5}', '⁶': r'\textsuperscript{6}', '⁻': r'\textsuperscript{-}',
}
ESC = {'&': r'\&', '%': r'\%', '#': r'\#', '_': r'\_', '$': r'\$',
       '{': r'\{', '}': r'\}', '^': r'\textasciicircum{}', '~': r'$\sim$',
       '\\': r'\textbackslash{}'}

def esc_plain(s):
    out = []
    for c in s:
        if c in ESC:
            out.append(ESC[c])
        elif c in UNI:
            out.append(UNI[c])
        elif c in '\n':
            out.append(' ')
        elif ord(c) > 127:
            out.append(c)  # latin accents (á é ö) pass through: utf8 inputenc + T1
        else:
            out.append(c)
    return ''.join(out)

PH_L, PH_R = '\x01', '\x02'
def tex(s):
    """Convert markup text to LaTeX."""
    holds = []
    def hold(x):
        holds.append(x)
        return f'{PH_L}{len(holds)-1}{PH_R}'
    # citations ⟨⟨1,3⟩⟩，支持 en-dash 区间 ⟨⟨13–15⟩⟩（展开为逐条 ref）
    def cite_sub(m):
        nums = []
        for part in m.group(1).split(','):
            part = part.strip()
            rgn = re.match(r'^(\d+)\s*[–—-]\s*(\d+)$', part)
            if rgn:
                a, b = int(rgn.group(1)), int(rgn.group(2))
                nums.extend(range(a, b + 1))
            elif part:
                nums.append(int(part))
        keys = ','.join('ref' + str(n) for n in nums)
        return hold(r'\textsuperscript{\cite{' + keys + '}}')
    s = re.sub(r'⟨⟨([0-9, –—-]+)⟩⟩', cite_sub, s)
    # bold [[...]]
    s = re.sub(r'\[\[(.+?)\]\]', lambda m: hold(r'\textbf{' + esc_plain(m.group(1)) + '}'), s)
    # true subscript {_...}
    s = re.sub(r'\{_(.+?)\}', lambda m: hold(r'$_{\mathrm{' + re.sub(r'[^A-Za-z0-9]', '', m.group(1)) + '}}$'), s)
    s = esc_plain(s)
    # restore placeholders
    def unhold(m):
        return holds[int(m.group(1))]
    return re.sub(re.escape(PH_L) + r'(\d+)' + re.escape(PH_R), unhold, s)

# ---------- document skeleton ----------
PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{textcomp}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage[margin=1in]{geometry}
\usepackage{setspace}
\usepackage{lineno}
\usepackage{longtable,array}
\usepackage{caption}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\doublespacing
\linenumbers
"""

FIGMAP = {  # legend prefix -> figure file
    'Fig. 1 |': 'paper_fig1_schematic.png',
    'Fig. 2 |': 'paper_fig2_degree.png',
    'Fig. 3 |': 'paper_fig4_ring.png',
    'Fig. 4 |': 'paper_fig5_gamma.png',
    'Fig. 5 |': 'paper_fig7_conserve.png',
    'Extended Data Fig. 1 |': 'paper_fig3_space.png',
    'Extended Data Fig. 2 |': 'paper_fig6_male_port.png',
}

def bibitem(text):
    # wrap trailing URL in \url{}
    m = re.search(r'(https?://\S+)\s*$', text)
    if m:
        body = esc_plain(text[:m.start()].rstrip())
        return body + r' \url{' + m.group(1) + '}'
    return esc_plain(text)

def tabular(rows, cols_spec, footnotesize=False):
    lines = []
    if footnotesize:
        lines.append(r'\footnotesize')
        lines.append(r'\setlength{\tabcolsep}{3pt}')
    lines.append(r'\begin{tabular}{' + cols_spec + '}')
    lines.append(r'\hline')
    for ri, row in enumerate(rows):
        cells = [tex(str(c)) for c in row]
        lines.append(' & '.join(cells) + r' \\')
        if ri == 0:
            lines.append(r'\hline')
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    return '\n'.join(lines)

def pspec(fracs):
    return ''.join('p{%.2f\\textwidth}' % f for f in fracs)

# ---------- main text ----------
def build_main(dj):
    body, refs, edt1 = dj['body'], dj['refs'], dj['edt1']
    L = [PREAMBLE]
    L.append(r'\title{\textbf{The wiring program of a fly brain}}')
    L.append(r'\author{Xining Xun\textsuperscript{1}, Hang Fan\textsuperscript{2,*}, Tianyi Hao\textsuperscript{1}\\[4pt]'
             r'\footnotesize \textsuperscript{1}Tsingjiao Information Science (Beijing) Co., Ltd., Beijing, China\\'
             r'\footnotesize \textsuperscript{2}School of Economics and Management, North China Electric Power University, China\\[4pt]'
             r'\footnotesize *Corresponding author: Hang Fan, ORCID: 0000-0002-8308-9332'
             r' (\url{https://orcid.org/0000-0002-8308-9332}), fanhang@ncepu.edu.cn.\\'
             r'\footnotesize X.X. ORCID: 0009-0001-8492-4758 (\url{https://orcid.org/0009-0001-8492-4758}).}')
    L.append(r'\date{}')
    L.append(r'\begin{document}')
    L.append(r'\maketitle')
    L.append(r'\begin{abstract}')
    abs_block = next(b for b in body if b.get('kind') == 'abstract')
    L.append(tex(abs_block['text']))
    L.append(r'\end{abstract}')
    L.append('')

    mode = None  # track figure-legend sections
    pending_figs, pending_tabs = [], []
    # 头块 0-4 = title/authors/meta/abstract；正文从首个段落块起
    i = next(n for n, b in enumerate(body) if b.get('kind') == 'abstract') + 1
    while i < len(body):
        b = body[i]
        k, t = b.get('kind'), b.get('text', '')
        if k == 'refs_anchor':
            L.append(r'\begin{thebibliography}{99}')
            for r in refs:
                L.append(r'\bibitem{ref%d} %s' % (r['n'], bibitem(r['text'])))
            L.append(r'\end{thebibliography}')
            L.append('')
            i += 1
            continue
        if k == 'h2':
            L.append(r'\section*{' + tex(t) + '}')
        elif k == 'h1':
            if t == 'Figure legends':
                mode = 'figleg'
            elif t == 'Extended Data legends':
                mode = 'edleg'
            else:
                mode = None
                L.append(r'\section*{' + tex(t) + '}')
        elif k == 'h3':
            L.append(r'\subsection*{' + tex(t) + '}')
        elif k in ('p', 'mp'):
            L.append(tex(t))
            L.append('')
        elif k == 'mctable':
            pending_tabs.append(
                r'\begin{table}[!htb]' + '\n' + r'\centering' + '\n' +
                r'\caption*{' + tex(b['caption']) + '}' + '\n' +
                tabular(b['rows'], 'l' + 'l' * (len(b['rows'][0]) - 1)) + '\n' +
                r'\end{table}')
        elif k == 'table1':
            pass  # emitted after its caption (block 74)
        elif k == 'caption':
            if mode == 'figleg':
                for pref, fn in FIGMAP.items():
                    if t.startswith(pref) and 'Extended' not in pref:
                        pending_figs.append(
                            r'\begin{figure}[p]' + '\n' + r'\centering' + '\n' +
                            r'\includegraphics[width=\textwidth]{figures/' + fn + '}' + '\n' +
                            r'\caption*{' + tex(t) + '}' + '\n' + r'\end{figure}')
                        break
                if t.startswith('Table 1 |'):
                    tb = body[i + 1]
                    assert tb['kind'] == 'table1'
                    pending_tabs.append(
                        r'\begin{table}[p]' + '\n' + r'\centering' + '\n' +
                        r'\caption*{' + tex(t) + '}' + '\n' +
                        tabular(tb['rows'], pspec([0.19, 0.11, 0.13, 0.11, 0.13, 0.13, 0.11]), footnotesize=True) + '\n' +
                        r'\end{table}')
            elif mode == 'edleg':
                if t.startswith('Extended Data Table 1 |'):
                    pending_tabs.append(
                        r'\begin{table}[p]' + '\n' + r'\centering' + '\n' +
                        r'\caption*{' + tex(t) + '}' + '\n' +
                        tabular(edt1, pspec([0.32, 0.19, 0.19, 0.19]), footnotesize=True) + '\n' +
                        r'\end{table}')
                elif t.startswith('Extended Data Table 2 |'):
                    pending_tabs.append(
                        r'\begin{table}[p]' + '\n' + r'\centering' + '\n' +
                        r'\caption*{' + tex(t) + '}' + '\n' +
                        tabular(dj['edt2'], pspec([0.08, 0.38, 0.19, 0.17, 0.18]), footnotesize=True) + '\n' +
                        r'\end{table}')
                else:
                    for pref, fn in FIGMAP.items():
                        if t.startswith(pref):
                            pending_figs.append(
                                r'\begin{figure}[p]' + '\n' + r'\centering' + '\n' +
                                r'\includegraphics[width=\textwidth]{figures/' + fn + '}' + '\n' +
                                r'\caption*{' + tex(t) + '}' + '\n' + r'\end{figure}')
                            break
        i += 1

    # flush tables/figures that belong mid-document (mctable Table 2 stays inline where pending)
    # simpler: Table 2 was appended in order; figleg tables/figures go at the end
    # Separate: pending_tabs already contains Table 2 (inline position) only if encountered before figleg.
    # We appended Table 2 to pending_tabs at its body position -- but L has no marker. Handle: emit Table 2
    # immediately when encountered. Refactor: instead of pending, we should have appended to L. (Table 2 only.)
    # -- handled below by construction order: we emit pending lists at the end; Table 2 moves to end too,
    #    which is acceptable for a submission manuscript (floats at end). Keep it simple.
    L.extend(pending_tabs)
    L.append('')
    L.extend(pending_figs)
    L.append(r'\end{document}')
    return '\n'.join(L)

# ---------- supplement ----------
def build_si(sj):
    sb = sj['body']
    L = [PREAMBLE]
    L.append(r'\title{\textbf{Supplementary Information}\\[6pt]\large The wiring program of a fly brain}')
    L.append(r'\author{Xining Xun\textsuperscript{1}, Hang Fan\textsuperscript{2}, Tianyi Hao\textsuperscript{1}\\[4pt]'
             r'\small \textsuperscript{1}Tsingjiao Information Science (Beijing) Co., Ltd.\\'
             r'\small \textsuperscript{2}School of Economics and Management, North China Electric Power University}')
    L.append(r'\date{}')
    L.append(r'\begin{document}')
    L.append(r'\maketitle')
    for b in sb[3:]:
        k, t = b.get('kind'), b.get('text', '')
        if k == 'h1':
            L.append(r'\section*{' + tex(t) + '}')
        elif k == 'h3':
            L.append(r'\subsection*{' + tex(t) + '}')
        elif k in ('p', 'mp'):
            L.append(tex(t)); L.append('')
        elif k == 'stable':
            cap = tex(re.sub(r'^Supplementary Table \d+ \|\s*', '', b.get('caption', '')))
            rows = b['rows']
            ncol = len(rows[0])
            if ncol == 7:
                spec = pspec([0.03, 0.19, 0.06, 0.19, 0.15, 0.19, 0.04])
            elif ncol == 6:
                spec = pspec([0.03, 0.18, 0.19, 0.19, 0.24, 0.04])
            else:
                spec = 'l' * ncol
            L.append(r'\begingroup\footnotesize')
            L.append(r'\setlength{\tabcolsep}{3pt}')
            L.append(r'\renewcommand{\tablename}{Supplementary Table}')
            L.append(r'\begin{longtable}{' + spec + '}')
            L.append(r'\caption{' + cap + r'}\\')
            L.append(r'\hline')
            for ri, row in enumerate(rows):
                L.append(' & '.join(tex(str(c)) for c in row) + r' \\')
                if ri == 0:
                    L.append(r'\hline\endhead')
            L.append(r'\hline')
            L.append(r'\end{longtable}')
            L.append(r'\endgroup')
    L.append(r'\end{document}')
    return '\n'.join(L)

# ---------- run ----------
def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / 'figures').mkdir(parents=True)
    for f in FIGSRC.glob('*.png'):
        shutil.copy(f, OUT / 'figures' / f.name)
    dj = json.load(open(ROOT / 'paper' / 'paper_doc.json'))
    sj = json.load(open(ROOT / 'paper' / 'si_doc.json'))
    (OUT / 'main.tex').write_text(build_main(dj), encoding='utf-8')
    (OUT / 'supplementary.tex').write_text(build_si(sj), encoding='utf-8')
    print('wrote', OUT)

if __name__ == '__main__':
    main()
