"""Builds the same 3-page report as LaTeX (report/TinyCD_report.tex) and Word (report/TinyCD_report.docx)
from one content definition, so the two versions cannot drift apart.   python report/build_report.py

Inline markup in the text: **bold**, *italic*, <https://url> (printed as a plain URL).
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"

TITLE = "TinyCD: spotting changes between two photos with a very small neural network"
AUTHORS = "Shresth Keshari"
SUBTITLE = "CE712 Mini Project report"

REPO = "https://github.com/shresth-keshari/tinycd-demo"
DEMO = "https://shresth-keshari.github.io/tinycd-demo/"

CONTENT = [
    ("h", "1. What is TinyCD?"),
    ("p", "**Change detection** means comparing two photos of the same place taken at different times (for example an aerial "
          "photo from 2015 and another from 2020) and marking the pixels where something really changed, such as a new "
          "building. The answer is a black-and-white map: white where there is change, black everywhere else."),
    ("p", "This is harder than subtracting one photo from the other. Light, season, shadows and small shifts of the camera "
          "also change the pixels, but they are not real changes. Real changes are also rare: in the test set we use, "
          "only about 5% of the pixels belong to a change."),
    ("p", "Good models for this task are usually big. BIT has about 3 million numbers to learn (its *parameters*) and "
          "ChangeFormer about 41 million. **TinyCD** (Codegoni, Lombardi and Ferrari, *Neural Computing and Applications*, "
          "2023) reports accuracy as good as or better than such models with only **285,128** parameters. That is about 10 "
          "times fewer than BIT and 140 times fewer than ChangeFormer. A small model needs less memory and computing power, "
          "which matters when pictures must be processed on cheap or on-board hardware."),
    ("p", "**What we did.** We did not train anything. We wrote the TinyCD network ourselves from the paper, loaded the "
          "weights that the authors released, and checked whether it behaves as the paper says. Section 2 explains the "
          "model in plain words; Section 3 shows what the replication demonstrates."),

    ("h", "2. How TinyCD works"),
    ("fig", "architecture.png", 0.92,
     "**Figure 1.** TinyCD. Both photos pass through the same small feature extractor (top row). At each size a *Mix* block "
     "compares the \"before\" and \"after\" features (middle row). Three of these also produce a one-number-per-pixel "
     "*mask* (dashed arrows) that guides the decoder (bottom row), which grows the picture back to full size."),
    ("p", "The model is a \"U-shaped\" network with four simple ideas:"),
    ("numbered", [
        "**One feature extractor for both photos.** Each photo goes through the same network (a \"Siamese\" design), the first "
        "four blocks of EfficientNet-B4. It turns the photo into compact maps of edges, textures and shapes at three sizes "
        "(128, 64 and 32 pixels wide for a 256-pixel photo). Using the same weights for both photos means a feature means "
        "the same thing before and after.",
        "**Mix: a cheap way to compare.** Common ways to compare are to subtract the features or to stack them and apply a "
        "large filter. Mix instead interleaves the channels (before-1, after-1, before-2, after-2, and so on) and gives each "
        "pair its own tiny filter. Each filter therefore sees one feature as it was before and as it is after, and only "
        "has to learn \"did this change?\". This costs very few parameters.",
        "**Masks: tell the decoder where to look.** At full photo size and at the 128 and 64 sizes, Mix is followed by a tiny "
        "per-pixel network that outputs one number per pixel: how likely it is that something changed here. The decoder "
        "multiplies its own features by these masks, so places that did not change are quieted down.",
        "**A small decoder.** Three steps each double the size, multiply by a mask, and apply a light two-step filter "
        "(a depth-wise separable convolution). A last tiny per-pixel network gives the probability of change; we call "
        "a pixel changed when it is at least 0.5.",
    ]),
    ("table", ["Part", "Parameters", "Share"],
     [["Feature extractor (EfficientNet-B4 blocks 0-3)", "269,362", "94.5%"],
      ["Mix and masks (4 scales)", "3,357", "1.2%"],
      ["Decoder (3 Up blocks)", "11,734", "4.1%"],
      ["Per-pixel classifier", "675", "0.2%"],
      ["**Total**", "**285,128**", "**100%**"]],
     "**Table 1.** Where TinyCD's parameters are (counted from our implementation). The parts that do the comparing "
     "(Mix, masks, classifier) hold only 4,032 parameters, 1.4% of the model.", "lrr"),

    ("h", "3. What the replication shows"),
    ("p", "We evaluated on **LEVIR-CD**: 2,048 test pairs of 256 x 256 aerial photos (0.5 m per pixel) in which people marked "
          "new or demolished buildings. Every model used the same 0.5 threshold and the weights released by its authors. "
          "BIT and ChangeFormer were run with their official code for comparison. Each point below is a claim we checked."),
    ("p", "**(a) Our network is the authors' network.** The released weights load into our code with strict matching: every "
          "name and shape agrees, and the total is 285,128 parameters, as in the paper. We also rebuilt the paper's variants "
          "(Mix replaced by subtraction or by stacking, no final classifier): the full model, seven variants and the B3 "
          "backbone match the paper's counts exactly. The B0, B1 and B2 backbones differ from the paper by 272 parameters "
          "(about 0.2%); we did not find the cause."),
    ("table", ["Variant", "Our count", "Paper"],
     [["Full TinyCD", "285,128", "285,128"],
      ["Mix replaced by subtraction (everywhere)", "282,939", "282,939"],
      ["Mix replaced by stacking + filter (everywhere)", "368,468", "368,468"],
      ["No final classifier", "282,438", "282,438"],
      ["Backbone B3 (instead of B4)", "178,716", "178,716"],
      ["Backbone B2", "147,768", "148,040"],
      ["Backbone B1", "121,820", "122,092"],
      ["Backbone B0", "79,208", "79,480"]],
     "**Table 2.** Parameter counts of the paper's variants, rebuilt by us (a selection of the 13 we checked). Only the full "
     "model has released weights, so only counts, not accuracies, are compared.", "lrr"),
    ("p", "**(b) The accuracy is as reported.** *F1* is one score from 0 to 100 that rewards finding real changes and not "
          "raising false alarms. TinyCD scores 91.07, against 91.05 in the paper (Table 3). ChangeFormer matches its paper "
          "(90.40). BIT scores 89.94, higher than the 89.31 it reports; we do not know why and report it as measured. "
          "To see whether TinyCD's lead is real, we resampled the test images 2,000 times: it is ahead of BIT by 1.12 points "
          "(95% interval 0.69 to 1.57) and ahead of ChangeFormer by 0.67 (0.29 to 1.00). So TinyCD is at least as accurate "
          "here, but not dramatically better; the point is the same accuracy at a tiny size."),
    ("table", ["Model", "F1 % (95% interval)", "Paper F1", "Parameters", "GFLOPs", "CPU s / pair"],
     [["**TinyCD**", "**91.07** (90.63-91.45)", "91.05", "**285,128**", "**1.46**", "**0.19**"],
      ["BIT", "89.94 (89.35-90.53)", "89.31", "3,037,026", "8.76", "0.27"],
      ["ChangeFormer V6", "90.40 (89.82-90.93)", "90.40", "41,026,674", "138.80", "2.45"]],
     "**Table 3.** LEVIR-CD test set (2,048 pairs), threshold 0.5, released weights. GFLOPs: billions of operations for one "
     "256 x 256 pair. CPU time: median for one pair on our laptop (batch of 1). BIT's checkpoint holds 11.9 M numbers; "
     "3.04 M are used at test time.", "lrrrrr"),
    ("p", "**(c) It is much cheaper to run.** TinyCD needs 1.46 billion operations per pair, against 8.76 for BIT and 138.8 "
          "for ChangeFormer. On our laptop CPU it takes 0.19 s per pair, against 0.27 s and 2.45 s. (BIT is only 1.4 times "
          "slower despite 6 times more operations: running time depends on more than the operation count.)"),
    ("pagebreak",),
    ("p", "**(d) We can see it work, and fail.** Figure 2 shows TinyCD finding a large and a small new building. It also "
          "shows a miss and a false alarm. Over the whole test set it finds 89.2% of the 8,673 separate changed regions "
          "(BIT 88.4%, ChangeFormer 88.7%)."),
    ("fig", "qualitative.png", 0.86,
     "**Figure 2.** Four test pairs (before, after, and TinyCD's answer in red; the human label is outlined in cyan). "
     "Left to right: a large new building, a small new house, a building TinyCD misses, and a false alarm on roofs that "
     "did not change."),
    ("p", "**(e) The masks matter.** As a probe, we switched off one mask at a time in the released model (by setting it to "
          "1). Without the 64-size mask, F1 falls to 82.1; without the 128-size mask, to 90.1; without the full-size "
          "mask, to 0.3. This shows the trained network depends on its masks. It is not a retrained ablation, so it does not "
          "show how a network trained without masks would do."),
    ("p", "**What we did not show.** We did no training, so the paper's ablation accuracies and results on other datasets are "
          "not re-measured. TinyCD was trained on 0.5 m building changes; it should not be expected to work on very "
          "different resolutions or satellites. All timings are from one laptop."),
    ("h", "4. Summary"),
    ("bullets", [
        "TinyCD is small because it keeps only the first four blocks of a compact feature extractor and does the comparing "
        "with very cheap blocks (Mix and masks).",
        "With the authors' weights, our implementation reproduces the reported accuracy on LEVIR-CD (F1 91.07 against 91.05) "
        "with about 10 times fewer parameters than BIT and 140 times fewer than ChangeFormer, and about 6 and 95 times "
        "fewer operations.",
        "It is not perfect (it misses some buildings and raises some false alarms), and our checks cover one dataset "
        "and no retraining.",
    ]),
    ("p", "**Try it.** Live demo in the browser (no install): <" + DEMO + ">. Code, Colab notebook and a one-step run script: "
          "<" + REPO + ">. The demo runs the real TinyCD model on sample pairs or on your own pair of photos."),
]

# --------------------------------------------------------------------------------------------- LaTeX
TEX_MAP = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$"}


def tex_quotes(s: str) -> str:
    out, open_q = [], True
    for ch in s:
        if ch == '"':
            out.append("``" if open_q else "''")
            open_q = not open_q
        else:
            out.append(ch)
    return "".join(out)


def tex(s: str) -> str:
    s = tex_quotes(s)
    urls = []

    def keep(m):
        urls.append(m.group(1))
        return f"\x00{len(urls) - 1}\x00"
    s = re.sub(r"<(https?://[^>]+)>", keep, s)
    s = "".join(TEX_MAP.get(c, c) for c in s)
    s = s.replace(" x ", r" $\times$ ")
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"\*(.+?)\*", r"\\emph{\1}", s)
    return re.sub(r"\x00(\d+)\x00", lambda m: r"\url{" + urls[int(m.group(1))] + "}", s)


def build_tex() -> str:
    o = [r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[utf8]{inputenc}
\usepackage[margin=1.8cm,top=1.5cm,bottom=1.5cm]{geometry}
\usepackage{graphicx,booktabs,enumitem,url,caption,titlesec}
\usepackage[colorlinks=true,urlcolor=blue,linkcolor=black]{hyperref}
\captionsetup{font=small,labelsep=none,skip=3pt}
\setlength{\parindent}{0pt}\setlength{\parskip}{3.5pt}
\titleformat{\section}{\large\bfseries}{}{0pt}{}
\titlespacing*{\section}{0pt}{7pt}{3pt}
\setlist{nosep,leftmargin=1.4em,itemsep=2pt}
\pagestyle{empty}
\begin{document}
\begin{center}
{\Large\bfseries """ + tex(TITLE) + r"""}\\[3pt]
""" + tex(AUTHORS) + r" \quad|\quad " + tex(SUBTITLE) + r"""
\end{center}
\vspace{-4pt}
"""]
    for blk in CONTENT:
        kind = blk[0]
        if kind == "h":
            o.append(r"\section*{" + tex(blk[1]) + "}")
        elif kind == "p":
            o.append(tex(blk[1]) + "\n")
        elif kind == "pagebreak":
            o.append(r"\clearpage")
        elif kind == "bullets":
            o.append(r"\begin{itemize}" + "".join(r"\item " + tex(t) + "\n" for t in blk[1]) + r"\end{itemize}")
        elif kind == "numbered":
            o.append(r"\begin{enumerate}" + "".join(r"\item " + tex(t) + "\n" for t in blk[1]) + r"\end{enumerate}")
        elif kind == "fig":
            _, name, w, cap = blk
            o.append(r"\begin{figure}[h!]\centering\includegraphics[width=" + f"{w}" + r"\linewidth]{figures/" + name +
                     r"}\caption*{" + tex(cap) + r"}\end{figure}")
        elif kind == "table":
            _, head, rows, cap, align = blk
            o.append(r"\begin{table}[h!]\centering\small" + "\n" + r"\begin{tabular}{" + align + "}\n\\toprule\n" +
                     " & ".join(tex(h) for h in head) + " \\\\\n\\midrule\n" +
                     "".join(" & ".join(tex(c) for c in r) + " \\\\\n" for r in rows) + "\\bottomrule\n\\end{tabular}\n" +
                     r"\caption*{" + tex(cap) + r"}\end{table}")
    o.append(r"\end{document}")
    return "\n".join(o)


# --------------------------------------------------------------------------------------------- Word
def build_docx(path: Path):
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(1.8)
    sec.top_margin = sec.bottom_margin = Cm(1.5)
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    st.paragraph_format.space_after, st.paragraph_format.space_before = Pt(3.5), Pt(0)
    st.paragraph_format.line_spacing = 1.0

    def runs(par, text, size=None):
        for tok in re.split(r"(\*\*.+?\*\*|\*.+?\*)", text):
            if not tok:
                continue
            if tok.startswith("**"):
                r = par.add_run(tok[2:-2])
                r.bold = True
            elif tok.startswith("*"):
                r = par.add_run(tok[1:-1])
                r.italic = True
            else:
                r = par.add_run(tok)
            if size:
                r.font.size = Pt(size)

    def clean(t):
        t = re.sub(r"<(https?://[^>]+)>", r"\1", t)
        out, op = [], True
        for ch in t:
            if ch == '"':
                out.append("\u201c" if op else "\u201d")
                op = not op
            else:
                out.append(ch)
        return "".join(out).replace(" x ", " \u00d7 ")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(TITLE)
    r.bold, r.font.size = True, Pt(15)
    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.add_run(f"{AUTHORS}  |  {SUBTITLE}")

    def shade(cell, hexcol):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hexcol)
        tcPr.append(shd)

    for blk in CONTENT:
        kind = blk[0]
        if kind == "h":
            par = doc.add_paragraph()
            par.paragraph_format.space_before, par.paragraph_format.space_after = Pt(7), Pt(3)
            par.paragraph_format.keep_with_next = True
            r = par.add_run(blk[1])
            r.bold, r.font.size = True, Pt(13)
        elif kind == "p":
            runs(doc.add_paragraph(), clean(blk[1]))
        elif kind == "pagebreak":
            doc.add_page_break()
        elif kind == "bullets":
            for t in blk[1]:
                par = doc.add_paragraph()
                par.paragraph_format.left_indent = Cm(0.6)
                par.paragraph_format.first_line_indent = Cm(-0.6)
                par.paragraph_format.space_after = Pt(2)
                par.add_run("•  ")
                runs(par, clean(t))
        elif kind == "numbered":
            for i, t in enumerate(blk[1], 1):
                par = doc.add_paragraph()
                par.paragraph_format.left_indent = Cm(0.6)
                par.paragraph_format.first_line_indent = Cm(-0.6)
                par.paragraph_format.space_after = Pt(2)
                par.add_run(f"{i}.  ")
                runs(par, clean(t))
        elif kind == "fig":
            _, name, w, cap = blk
            par = doc.add_paragraph()
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            par.paragraph_format.keep_with_next = True
            par.add_run().add_picture(str(FIG / name), width=Cm(17.4 * w))
            c = doc.add_paragraph()
            runs(c, clean(cap), size=9)
        elif kind == "table":
            _, head, rows, cap, align = blk
            t = doc.add_table(rows=1 + len(rows), cols=len(head))
            t.style = "Table Grid"
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            for j, h in enumerate(head):
                cell = t.rows[0].cells[j]
                cell.text = ""
                runs(cell.paragraphs[0], "**" + h + "**", size=9.5)
                shade(cell, "E6E8EB")
            for i, row in enumerate(rows, 1):
                for j, v in enumerate(row):
                    cell = t.rows[i].cells[j]
                    cell.text = ""
                    runs(cell.paragraphs[0], clean(v), size=9.5)
            t.autofit = False
            lens = [max(len(re.sub(r"\*", "", str(r_[j]))) for r_ in [head] + rows) + 3 for j in range(len(head))]
            total = min(17.4, 0.27 * sum(lens))
            for row in t.rows:
                for j, cell in enumerate(row.cells):
                    cell.width = Cm(total * lens[j] / sum(lens))
            for row in t.rows:
                for j, cell in enumerate(row.cells):
                    for par in cell.paragraphs:
                        par.paragraph_format.space_after = Pt(0)
                        par.alignment = WD_ALIGN_PARAGRAPH.LEFT if align[j] == "l" else WD_ALIGN_PARAGRAPH.RIGHT
            c = doc.add_paragraph()
            c.paragraph_format.space_before = Pt(3)
            runs(c, clean(cap), size=9)
    doc.save(path)


if __name__ == "__main__":
    (HERE / "TinyCD_report.tex").write_text(build_tex(), encoding="utf-8")
    build_docx(HERE / "TinyCD_report.docx")
    print("wrote TinyCD_report.tex and TinyCD_report.docx")
