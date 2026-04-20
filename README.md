# Book Chapter Summarizer

A pipeline that turns a messy book PDF into clean, chapter-by-chapter Markdown and summaries. Given any PDF, it figures out where each chapter actually starts, extracts the text chapter-by-chapter, and produces long and short summaries for each one.

Built on top of Google's Gemini API (vision + text). Works on scanned books, OCR'd books, double-page spreads, books with eccentric front matter, and books whose printed page numbers bear no relation to their PDF page indices.

```
input : some_book.pdf
output: output/some_book_20260420_140312/
          chapters_full_data.json
          chapters_markdown/
            01 - Introduction.md
            02 - The Argument Begins.md
            ...
          long_summaries.md
          short_summaries.md
          book_data_with_text.json
          raw_toc_attempt_1.json
          process_log_*.txt
```

## Table of contents

- [Why this problem is harder than it looks](#why-this-problem-is-harder-than-it-looks)
- [Why a pure algorithmic approach is almost impossible](#why-a-pure-algorithmic-approach-is-almost-impossible)
- [Why a pure LLM approach is also wrong](#why-a-pure-llm-approach-is-also-wrong)
- [How this pipeline solves it](#how-this-pipeline-solves-it)
- [The seven steps, in detail](#the-seven-steps-in-detail)
- [Installation](#installation)
- [Usage](#usage)
- [Output format](#output-format)
- [Cost and model selection](#cost-and-model-selection)
- [Caching and resume](#caching-and-resume)
- [Known limitations](#known-limitations)
- [License](#license)

## Why this problem is harder than it looks

"Split this book PDF into chapters" sounds like a trivial job. It is not. The goal is: for each chapter in the Table of Contents, find the exact PDF page where that chapter's body actually starts, and extract only that chapter's text.

Every piece of that is contaminated:

1. **Printed page numbers almost never match PDF page indices.** A book whose "page 1" is the first chapter typically has 10 to 40 pages of front matter in the PDF before it: cover, title page, copyright page, dedication, epigraph, table of contents, foreword, preface, acknowledgments. So printed "page 1" can be PDF page 12, 18, 34, or anything else. The offset varies per book and there's no metadata that tells you what it is.

2. **Scans use double page layouts.** Many scanned books show two printed pages on a single PDF page. The mapping then becomes `printed_page = 2 * pdf_index + c` instead of `printed_page = pdf_index + c`, and the script has to figure out which.

3. **Front matter uses roman numerals.** Pages i through xxiv (or similar) appear before arabic numbers begin. Treating "xii" as 12 is wrong. Treating "xii" as "ignore" is also wrong because it tells you where arabic numbering starts.

4. **TOCs lie, or get lazy, or split across pages.** Real-world tables of contents misalign, skip numbers, put dots of varying density between title and page, indent subchapters ambiguously, span two or three PDF pages, or just give chapter titles with no numbers at all.

5. **Page numbers in running headers can appear anywhere on the page.** Top-left, top-right, centered bottom, tucked into a margin next to the chapter title, or not present at all on chapter-start pages (which traditionally drop the header).

6. **Other numbers on the page mimic page numbers.** Footnote references, equation numbers, figure numbers, years of publication in citations, and "Chapter 12" headings are all digits near the margins.

7. **Chapter start pages often have non-standard layouts.** A chapter opener may have the title halfway down the page in a decorative font, with no header, no page number, and a drop cap that confuses OCR. An algorithm looking for a page number header finds nothing there.

8. **Books don't agree on what "chapter" means.** Edited volumes have chapters by different authors. Handbooks have parts that contain chapters. Textbooks nest sections three deep. Philosophy monographs sometimes have one 400-page "chapter." An extractor needs to operate at the level the user cares about.

   So "find the start page of chapter X" is not a string-matching problem. It is an inference problem over a noisy, ambiguous, multimodal input.

## Why a pure algorithmic approach is almost impossible

Consider the obvious algorithmic pipeline and notice where each step breaks:

- **"Read the TOC, get title and page, then jump to that page."** This fails at step 1, because the TOC page numbers are printed page numbers, and you do not know the offset between printed pages and PDF indices. Offsets vary from 0 to 50+.

- **"Compute the offset from the first chapter's printed page number vs. where it actually appears."** You do not know where it actually appears; that is what you are trying to compute.

- **"Read the header or footer of each page and grab the number."** Chapter-opener pages typically hide the header. Multi-column layouts put numbers in column margins. Front matter uses roman numerals. Scanned books have OCR errors like "l" vs "1" vs "I" or "O" vs "0". Even good OCR leaks in equation numbers and footnote markers next to the real page number.

- **"Regex for the page number on each page."** You end up with 10 to 30 candidate numbers per page. Which one is the real page number? On any individual page, impossible to decide without context.

- **"Use PDF bookmarks."** Many PDFs do not have bookmarks. Many have them but they point at the wrong places, especially in scans. OCR'd books from archives almost never have usable bookmarks.

- **"Use TOC dotted-leader detection in a layout-aware parser."** This works on a subset of well-behaved PDFs and catastrophically fails on the rest: multi-column TOCs, TOCs with subchapters indented, TOCs rendered as images, TOCs split across pages, TOCs that use chapter numbers but no page numbers, TOCs with non-Latin scripts.

- **"Train a classifier."** With what labeled data? Every book is laid out differently. A classifier that gets 95% accuracy on this page is not good enough, because a single wrong page index for one chapter silently corrupts that chapter's extracted text and every downstream summary.

  You can build a heuristic pipeline that works on 60 or 70 percent of books and fails ungracefully on the rest. The failures are the expensive kind: silent, plausible-looking corruption, where chapter 7's "summary" is actually a summary of the second half of chapter 6 and the first half of chapter 7.

## Why a pure LLM approach is also wrong

The opposite mistake is "just feed the whole book to a big LLM and ask for a chapter breakdown." That fails for different reasons:

- **Hallucinated page numbers.** LLMs will happily invent page numbers that sound plausible and are wrong. They will sometimes output the same page for every chapter because the TOC page was visually ambiguous.
- **Long context is not dense reasoning.** A model with a 1M-token window does not pay equal attention to every page. Chapter headers buried in the middle of the book get missed.
- **Cost.** Sending every page of every book to a flagship model is wasteful when most pages require no reasoning at all.
- **No verification.** An end-to-end LLM has nothing to check its own output against. If it says "Chapter 4 starts on PDF page 87" it either agrees with itself forever or flip-flops, with no mechanism for grounding.

## How this pipeline solves it

The insight behind this tool is that each subproblem has a technique that works, but no single technique works for all of them. So we chain them:

- Use a **vision LLM** for the unstructured visual tasks (reading a TOC image, confirming a chapter heading).

- Use **statistics** to find the global printed-to-PDF page offset (RANSAC on candidate page numbers).

- Use **targeted vision verification** to ground the prediction on the actual rendered page, with a physical trick (a red stamp) to prevent the LLM from hallucinating page indices.

- Use **validation heuristics** to catch silent failure modes early (non-monotonic TOCs, duplicate page clusters, impossibly-low coverage).

- Use a **retry hierarchy** that escalates from a cheap flash model to a heavier pro model only when needed.

- Use a **layout-aware Markdown extractor** (pymupdf4llm) to pull the body text once we know the page ranges.

  The pipeline is robust because it cross-checks: the TOC reader proposes, the linear fit locates, and the vision verifier confirms. Any one of them can be wrong; the combination rarely is.

## The seven steps, in detail

### Step 1: TOC extraction (vision)

Render the first 20 PDF pages to images. Hand them all to a Gemini vision model with a structured-output schema asking for `[{chapter_title, printed_page_number}]`. The prompt explicitly warns the model not to confuse the TOC page's own page number with the per-chapter page numbers (a real failure mode).

Then **validate the output** with conservative heuristics before trusting it:

- At least 50% of chapters must have page numbers.

- Page numbers must be monotonically increasing (with some tolerance).

- No more than 30% of chapters can share the same page number (catches the "all mapped to page 8" hallucination).

- Pages must span at least 20 numeric units for books with 5+ chapters.

- The first chapter cannot start at page 100+ (catches models that read the TOC page number instead of the chapter page numbers).

  If validation fails, retry with a heavier model (`gemini-2.5-pro`). If that also fails, stop. Do not proceed with corrupt TOC data, because every downstream step depends on it.

  Output: a list of `(chapter_title, printed_page_number)` pairs.

### Step 2: Dense page number candidate extraction

For every page in the PDF, use `pdfplumber` to pull all 1-4 digit numbers from the text. This gives a noisy cloud of candidates: the real page number, plus footnote numbers, equation numbers, publication years, etc.

Apply conservative filters that only cut obvious impossibilities:

- Drop numbers larger than `3 * total_pdf_pages + 50`, since no book's printed page count exceeds that.

- Drop numbers whose value differs from the PDF index by more than 60% of the total page count (with a floor of 30), since the real page number and PDF index must correlate roughly linearly.

- In the first 5% of the PDF, drop 1900-2099 values since those are almost certainly copyright dates, not page numbers.

  These filters are intentionally loose. We do not try to pick *the* page number on each page. We trust RANSAC to do that in the next step.

  Output: for each PDF index, a list of candidate printed page numbers.

### Step 3: Robust linear fit (RANSAC)

The mapping from PDF index to printed page number is linear:

    printed_page = slope * pdf_index + intercept

For single-page scans, `slope` is 1.0. For double-page scans (two printed pages per PDF page), slope is 2.0. The intercept captures the front-matter offset.

We fit this with RANSAC:

1. Take every `(pdf_index, candidate_number)` pair from Step 2.

2. Repeatedly pick two random pairs, fit a line through them.

3. Count how many of the other pairs agree with that line (within ±2 pages).

4. Keep the line with the most agreement.

5. Refine it with ordinary least squares on the winning inliers.

   RANSAC is the right tool here because the ratio of signal to noise is low (maybe 1 out of 10 candidates on a given page is the real page number), but the signal is consistent (every real page number is on the same line) while the noise is not. Even with massive noise, the correct line wins by vote.

   Constraining the slope search to `[0.5, 3.0]` rules out nonsense fits and lets the script automatically detect whether the PDF is single-page or double-page.

   Output: `slope`, `intercept`, layout mode (`single` or `double`), and two callable functions for converting between printed page and PDF index.

### Step 4: Map TOC to PDF pages via the fit

For each `(chapter_title, printed_page)` from Step 1, apply the inverse fit to get a predicted PDF index. This is a guess. It is usually within 0 to 3 pages of the truth, but it can be off by more on books with irregular pagination, multi-volume numbering, or mid-book restarts.

Output: the TOC augmented with each chapter's predicted PDF index.

### Step 5: Targeted vision verification (with the red-stamp trick)

For each chapter, render a **small window** of pages (±6 around the predicted index) into images. Stamp a huge red box onto each image that says **"PDF PAGE: N"** where N is the PDF index of that page.

Send this window to the vision model with a prompt that says, in effect: "Find the page where chapter 'X' begins as a prominent chapter heading. When you find it, read the number inside the red box on that page and return it as an integer."

This solves three problems at once:

- **Confirmation.** The model only says "yes, the title appears here" when it visually recognizes the chapter opener, including cases where the opener has a drop cap, a decorative font, or starts partway down the page.

- **No hallucinated indices.** The model does not have to know what PDF index we are on. It just reads the stamped red number back to us. If the model is uncertain, it returns 0 and we fall back.

- **Cheap.** Each chapter verification is ~13 images (±6 pages), not the whole book. With ~30 chapters, that's ~400 image tokens total, not ~40,000.

  If the first attempt fails (returns 0 or a page index outside the window), retry with a wider window (±9 pages) and a heavier model (`gemini-2.5-pro`). If even that fails, keep the fit prediction from Step 4 and mark the chapter as unverified in the log.

  Output: every chapter now has a confirmed `pdf_page_index_0_based` plus a `status` string describing how it was determined (verified by vision, corrected by vision, kept from fit, etc.).

### Step 6: Chapter content extraction

Now that we know every chapter's start page, chapter N's content is the slice `[start_page_N, start_page_{N+1} - 1]`. For the last chapter, we go to the end of the PDF.

Pass those page ranges to `pymupdf4llm.to_markdown()`, which converts the layout-aware PDF content into Markdown while preserving headings, lists, tables, and code blocks. This is much cleaner than raw PDF text extraction, which smashes columns together and loses structure.

Output: each chapter dict now has a `full_text_markdown` field.

### Step 7: Chapter summarization

For each chapter, send the Markdown to the summary model with a structured prompt that:

- Targets a **long summary** at 10% of chapter word count (±20%), organized as `# Long Summary` with themed `##` and `###` subheadings.

- Targets a **short summary** at 1% of chapter word count (±20%), as flat prose.

- Preserves citations in the long summary and removes them from the short one.

- Uses precise Markdown headings (`# Long Summary`, `# Short Summary`) so a downstream script can parse them deterministically.

  After each response, compute the actual word counts. If either is off by more than 20-30% of target, **retry with feedback**: paste the previous attempt back into the prompt with explicit word counts, and ask the model to fix it. Up to 3 retries per chapter.

  Each retry also handles transient network errors (RemoteProtocolError, 503, timeout) with exponential backoff.

  Output: every chapter has `short_summary` and `long_summary` fields, written to `short_summaries.md`, `long_summaries.md`, and the per-chapter Markdown files.

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

Two system dependencies matter:

- **Poppler** (for `pdf2image`, a fallback to PyMuPDF). On Windows, grab a build and put `poppler/bin` on your PATH. On macOS, `brew install poppler`. On Linux, `apt install poppler-utils`. You can usually skip this if PyMuPDF imports cleanly; it's only used as a fallback.

- **No other binaries required.**

  Then set your Gemini API key. Get one free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

```bash
# Option A: environment variable
export GEMINI_API_KEY=your_key_here

# Option B: .env file next to the script
cp .env.example .env
# edit .env
```

## Usage

### Interactive mode (GUI)

Run the script with no arguments:

```bash
python book_chapter_summarizer.py
```

A file dialog appears to pick a PDF, then a configuration dialog shows the estimated cost for each Gemini model for each pipeline step. Choose models, click Process.

### Batch mode (one PDF)

```bash
python book_chapter_summarizer.py path/to/book.pdf
```

Uses default models (`gemini-3-flash-preview` for all three steps).

### Batch mode (many PDFs)

Drop PDFs into a `pdfs/` folder next to the script and run the batch runner:

```bash
# Windows
run_batch.bat

# Linux / macOS
./run_batch.sh
```

Or pass paths explicitly:

```bash
./run_batch.sh one.pdf two.pdf three.pdf
```

Output goes to `./output/[BookName_timestamp]/` by default. Override with the `OUTPUT_BASE_DIR` environment variable.

### Overriding models

Pass a comma-separated `toc_model,visual_model,summary_model` triple as the second argument:

```bash
python book_chapter_summarizer.py book.pdf gemini-3.1-flash-lite-preview,gemini-3-flash-preview,gemini-3.1-pro-preview
```

Available model ids are listed at the top of the script under `AVAILABLE_MODELS`.

## Output format

Each processed book produces a folder:

| File                              | Contents                                                     |
| --------------------------------- | ------------------------------------------------------------ |
| `long_summaries.md`               | All long summaries, one chapter per section                  |
| `short_summaries.md`              | All short summaries, one chapter per section                 |
| `chapters_full_data.json`         | Structured dump: title, page, full markdown, both summaries  |
| `book_data_with_text.json`        | Intermediate: chapters with text but pre-summary (used for resume) |
| `chapters_markdown/NN - Title.md` | One Markdown file per chapter, full text                     |
| `raw_toc_attempt_1.json`          | Diagnostic: what the TOC model returned, before validation   |
| `raw_toc_attempt_2.json`          | (Only if attempt 1 failed validation) Same from the fallback model |
| `process_log_*.txt`               | Full pipeline log with token counts and cost breakdown       |

## Cost and model selection

The interactive GUI prices every model against the actual book before you start, so you can pick the cheapest combo that fits. For most books under 300 pages, running `gemini-3-flash-preview` across all three steps costs a few cents to a few dimes.

The three steps have very different cost profiles:

- **TOC**: ~20 pages of images. Tiny cost. Even on a pro model this is negligible.

- **Vision verification**: proportional to chapter count, ~13 pages each. Still cheap.

- **Summarization**: proportional to book length. This dominates cost. Use flash for non-critical reading, pro if you need the summaries to hold up to close reading.

  The script automatically applies `thinking_level=LOW` to pro models during summarization (keeps them fast without losing much quality for this task).

## Caching and resume

Two mechanisms prevent redoing work:

- **Global TOC cache.** The verified chapter mapping is keyed by a hash of the PDF (size + first and last 512KB). If you rerun the same book, Steps 1-5 are skipped entirely and we go straight to extraction and summarization. Cache lives in `.toc_cache.json` next to the script.
- **Mid-run resume.** After content extraction and before summarization, the script writes `book_data_with_text.json`. If the script dies during summarization (network error, rate limit, you Ctrl-C'd), rerunning picks up where it left off.

## Known limitations

- **Non-Latin scripts** in TOCs work in principle (Gemini handles them fine) but have not been extensively tested.
- **Handwritten or heavily-decorated chapter openers** can defeat vision verification. The pipeline falls back to the statistical fit in this case; log the `status` field to see.
- **Very short books (< 5 chapters)** leave RANSAC with thin data. The script falls back to identity mapping, which works for modern PDFs with correct metadata but will be off for scans.
- **Encrypted PDFs** must be decrypted first (use `qpdf --decrypt` or similar).
- **Scanned PDFs with no OCR** will fail at Step 2 (extractable text ratio < 1%). OCR the PDF first with something like `ocrmypdf`.
- **Summaries are hallucination-prone at the detail level**, as with any LLM summarization. The long summary is organized by theme, not by paragraph-to-paragraph compression, so it can smooth over or misattribute details. Use it as a map, not as a substitute for the chapter.

## License

MIT. See `LICENSE`.
