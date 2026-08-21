# School-template compliance contract

## Authoritative source

`reports/Template/MUST_FOLLOW.md` and the accompanying school files are immutable.
Their source hashes were verified before and after creating the V5 working copy:

| File | SHA-256 |
|---|---|
| `MUST_FOLLOW.md` | `b8d9e4a6b19012f8c9c6066926551173933f2e68d5bfc9bacb364ad09b1734c7` |
| `student-number.tex` | `6c3d3a711059f2bb159294d74a6c39ea2c95ecbb00a7fd7e68c56921212918df` |
| `biblio.bib` | `230a804e33d7cd99173e0329d191e359973513038e6b386cce1c26462dccfe79` |
| `USTH-logo.png` | `1771dae63a3e5d837a496dfe33b63e3ef82da51351f6cf0fed3f357a0f234b68` |
| `student-number.pdf` | `99337b081b6c190929882606d856ab1ba45fb214e0265f2440ac099bdb79898a` |

The V5 working copy is `reports/thesis_v5/school_submission/2440059.tex`. It was
initially byte-identical to `student-number.tex`. The logo was copied byte-identically
under lowercase `usth-logo.png` only because the supplied LaTeX source refers to that
case-sensitive filename.

## Non-negotiable format

- `report` document class, 12pt, A4;
- 2.5 cm margins on all sides and zero binding offset;
- one-and-a-half line spacing;
- supplied header, footer, title page, front matter and numeric `biblatex` workflow;
- exactly five main chapters;
- 30-35 pages across Chapters 1-5;
- Introduction 2-3 pages, Related Works 5-7, Contributions 10-12, Experiments 8-10,
  Conclusion 3-4;
- self-contained 150-300-word abstract;
- final PDF named with student ID and smaller than 10 MB;
- prescribed `pdflatex -> biber -> pdflatex -> pdflatex` compilation route.

Subsections may be added inside the mandatory chapters. No standalone extra scientific
chapter may replace or extend the five-chapter skeleton.

## Permitted edits in the separate working copy

- identity, supervisor, title, date and academic-year macros;
- Abstract and optional Acknowledgments placeholder content;
- content inside Chapters 1-5, including appropriate subsections;
- bibliography entries in the working `biblio.bib`.

No source file inside `reports/Template/` may be edited. The source hashes above are a
hard pre-build and pre-delivery gate.

