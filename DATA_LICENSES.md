# Data licenses

This file is the single authoritative record of every dataset used in this
repo, its provenance, and its verified license. Per this project's own
rule ("verify provenance and license before using any new dataset,
document in DATA_LICENSES.md, no exceptions for convenience"), any new
dataset added to this repo going forward must get an entry here before
its data is used for training or evaluation. This consolidates and
formalizes licensing information that, for datasets added earlier in the
project, previously lived only in README.md's "Dataset provenance &
licensing" section (that section still exists and is not contradicted by
this file — this file is the canonical, dedicated record; README.md's
section can be treated as a summary of it).

| Dataset | Used for | Source fetched from | Retrieved | License | Verification method |
|---|---|---|---|---|---|
| SMS Spam Collection | Noul (spam/ham binary) | `data/spam.csv` — bundled with this project from its start | n/a (pre-existing file) | **Not independently verified.** | Long-standing academic benchmark (Almeida & Gómez Hidalgo). This repo has not re-fetched or re-read the authoritative license text from the canonical source (`archive.ics.uci.edu`, blocked by this sandbox's network policy). Treat as research-use-only pending verification. Re-checked 2026-09-24: `archive.ics.uci.edu` and the authors' page (`dt.fee.unicamp.br/~tiago/smsspamcollection`) are both still blocked by the egress proxy, so it remains unverified. |
| BANKING77 | Choice (77-class banking intent) | `https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{train,test}.csv`, `categories.json` | 2026-09-21 | **CC-BY 4.0** | Confirmed by fetching and reading `https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/LICENSE` directly — first-party LICENSE file, read byte for byte, not inferred from the paper or a secondary source. |
| Amazon Fine Food Reviews | Score (review-rating regression) | `data/raw/amazon_food_reviews.csv`, first 10,112 of ~568,454 rows, via a third-party GitHub mirror (Kaggle itself requires login this sandbox cannot provide) | 2026-09-21 | **CC0 (public domain)** | Corroborated by multiple independent sources (the dataset's Kaggle listing under the SNAP/creator account, and a matching Hugging Face mirror, both tagging it CC0-1.0) — a weaker form of verification than BANKING77's since `kaggle.com` itself could not be reached to read the license field directly. First two rows spot-checked against the well-known original dataset's first rows to confirm the mirror is genuine. |
| CLINC150 (`oos-eval`) | Workstream D: new 151-class intent task (150 in-scope intents + explicit out-of-scope class) | `https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json` (data) and `https://raw.githubusercontent.com/clinc/oos-eval/master/LICENSE` (license) | 2026-09-22 | **CC-BY 3.0 (Creative Commons Attribution 3.0 Unported)** | Confirmed by fetching and reading the repo's own `LICENSE` file directly at the URL above — first-party document, read byte for byte (re-verified again on this pass), same verification standard as BANKING77. The file's own header text reads "Creative Commons Legal Code / Attribution 3.0 Unported" — this is the standard CC BY 3.0 Unported legal code (not a NonCommercial or ShareAlike variant; Section 3 grants Reproduce/Distribute/Publicly-Perform/Adaptation rights, Section 4(a)-(b) impose only a "keep license notice + give attribution" condition, no NC/SA restriction anywhere in the text). Attribution requirement: cite the original author/work per Section 4(b) when redistributing — satisfied here by citing Larson et al. 2019 in this repo's writeups (`CLINC150.md`). Structure independently confirmed from the fetched `data_full.json` itself: 15,000/3,000/4,500 in-scope train/val/test examples across 150 intent classes, plus 100/100/1,000 out-of-scope train/val/test examples under a single `oos` label — not assumed from the paper. |

**What "verified" means in this table, concretely**: for BANKING77 and
CLINC150, this repo fetched the license text as a live HTTP request during
this project and read the actual license name/terms in the response body
— not a citation of the associated paper, not a search-engine snippet,
and not an assumption based on the dataset's reputation. Amazon Fine Food
Reviews is one level weaker (corroborated via multiple independent
secondary sources rather than a first-party document this repo read
itself) and is flagged as such rather than overstated. SMS Spam Collection
is flagged as unverified rather than assumed permissive by default,
consistent with this project's rule to stop and flag ambiguous licensing
rather than proceed on convenience.

This repository's own code is MIT-licensed (see `LICENSE`). That license
covers the code in this repo only — it does not change or override the
separate license terms on any dataset listed above, each of which keeps
its own terms regardless of this repo's code license.

## Datasets considered for Workstream D and rejected (not used)

| Dataset | Considered for | Why rejected |
|---|---|---|
| Hate Speech and Offensive Language (Davidson et al. 2017, `t-davidson/hate-speech-and-offensive-language`) | A toxicity/moderation Noul-type task, to broaden domain coverage beyond CLINC150 and BANKING77 | **Ambiguous licensing — flagged, not used, per project rule.** The GitHub repo carries a top-level `LICENSE` file that reads as plain MIT (confirmed by fetching it directly, 2026-09-22). However, the repo's own README explicitly requests citation of the associated paper for any published work using the data ("Please cite our paper..."), directs users to a tracking form for research use, and repeats a content warning about the data's nature — signals of an academic-use norm layered on top of the repository's code license, not clearly covered or overridden by the MIT LICENSE file's terms (which govern reproduction/redistribution of the *file*, not necessarily research-use expectations the authors are signaling separately). This is the same category of mismatch (a permissive top-level repo license vs. non-uniform or unstated expectations on the actual data) that caused this project's earlier STS-B licensing mistake (see FINDINGS.md §2). Per this project's explicit rule ("if licensing is ambiguous... STOP and flag" rather than proceed on convenience), this dataset was not used, and no from-scratch relabeling or substitute was attempted as a workaround. |

This entry exists so the decision is on record rather than silently
absent — Workstream D's dataset-breadth goal (2-4 additional tasks) was
only partially completed (one dataset, CLINC150, added — see
`CLINC150.md`) specifically because a second reasonable candidate was
found and rejected for licensing reasons rather than replaced with a
less scrutinized one. Workstream D is explicitly a background/
lower-priority item in this project's task ordering, so this partial
result is treated as an acceptable stopping point rather than a gap
requiring a rushed substitute dataset.
