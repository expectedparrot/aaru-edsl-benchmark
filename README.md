# One-shot survey forecasts with EDSL and Expected Parrot

**A fixed sample of 100 Aaru benchmark questions, answered by GPT-6 Astra at medium and high reasoning effort, Fable 5.1, and Gemini 3.8 Flash through EDSL / Expected Parrot.** Each configuration predicts the whole answer distribution in one response. Aaru’s published predictions are scored on the same questions. Invalid-format responses may receive one explicitly requested retry, as documented below.

| Method | Mean TVD ↓ | Difference from Aaru | 95% paired family-bootstrap interval |
|---|---:|---:|---|
| Aaru | 7.50 | +0.00 | — |
| GPT-6 Astra (high) | 9.05 | +1.55 | [+0.56, +2.62] |
| GPT-6 Astra (medium) | 8.85 | +1.35 | [+0.25, +2.57] |
| Fable 5.1 | 10.33 | +2.83 | [+1.69, +4.04] |
| Gemini 3.8 Flash | 11.04 | +3.54 | [+2.09, +5.07] |

GPT-6 Astra (medium) has the lowest mean error among the four one-shot configurations in this sample. Its mean TVD is 8.85, versus 7.50 for Aaru. This is a descriptive ranking from the retained valid responses, not evidence of a stable ordering across repeated runs.

## GPT-6: medium versus high

High minus medium mean TVD is **+0.20 points** (95% paired family-bootstrap interval **[-0.30, +0.64]**). Negative favors high. High has lower error on **37/100** questions, medium on **43/100**, with **20** ties.

The interval includes zero: this sample does not establish a reliable accuracy advantage for either effort level.

Both configurations use exactly the same 100 questions, rendered prompts, EP service (`openai`), and 8,192-token output cap. The only configured difference is `reasoning_effort`. Medium was added after the high/Fable/Gemini results were inspected; existing responses were retained. Each configuration has one retained draw per question, and the runs occurred at different times, so this is an exploratory paired comparison rather than a randomized, repeated experiment isolating effort from all provider variation.

![GPT-6 medium versus high on matched questions](figures/astra_effort.png)

The separate full-corpus 2,808-question medium-effort analysis used a different prompting setup and reported GPT-6 TVD 10.11 versus Aaru 7.71. Those numbers are not the medium-effort results reported here.

| Model | Valid on first submission | Valid after bounded retry |
|---|---:|---:|
| GPT-6 Astra (high) | 100/100 | 100/100 |
| GPT-6 Astra (medium) | 100/100 | 100/100 |
| Fable 5.1 | 100/100 | 100/100 |
| Gemini 3.8 Flash | 97/100 | 100/100 |

Only failed-format questions were retried, once with identical model settings and prompts; successful distributions were retained regardless of accuracy. Failed attempts are listed in [failures.json](data/failures.json). No invalid response was repaired into a forecast or replaced with a uniform distribution. Backend error reports did not include full raw responses for these failures, so their provider finish reasons could not be independently checked.

Sensitivity check on the **97 questions with valid first-submission responses from every model**, excluding retry outcomes:

| Method | Mean TVD on common first-submission questions |
|---|---:|
| Aaru | 7.65 |
| GPT-6 Astra (high) | 9.21 |
| GPT-6 Astra (medium) | 9.03 |
| Fable 5.1 | 10.54 |
| Gemini 3.8 Flash | 11.19 |

![Mean errors and TVD distributions](figures/comparison.png)

## Reproduce the results without inference

Requires Git and [uv](https://docs.astral.sh/uv/). No credential or model call is needed to rebuild the numerical results and plots from the included response archive. The first run downloads the locked dependencies.

```sh
git clone https://github.com/expectedparrot/aaru-edsl-benchmark.git
cd aaru-edsl-benchmark
uv run --frozen aaru-edsl reproduce --check
```

Outputs include [scores](results/scores.csv), [summary](results/summary.json), [cost reconciliation](data/cost_audit.json), [example questions and sources](results/examples.json), and a [four-page PDF](figures/benchmark.pdf). Checksums protect archive integrity; they are not independent attestations of inference provenance.

## The EDSL example

The benchmark uses native `QuestionDistribution` objects. The same labels, population, question text, date, and instructions are supplied to each model. `make_job` builds the following pattern for every sampled question and applies the frozen provider settings:

```python
import json
from edsl import Agent, Model, QuestionDistribution, Survey
from aaru_edsl.protocol import SYSTEM, MODELS, context

sample = json.load(open("data/sample.json"))
questions = [
    QuestionDistribution(
        question_name=row["question_name"],
        question_text=json.dumps(context(row), ensure_ascii=False),
        question_options=[o["label"] for o in row["options"]],
        include_comment=False,
    )
    for row in sample
]
agent = Agent(traits={"role": "forecaster"}, instruction=SYSTEM,
              traits_presentation_template="")
spec = MODELS["astra"]  # high; also "astra_medium", "fable", or "gemini"
model = Model(spec["model"], service_name=spec["service"], **spec["parameters"])
jobs = Survey(questions).by(agent).by(model)
```

The runner submits these jobs explicitly to production EP remote inference. Only `EXPECTED_PARROT_API_KEY` is needed; no direct provider API key is used. Each interview has empty cross-question memory. Model outputs are label-keyed probabilities summing to one; native validation and an independent archive check reject invalid distributions. There is no uniform fallback, normalization, retrieval, persona simulation, or fine-tuning.

## Run new forecasts through EP — paid

From the repository directory, log in to Expected Parrot to save your EP key in the local `.env` file. Complete the sign-in in your browser:

```sh
uv run --frozen ep auth login
uv run --frozen ep check
```

If EDSL is already installed in your active environment, the login command is simply `ep auth login`. The benchmark reads the saved key automatically. Then run:

```sh
uv run --frozen aaru-edsl infer --stage smoke --allow-paid-inference
uv run --frozen aaru-edsl status
# Repeat status until all four smoke jobs have been saved.
uv run --frozen aaru-edsl infer --stage full --allow-paid-inference
uv run --frozen aaru-edsl status
# Repeat status until all remaining jobs have been saved.
uv run --frozen aaru-edsl archive
uv run --frozen aaru-edsl reproduce
```

The first stage runs the first sampled question for each configuration; the second runs the remaining 99. Together they are exactly the same 100 for all four configurations. To submit only the medium arm, add `--models astra_medium` to both `infer` commands; other choices are `astra` (high), `fable`, and `gemini`. Existing submission receipts prevent duplicate jobs; completed Results are retained under ignored `runs/`. Remote caching is enabled (`fresh=False`). An ambiguous submission attempt blocks resubmission until its receipt is reconciled. If the main run contains validation failures, inspect its saved failure records, then explicitly request one recovery round with `uv run --frozen aaru-edsl infer --stage retry --allow-paid-inference`, followed by `status`. Only failed questions are retried; this stage bypasses the response cache (`fresh=True`) and uses identical model settings and prompts. Archiving requires 100 valid forecasts per configuration, including retained records for arms not newly submitted. It explicitly replaces the included response archive; perform fresh runs on a new branch if you want to retain the original checkout unchanged.

## Sample and model settings

Selection uses `random.Random(20260927).sample(sorted(eligible, key=id), 100)` from the 2,808 eligible categorical questions. Eligibility excludes 143 noncategorical entries and 42 strict duplicates from the original 2,993-record export. The seed fixes selection only; inference is stochastic. The selected questions span 40 operational survey families.

| Model | EP service | Reasoning | Output token cap |
|---|---|---|---:|
| `gpt-6-astra` (`astra`) | `openai` | high | 8,192 |
| `gpt-6-astra` (`astra_medium`) | `openai` | medium | 8,192 |
| `claude-fable-5-1` | `anthropic` | adaptive, high effort | 8,192 |
| `gemini-3.8-flash` | `google` | dynamic thinking budget (`-1`) | 8,192 |

Reasoning controls differ across providers; these settings do not equate compute budgets. Gemini 3.8 Flash was the latest general-purpose text release listed in [Google’s model catalog](https://ai.google.dev/gemini-api/docs/models) and EP’s catalog when the experiment was prepared on September 27, 2026. Exact names are frozen rather than using moving “latest” aliases. EDSL is pinned to public Git commit `cfef949e4ea87ab0bf6e998ab64b54a1bfcec6e4`, which includes `QuestionDistribution`.

See [protocol](data/protocol.json), [model catalog snapshot](data/catalog.json), and the `*_jobs.json` / `*_prompts.json` files for the precise settings and rendered prompts. Reference shares, Aaru forecasts, source URLs, and other survey answers are excluded from model input. Audience descriptions are supplied by the benchmark and may contain substantive context; excluding numeric target fields does not prove freedom from leakage.

## Costs and reconciliation

| Model | Results cost | Reported EP job cost | Reconciliation |
|---|---:|---:|---|
| GPT-6 Astra (high) | $3.28681 | $3.28670 | matched |
| GPT-6 Astra (medium) | $1.58811 | $1.58810 | matched |
| Fable 5.1 | $2.90797 | $2.90770 | matched |
| Gemini 3.8 Flash | $3.45663 | $3.45630 | unresolved |

Medium used **$1.59** in recorded Results cost versus **$3.29** for high (48% as much). This is the cost of these retained runs, not a fixed price or a guarantee for future runs.

Results costs use recorded token-price metadata and exclude cache hits. Reported EP costs are finalized job-accounting records, not independently verified account debits. The runner emits `CostReconciliationWarning` when those totals differ by more than $0.0002 per job (allowing two token-type billing line items to round to $0.0001). Answers are saved before warning; no automatic paid retry occurs. This local guard addresses the discrepancy documented in [EDSL #2668](https://github.com/expectedparrot/edsl/issues/2668); it is not a fix to EP’s billing backend.

A separate one-question Astra adapter check on `openai_v2` returned a valid answer but exposed the earlier accounting mismatch ($0.04981 in Results versus $0.61990 reported by EP). It is excluded from the 100-question accuracy comparison and the table above; [its cost record](data/adapter_diagnostic.json) is preserved. We switched to EP’s catalog-listed `openai` route based on billing compatibility before the full run, not based on forecast accuracy. Add that diagnostic cost when totaling all experiment spending.

## Interpretation

TVD is `50 * sum(abs(predicted_share - survey_share))` for shares on a 0–1 scale, reported in percentage points. Lower is better. Ten TVD points means ten percentage points of probability mass must move to match the reference; it is not “10% of people predicted incorrectly.” Questions have equal weight. Aaru and reference shares are scored as exported.

The paired intervals use 20,000 bootstrap draws over connected survey families, keeping sampled questions in a family together (seed 20260927). Families share cluster, series, canonical source, or sufficiently long identical question/option text, with a documented Conference Board series join. They are an operational dependence grouping, not proof of independence. Intervals describe variation across sampled families; they exclude stochastic reruns, model selection, and human-survey sampling error.

![Paired error distributions](figures/paired_errors.png)

This is a retrospective benchmark against published survey results. Those results may occur in model training data. It does not establish prospective performance on novel questions, the quality of individual simulated respondents, joint distributions, or how Aaru generated its predictions. Results apply to this seeded 100-question sample, not the full benchmark.

![Example marginal distributions](figures/examples.png)

Examples are the first six sampled questions with at most four options and ASCII question/option text, selected for legibility without reference to errors. Full wording and original source links are preserved in the data.

## Data and checks

Source: [Aaru’s research publication](https://aaru.com/publications/population-simulation-at-the-replication-floor-research) and [question export](https://aaru.com/data/population-simulation/questions.json). The byte-preserving source snapshot and provenance are included. [NOTICE.md](NOTICE.md) describes third-party materials. The MIT license covers original code only.

```sh
uv run --frozen pytest -q
```

GitHub Actions runs tests and offline reproduction without credentials. Response exports contain answer text, usage, price metadata, cache flags, and rendered prompts; they omit provider/account identifiers and reasoning content. EP job UUIDs in cost audits identify the supporting private accounting records.
