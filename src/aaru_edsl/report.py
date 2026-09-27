"""Offline scoring and publication artifacts from frozen responses."""
import csv
import hashlib
import math
import textwrap
from pathlib import Path
import numpy as np
from .data import ROOT, SEED, read, write
from .protocol import MODELS, selected

LABELS = {'aaru': 'Aaru', **{k:v['label'] for k,v in MODELS.items()}}
COLORS = {'aaru':'#383e49','astra':'#187c80','fable':'#c77446','gemini':'#6f62b5'}

def equivalent(a,b):
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):
        return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if isinstance(a,float) and isinstance(b,(int,float)):
        return math.isclose(a,b,rel_tol=0,abs_tol=1e-10)
    return a==b

def tvd(p, q):
    if len(p)!=len(q) or len(p)<2:
        raise ValueError('Mismatched probability vectors')
    for vector in (p,q):
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in vector) or abs(sum(vector)-1)>1e-5:
            raise ValueError('Invalid probability vector')
    return 50*sum(abs(a-b) for a,b in zip(p,q))

def score(questions, forecasts):
    by_key={}
    for r in forecasts:
        key=(r['model_key'],r['id'])
        if key in by_key: raise ValueError('Duplicate forecast')
        by_key[key]=r
    expected={(k,q['id']) for k in MODELS for q in questions}
    if set(by_key)!=expected: raise ValueError('Missing or unexpected forecasts')
    rows=[]
    for q in questions:
        reference=[o['p'] for o in q['options']]
        rows.append({'id':q['id'],'family':q['family'],'model':'aaru',
                     'tvd':tvd(reference,[o['f'] for o in q['options']])})
        for key in MODELS:
            r=by_key[key,q['id']]
            if set(r['answer'])!={o['label'] for o in q['options']}: raise ValueError('Answer label mismatch')
            rows.append({'id':q['id'],'family':q['family'],'model':key,
                         'tvd':tvd(reference,[r['answer'][o['label']] for o in q['options']])})
    return rows

def summarize(rows):
    questions=list(dict.fromkeys(r['id'] for r in rows))
    families={r['id']:r['family'] for r in rows}
    names=sorted(set(families.values()))
    values={(r['model'],r['id']):r['tvd'] for r in rows}
    counts=np.array([sum(families[q]==f for q in questions) for f in names])
    rng=np.random.default_rng(SEED)
    draws=rng.integers(0,len(names),size=(20000,len(names)))
    summary={'questions':len(questions),'families':len(names),'seed':SEED,'bootstrap_repetitions':20000,'models':{}}
    for key in LABELS:
        v=np.array([values[key,q] for q in questions])
        diffs=np.array([values[key,q]-values['aaru',q] for q in questions])
        sums=np.array([sum(values[key,q]-values['aaru',q] for q in questions if families[q]==f) for f in names])
        boot=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
        summary['models'][key]={'label':LABELS[key],'mean_tvd':float(v.mean()),'median_tvd':float(np.median(v)),
             'mean_difference_from_aaru':float(diffs.mean()),'paired_family_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),
             'wins_against_aaru':int((diffs < -1e-9).sum()),'ties_against_aaru':int((abs(diffs)<=1e-9).sum())}
    return summary

def plots(rows, summary, questions, forecasts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    keys=list(LABELS)
    values={key:np.array([r['tvd'] for r in rows if r['model']==key]) for key in keys}
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    with PdfPages(out/'benchmark.pdf') as pdf:
        fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
        bars=axes[0].bar([LABELS[k] for k in keys],[values[k].mean() for k in keys],color=[COLORS[k] for k in keys])
        axes[0].bar_label(bars,fmt='%.2f',padding=4)
        axes[0].set(ylabel='Mean TVD (percentage points)',title='Same 100 survey questions; lower is better',ylim=(0,max(v.mean() for v in values.values())*1.25))
        axes[0].tick_params(axis='x',labelrotation=15)
        for key in keys:
            v=np.sort(values[key]);axes[1].step(v,np.arange(1,len(v)+1)/len(v),where='post',label=LABELS[key],color=COLORS[key],lw=2)
        axes[1].set(xlabel='TVD (percentage points)',ylabel='Share of questions at or below this error',title='Distribution of question-level errors')
        axes[1].legend(frameon=False)
        fig.suptitle('One-shot marginal forecasts through EDSL / Expected Parrot',fontsize=15)
        fig.savefig(out/'comparison.png',dpi=180);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(12,4.3),layout='constrained')
        for ax,key in zip(axes,MODELS):
            diffs=values[key]-values['aaru']
            ax.hist(diffs,bins=np.linspace(-50,50,26),color=COLORS[key],alpha=.9)
            ax.axvline(0,color='#333',lw=1);ax.axvline(diffs.mean(),color=COLORS[key],ls='--')
            ax.set(title=LABELS[key],xlabel='Model TVD minus Aaru TVD',ylabel='Questions')
        fig.suptitle('Paired errors: negative values favor the one-shot model',fontsize=14)
        fig.savefig(out/'paired_errors.png',dpi=180);pdf.savefig(fig);plt.close(fig)
        by_key={(r['model_key'],r['id']):r for r in forecasts}
        # Sample order is frozen before inference; text restrictions only improve plot legibility.
        examples=[q for q in questions if len(q['options'])<=4 and q['prompt'].isascii() and all(o['label'].isascii() for o in q['options'])][:6]
        fig,axes=plt.subplots(3,2,figsize=(12,12),layout='constrained')
        for ax,q in zip(axes.flat,examples):
            labels=[o['label'] for o in q['options']];x=np.arange(len(labels));width=.16
            vectors={'survey':[o['p'] for o in q['options']],'aaru':[o['f'] for o in q['options']],
                     **{k:[by_key[k,q['id']]['answer'][label] for label in labels] for k in MODELS}}
            for j,(key,vals) in enumerate(vectors.items()):
                ax.bar(x+(j-2)*width,np.array(vals)*100,width,label='Survey' if key=='survey' else LABELS[key],color='#b9c0c6' if key=='survey' else COLORS[key])
            ax.set_xticks(x,['\n'.join(textwrap.wrap(s,23)) for s in labels],fontsize=8)
            ax.set_title('\n'.join(textwrap.wrap(q['prompt'],67)),fontsize=9)
            ax.set_ylabel('Share (%)');ax.set_ylim(0,105)
            from urllib.parse import urlsplit
            ax.text(0,-.25,'Source: '+urlsplit(q['source']).netloc,transform=ax.transAxes,fontsize=8)
        for ax in list(axes.flat)[len(examples):]:ax.set_visible(False)
        if examples:fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',ncol=5,bbox_to_anchor=(.5,1.025),frameon=False)
        fig.savefig(out/'examples.png',dpi=180,bbox_inches='tight');pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
    write(ROOT/'results/examples.json',examples)

def readme(summary,audits):
    lines=['# One-shot survey forecasts with EDSL and Expected Parrot','',
        '**A fixed sample of 100 Aaru benchmark questions, answered by GPT-6 Astra, Fable 5.1, and Gemini 3.8 Flash through EDSL / Expected Parrot.** Each model predicts the whole answer distribution in one response. Aaru’s published predictions are scored on the same questions. Invalid-format responses may receive one explicitly requested retry, as documented below.','',
        '| Method | Mean TVD ↓ | Difference from Aaru | 95% paired family-bootstrap interval |',
        '|---|---:|---:|---|']
    for key,s in summary['models'].items():
        ci=s['paired_family_bootstrap_95ci'];lines.append(f"| {s['label']} | {s['mean_tvd']:.2f} | {s['mean_difference_from_aaru']:+.2f} | {'—' if key=='aaru' else f'[{ci[0]:+.2f}, {ci[1]:+.2f}]'} |")
    best=min(MODELS,key=lambda k:summary['models'][k]['mean_tvd']);s=summary['models'][best]
    lines += ['',f"{s['label']} has the lowest mean error among the three one-shot models in this sample. Its mean TVD is {s['mean_tvd']:.2f}, versus {summary['models']['aaru']['mean_tvd']:.2f} for Aaru. This is a descriptive ranking from the retained valid responses, not evidence of a stable ordering across repeated runs.",'']
    lines += ['| Model | Valid on first submission | Valid after bounded retry |','|---|---:|---:|']
    for key in MODELS:
        lines.append(f"| {LABELS[key]} | {summary['validity'][key]['first_submission_valid']}/100 | 100/100 |")
    if summary['first_submission_common_questions']<100:
        lines += ['', 'Only failed-format questions were retried, once with identical model settings and prompts; successful distributions were retained regardless of accuracy. Failed attempts are listed in [failures.json](data/failures.json). No invalid response was repaired into a forecast or replaced with a uniform distribution. Backend error reports did not include full raw responses for these failures, so their provider finish reasons could not be independently checked.','',
            f"Sensitivity check on the **{summary['first_submission_common_questions']} questions with valid first-submission responses from every model**, excluding retry outcomes:",'',
            '| Method | Mean TVD on common first-submission questions |','|---|---:|']
        for key in LABELS:lines.append(f"| {LABELS[key]} | {summary['first_submission_common_mean_tvd'][key]:.2f} |")
    lines += ['',
        '![Mean errors and TVD distributions](figures/comparison.png)','',
        '## Reproduce the results without inference','',
        'Requires Git and [uv](https://docs.astral.sh/uv/). No credential or model call is needed to rebuild the numerical results and plots from the included response archive. The first run downloads the locked dependencies.','',
        '```sh','git clone https://github.com/johnjosephhorton/aaru-edsl-benchmark.git','cd aaru-edsl-benchmark','uv run --frozen aaru-edsl reproduce --check','```','',
        'Outputs include [scores](results/scores.csv), [summary](results/summary.json), [cost reconciliation](data/cost_audit.json), [example questions and sources](results/examples.json), and a [three-page PDF](figures/benchmark.pdf). Checksums protect archive integrity; they are not independent attestations of inference provenance.','',
        '## The EDSL example','',
        'The benchmark uses native `QuestionDistribution` objects. The same labels, population, question text, date, and instructions are supplied to each model. `make_job` builds the following pattern for every sampled question and applies the frozen provider settings:','',
        '```python','import json','from edsl import Agent, Model, QuestionDistribution, Survey','from aaru_edsl.protocol import SYSTEM, MODELS, context','',
        'sample = json.load(open("data/sample.json"))','questions = [','    QuestionDistribution(','        question_name=row["question_name"],','        question_text=json.dumps(context(row), ensure_ascii=False),','        question_options=[o["label"] for o in row["options"]],','        include_comment=False,','    )','    for row in sample',']',
        'agent = Agent(traits={"role": "forecaster"}, instruction=SYSTEM,','              traits_presentation_template="")','spec = MODELS["astra"]  # also "fable" or "gemini"','model = Model(spec["model"], service_name=spec["service"], **spec["parameters"])','jobs = Survey(questions).by(agent).by(model)','```','',
        'The runner submits these jobs explicitly to production EP remote inference. Only `EXPECTED_PARROT_API_KEY` is needed; no direct provider API key is used. Each interview has empty cross-question memory. Model outputs are label-keyed probabilities summing to one; native validation and an independent archive check reject invalid distributions. There is no uniform fallback, normalization, retrieval, persona simulation, or fine-tuning.','',
        '## Run new forecasts through EP — paid','',
        '```sh','cp .env.example .env','# Set EXPECTED_PARROT_API_KEY locally.','uv run --frozen aaru-edsl infer --stage smoke --allow-paid-inference','uv run --frozen aaru-edsl status','# Repeat status until all three smoke jobs have been saved.','uv run --frozen aaru-edsl infer --stage full --allow-paid-inference','uv run --frozen aaru-edsl status','# Repeat status until all remaining jobs have been saved.','uv run --frozen aaru-edsl archive','uv run --frozen aaru-edsl reproduce','```','',
        'The first stage runs the first sampled question for each model; the second runs the remaining 99. Together they are exactly the same 100 for all three models. Existing submission receipts prevent duplicate jobs; completed Results are retained under ignored `runs/`. Remote caching is enabled (`fresh=False`). An ambiguous submission attempt blocks resubmission until its receipt is reconciled. If the main run contains validation failures, inspect its saved failure records, then explicitly request one recovery round with `uv run --frozen aaru-edsl infer --stage retry --allow-paid-inference`, followed by `status`. Only failed questions are retried; this stage bypasses the response cache (`fresh=True`) and uses identical model settings and prompts. Archiving requires 100 valid forecasts per model. It explicitly replaces the included response archive; perform fresh runs on a new branch if you want to retain the original checkout unchanged.','',
        '## Sample and model settings','',
        f"Selection uses `random.Random({SEED}).sample(sorted(eligible, key=id), 100)` from the 2,808 eligible categorical questions. Eligibility excludes 143 noncategorical entries and 42 strict duplicates from the original 2,993-record export. The seed fixes selection only; inference is stochastic. The selected questions span {summary['families']} operational survey families.",'',
        '| Model | EP service | Reasoning | Output token cap |','|---|---|---|---:|',
        '| `gpt-6-astra` | `openai` | high | 8,192 |','| `claude-fable-5-1` | `anthropic` | adaptive, high effort | 8,192 |','| `gemini-3.8-flash` | `google` | dynamic thinking budget (`-1`) | 8,192 |','',
        'Reasoning controls differ across providers; these settings do not equate compute budgets. Gemini 3.8 Flash was the latest general-purpose text release listed in [Google’s model catalog](https://ai.google.dev/gemini-api/docs/models) and EP’s catalog when the experiment was prepared on September 27, 2026. Exact names are frozen rather than using moving “latest” aliases. EDSL is pinned to public Git commit `cfef949e4ea87ab0bf6e998ab64b54a1bfcec6e4`, which includes `QuestionDistribution`.','',
        'See [protocol](data/protocol.json), [model catalog snapshot](data/catalog.json), and the `*_jobs.json` / `*_prompts.json` files for the precise settings and rendered prompts. Reference shares, Aaru forecasts, source URLs, and other survey answers are excluded from model input. Audience descriptions are supplied by the benchmark and may contain substantive context; excluding numeric target fields does not prove freedom from leakage.','',
        '## Costs and reconciliation','',
        '| Model | Results cost | Reported EP job cost | Reconciliation |','|---|---:|---:|---|']
    for key in MODELS:
        selected_audits=[a for a in audits if a['model_key']==key]
        low=sum(a['results_cost_usd'] for a in selected_audits);high=sum(a['reported_job_cost_usd'] for a in selected_audits)
        state='discrepancy' if any(a['state']=='discrepancy' for a in selected_audits) else 'matched' if all(a['state']=='matched' for a in selected_audits) else 'unresolved'
        lines.append(f'| {LABELS[key]} | ${low:.5f} | ${high:.5f} | {state} |')
    lines+=['','Results costs use recorded token-price metadata and exclude cache hits. Reported EP costs are finalized job-accounting records, not independently verified account debits. The runner emits `CostReconciliationWarning` when those totals differ by more than $0.0002 per job (allowing two token-type billing line items to round to $0.0001). Answers are saved before warning; no automatic paid retry occurs. This local guard addresses the discrepancy documented in [EDSL #2668](https://github.com/expectedparrot/edsl/issues/2668); it is not a fix to EP’s billing backend.','',
        'A separate one-question Astra adapter check on `openai_v2` returned a valid answer but exposed the earlier accounting mismatch ($0.04981 in Results versus $0.61990 reported by EP). It is excluded from the 100-question accuracy comparison and the table above; [its cost record](data/adapter_diagnostic.json) is preserved. We switched to EP’s catalog-listed `openai` route based on billing compatibility before the full run, not based on forecast accuracy. Add that diagnostic cost when totaling all experiment spending.','',
        '## Interpretation','',
        'TVD is `50 * sum(abs(predicted_share - survey_share))` for shares on a 0–1 scale, reported in percentage points. Lower is better. Ten TVD points means ten percentage points of probability mass must move to match the reference; it is not “10% of people predicted incorrectly.” Questions have equal weight. Aaru and reference shares are scored as exported.','',
        'The paired intervals use 20,000 bootstrap draws over connected survey families, keeping sampled questions in a family together (seed 20260927). Families share cluster, series, canonical source, or sufficiently long identical question/option text, with a documented Conference Board series join. They are an operational dependence grouping, not proof of independence. Intervals describe variation across sampled families; they exclude stochastic reruns, model selection, and human-survey sampling error.','',
        '![Paired error distributions](figures/paired_errors.png)','',
        'This is a retrospective benchmark against published survey results. Those results may occur in model training data. It does not establish prospective performance on novel questions, the quality of individual simulated respondents, joint distributions, or how Aaru generated its predictions. Results apply to this seeded 100-question sample, not the full benchmark.','',
        '![Example marginal distributions](figures/examples.png)','',
        'Examples are the first six sampled questions with at most four options and ASCII question/option text, selected for legibility without reference to errors. Full wording and original source links are preserved in the data.','',
        '## Data and checks','',
        'Source: [Aaru’s research publication](https://aaru.com/publications/population-simulation-at-the-replication-floor-research) and [question export](https://aaru.com/data/population-simulation/questions.json). The byte-preserving source snapshot and provenance are included. [NOTICE.md](NOTICE.md) describes third-party materials. The MIT license covers original code only.','',
        '```sh','uv run --frozen pytest -q','```','',
        'GitHub Actions runs tests and offline reproduction without credentials. Response exports contain answer text, usage, price metadata, cache flags, and rendered prompts; they omit provider/account identifiers and reasoning content. EP job UUIDs in cost audits identify the supporting private accounting records.']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n')

def reproduce(check=False):
    for name,sha in read(ROOT/'data/manifest.json')['files'].items():
        if Path(name).name!=name or hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()!=sha:
            raise ValueError(f'Archive checksum mismatch: {name}')
    questions=read(ROOT/'data/sample.json')
    if questions!=selected(): raise ValueError('Frozen sample differs from seeded reconstruction')
    forecasts=read(ROOT/'data/forecasts.json')
    rows=score(questions,forecasts);summary=summarize(rows)
    summary['validity']={k:{'first_submission_valid':sum(r['model_key']==k and r['stage']!='retry' for r in forecasts),
                            'final_valid':sum(r['model_key']==k for r in forecasts)} for k in MODELS}
    retry_ids={r['id'] for r in forecasts if r['stage']=='retry'}
    summary['first_submission_common_questions']=len(questions)-len(retry_ids)
    summary['first_submission_common_mean_tvd']={k:float(np.mean([r['tvd'] for r in rows if r['model']==k and r['id'] not in retry_ids])) for k in LABELS}
    expected=ROOT/'results/expected_summary.json'
    if check and (not expected.exists() or not equivalent(read(expected),summary)): raise ValueError('Numerical summary differs from frozen expected results')
    write(ROOT/'results/summary.json',summary)
    if not expected.exists():write(expected,summary)
    with (ROOT/'results/scores.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['id','family','model','tvd'],lineterminator='\n');writer.writeheader();writer.writerows(rows)
    plots(rows,summary,questions,forecasts)
    readme(summary,read(ROOT/'data/cost_audit.json'))
    print(__import__('json').dumps(summary,indent=2))
