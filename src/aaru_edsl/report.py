"""Offline scoring and publication artifacts from frozen responses."""
import csv
import hashlib
import math
import textwrap
from pathlib import Path
import numpy as np
from .data import ROOT, SEED, read, write
from .protocol import BENCHMARK_MODELS as MODELS, MODELS as ALL_MODELS, SCREENED_OUT, selected

LABELS = {'aaru': 'Aaru', **{k:v['label'] for k,v in MODELS.items()}}
COLORS = {k: '#687d96' for k in MODELS}
COLORS.update({'aaru':'#383e49','astra':'#187c80','astra_medium':'#4d94cf','fable':'#c77446','gemini':'#6f62b5'})

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

def score(questions, forecasts, allow_incomplete=False):
    by_key={}
    for r in forecasts:
        key=(r['model_key'],r['id'])
        if key in by_key: raise ValueError('Duplicate forecast')
        by_key[key]=r
    expected={(k,q['id']) for k in MODELS for q in questions}
    if not set(by_key)<=expected or (not allow_incomplete and set(by_key)!=expected):
        raise ValueError('Missing or unexpected forecasts')
    rows=[]
    for q in questions:
        reference=[o['p'] for o in q['options']]
        rows.append({'id':q['id'],'family':q['family'],'model':'aaru',
                     'tvd':tvd(reference,[o['f'] for o in q['options']])})
        for key in MODELS:
            if (key,q['id']) not in by_key: continue
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
    summary={'questions':len(questions),'families':len(names),'seed':SEED,'bootstrap_repetitions':20000,'models':{},'incomplete_models':{}}
    for key in LABELS:
        present=[q for q in questions if (key,q) in values]
        if len(present)!=len(questions):
            observed=[values[key,q] for q in present]
            total=sum(observed);missing=len(questions)-len(present)
            summary['incomplete_models'][key]={'label':LABELS[key], 'valid_questions':len(present),
                'mean_tvd_observed_only':float(np.mean(observed)) if observed else None,
                'aaru_mean_on_same_observed_questions':float(np.mean([values['aaru',q] for q in present])) if present else None,
                'full_sample_mean_tvd_bounds':[total/len(questions),(total+100*missing)/len(questions)]}
            continue
        v=np.array([values[key,q] for q in questions])
        diffs=np.array([values[key,q]-values['aaru',q] for q in questions])
        sums=np.array([sum(values[key,q]-values['aaru',q] for q in questions if families[q]==f) for f in names])
        boot=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
        summary['models'][key]={'label':LABELS[key],'mean_tvd':float(v.mean()),'median_tvd':float(np.median(v)),
             'mean_difference_from_aaru':float(diffs.mean()),'paired_family_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),
             'wins_against_aaru':int((diffs < -1e-9).sum()),'ties_against_aaru':int((abs(diffs)<=1e-9).sum())}
    return summary

def summarize_effort(rows):
    high={r['id']:r for r in rows if r['model']=='astra'}
    medium={r['id']:r for r in rows if r['model']=='astra_medium'}
    if not high or high.keys()!=medium.keys():
        raise ValueError('Effort comparison requires matching high and medium questions')
    names=sorted({r['family'] for r in high.values()})
    counts=np.array([sum(r['family']==f for r in high.values()) for f in names])
    diffs={q:high[q]['tvd']-medium[q]['tvd'] for q in high}
    sums=np.array([sum(diffs[q] for q in high if high[q]['family']==f) for f in names])
    draws=np.random.default_rng(SEED).integers(0,len(names),size=(20000,len(names)))
    boot=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    d=np.array(list(diffs.values()))
    return {'questions':len(high),'direction':'high TVD minus medium TVD; negative favors high',
            'mean_difference':float(d.mean()),'paired_family_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),
            'high_wins':int((d < -1e-9).sum()),'medium_wins':int((d > 1e-9).sum()),
            'ties':int((abs(d)<=1e-9).sum())}

def plots(rows, summary, questions, forecasts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    keys=list(summary['models'])
    values={key:np.array([r['tvd'] for r in rows if r['model']==key]) for key in keys}
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    with PdfPages(out/'benchmark.pdf') as pdf:
        ranked=sorted(keys,key=lambda k: values[k].mean())
        fig,axes=plt.subplots(1,2,figsize=(14,max(6,len(keys)*.33)),layout='constrained')
        y=np.arange(len(ranked))
        bars=axes[0].barh(y,[values[k].mean() for k in ranked],color=[COLORS[k] for k in ranked])
        axes[0].bar_label(bars,fmt='%.2f',padding=4,fontsize=8)
        axes[0].set_yticks(y,[LABELS[k] for k in ranked]);axes[0].invert_yaxis()
        axes[0].set(xlabel='Mean TVD (percentage points)',title='Same 100 questions; lower is better',xlim=(0,max(v.mean() for v in values.values())*1.15))
        for i,k in enumerate(ranked):
            lo,q1,median,q3,hi=np.quantile(values[k],[.1,.25,.5,.75,.9])
            axes[1].plot([lo,hi],[i,i],color=COLORS[k],lw=1)
            axes[1].plot([q1,q3],[i,i],color=COLORS[k],lw=5)
            axes[1].plot(median,i,'o',color='white',mec=COLORS[k],ms=4)
        axes[1].set_yticks(y,[]);axes[1].invert_yaxis()
        axes[1].set_ylim(axes[0].get_ylim())
        axes[1].set(xlabel='Question-level TVD',title='Error spread: 10–90%, thick 25–75%, dot median')
        fig.suptitle('One-shot survey marginals: configurations with all 100 valid answers',fontsize=15)
        fig.savefig(out/'comparison.png',dpi=180);pdf.savefig(fig);plt.close(fig)
        model_order=[k for k in ranked if k!='aaru']
        fig,ax=plt.subplots(figsize=(11,max(6,len(model_order)*.32)),layout='constrained')
        for i,k in enumerate(model_order):
            stats=summary['models'][k];lo,hi=stats['paired_family_bootstrap_95ci']
            ax.plot([lo,hi],[i,i],color=COLORS[k],lw=2)
            ax.plot(stats['mean_difference_from_aaru'],i,'o',color=COLORS[k])
        ax.set_yticks(range(len(model_order)),[LABELS[k] for k in model_order]);ax.invert_yaxis()
        ax.axvline(0,color='.3',ls='--')
        ax.set(xlabel='Mean model TVD minus Aaru TVD (percentage points)',title='Paired mean differences and 95% family-bootstrap intervals\nNegative favors the one-shot model; intervals are not multiplicity-adjusted')
        fig.savefig(out/'paired_errors.png',dpi=180);pdf.savefig(fig);plt.close(fig)
        audits=read(ROOT/'data/cost_audit.json')
        costs={k:sum(a['reported_job_cost_usd'] or 0 for a in audits if a['model_key']==k) for k in model_order}
        fig,axes=plt.subplots(1,2,figsize=(13,max(5,len(model_order)*.29)),layout='constrained',gridspec_kw={'width_ratios':[1.25,1]})
        for i,k in enumerate(model_order,1):
            axes[0].scatter(costs[k],values[k].mean(),color=COLORS[k],s=55)
            near_higher=any(abs(math.log10(costs[k]/costs[j]))<.1 and 0<values[j].mean()-values[k].mean()<.4 for j in model_order if costs[k]>0 and costs[j]>0)
            axes[0].annotate(str(i),(costs[k],values[k].mean()),xytext=(5,-12 if near_higher else 5),textcoords='offset points',fontsize=8)
        axes[0].axhline(values['aaru'].mean(),color=COLORS['aaru'],ls='--',label='Aaru (cost unavailable)')
        axes[0].set(xscale='log',xlabel='Reported EP cost for 100 questions (USD; log scale)',ylabel='Mean TVD (percentage points)',title='Accuracy and observed cost: complete runs only')
        axes[0].legend(frameon=False,loc='upper right')
        axes[1].axis('off')
        table=axes[1].table(cellText=[[i,LABELS[k],f'${costs[k]:.3f}'] for i,k in enumerate(model_order,1)],colLabels=['#','Model','EP cost'],colWidths=[.08,.69,.23],loc='center',cellLoc='left')
        table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.4)
        fig.savefig(out/'cost_accuracy.png',dpi=180);pdf.savefig(fig);plt.close(fig)
        validity=summary['validity']
        validity_order=sorted(MODELS,key=lambda k:(-validity[k]['final_valid'],-validity[k]['first_submission_valid'],LABELS[k]))
        first=np.array([validity[k]['first_submission_valid'] for k in validity_order])
        recovered=np.array([validity[k]['final_valid'] for k in validity_order])-first
        fig,ax=plt.subplots(figsize=(11,max(6,len(validity_order)*.3)),layout='constrained')
        y=np.arange(len(validity_order))
        ax.barh(y,100,color='#eeeeee',label='Unresolved after retry')
        ax.barh(y,first,color='#187c80',label='Valid on first submission')
        ax.barh(y,recovered,left=first,color='#c77446',label='Recovered on one retry')
        for i,k in enumerate(validity_order):ax.text(validity[k]['final_valid']+1,i,str(validity[k]['final_valid']),va='center',fontsize=8)
        ax.set_yticks(y,[LABELS[k] for k in validity_order]);ax.invert_yaxis()
        ax.set(xlim=(0,107),xlabel='Questions with a valid probability distribution (out of 100)')
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=3,frameon=False,fontsize=8)
        fig.suptitle('Valid-answer coverage: every benchmark configuration',fontsize=14)
        fig.savefig(out/'validity.png',dpi=180);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
        medium,high=values['astra_medium'],values['astra']
        limit=max(float(medium.max()),float(high.max()))*1.08
        axes[0].scatter(medium,high,color=COLORS['astra'],alpha=.65)
        axes[0].plot([0,limit],[0,limit],ls='--',color='.5')
        axes[0].set(xlabel='GPT-6 medium TVD',ylabel='GPT-6 high TVD',
                    title='Each dot is the same question at both efforts',xlim=(0,limit),ylim=(0,limit))
        delta=high-medium
        axes[1].hist(delta,bins=np.linspace(-30,30,25),color=COLORS['astra_medium'])
        axes[1].axvline(0,color='.3');axes[1].axvline(delta.mean(),color=COLORS['astra'],ls='--',label=f'Mean: {delta.mean():+.2f}')
        axes[1].set(xlabel='High TVD minus medium TVD',ylabel='Questions',title='Negative favors high; positive favors medium')
        axes[1].legend(frameon=False)
        ci=summary['astra_effort_comparison']['paired_family_bootstrap_95ci']
        fig.suptitle(f'GPT-6 reasoning effort: paired mean difference {delta.mean():+.2f} TVD (95% CI {ci[0]:+.2f}, {ci[1]:+.2f})',fontsize=13)
        fig.savefig(out/'astra_effort.png',dpi=180);pdf.savefig(fig);plt.close(fig)
        by_key={(r['model_key'],r['id']):r for r in forecasts}
        # Sample order is frozen before inference; text restrictions only improve plot legibility.
        examples=[q for q in questions if len(q['options'])<=4 and q['prompt'].isascii() and all(o['label'].isascii() for o in q['options'])][:6]
        fig,axes=plt.subplots(3,2,figsize=(12,12),layout='constrained')
        for ax,q in zip(axes.flat,examples):
            labels=[o['label'] for o in q['options']];x=np.arange(len(labels));width=.13
            vectors={'survey':[o['p'] for o in q['options']],'aaru':[o['f'] for o in q['options']],
                     **{k:[by_key[k,q['id']]['answer'][label] for label in labels] for k in ('astra','astra_medium','fable','gemini')}}
            for j,(key,vals) in enumerate(vectors.items()):
                ax.bar(x+(j-(len(vectors)-1)/2)*width,np.array(vals)*100,width,label='Survey' if key=='survey' else LABELS[key],color='#b9c0c6' if key=='survey' else COLORS[key])
            ax.set_xticks(x,['\n'.join(textwrap.wrap(s,23)) for s in labels],fontsize=8)
            ax.set_title('\n'.join(textwrap.wrap(q['prompt'],67)),fontsize=9)
            ax.set_ylabel('Share (%)');ax.set_ylim(0,105)
            from urllib.parse import urlsplit
            ax.text(0,-.25,'Source: '+urlsplit(q['source']).netloc,transform=ax.transAxes,fontsize=8)
        for ax in list(axes.flat)[len(examples):]:ax.set_visible(False)
        if examples:fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',ncol=3,bbox_to_anchor=(.5,1.045),frameon=False)
        fig.savefig(out/'examples.png',dpi=180,bbox_inches='tight');pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
    write(ROOT/'results/examples.json',examples)

def readme(summary,audits):
    ranked=sorted(summary['models'],key=lambda k:summary['models'][k]['mean_tvd'])
    distinct=len({v['model'] for v in MODELS.values()})
    services=len({v['service'] for v in MODELS.values()})
    lines=['# One-shot survey forecasts with EDSL and Expected Parrot','',
        f'**{distinct} distinct models, {len(MODELS)} configurations, the same 100 survey questions.** Each model predicts the whole answer distribution in one response through production Expected Parrot. Aaru’s published predictions are scored against the same published survey shares. Lower total variation distance (TVD) is better. The main ranking contains {len(summary["models"])-1} configurations with all 100 valid distributions; {len(summary["incomplete_models"])} incomplete runs are reported separately below.','',
        '![Mean errors and question-level error spread](figures/comparison.png)','',
        '| Method | Mean TVD ↓ | Difference from Aaru | 95% paired family-bootstrap interval |',
        '|---|---:|---:|---|']
    for key in ranked:
        st=summary['models'][key];ci=st['paired_family_bootstrap_95ci']
        interval='—' if key=='aaru' else f'[{ci[0]:+.2f}, {ci[1]:+.2f}]'
        lines.append(f"| {st['label']} | {st['mean_tvd']:.2f} | {st['mean_difference_from_aaru']:+.2f} | {interval} |")
    best=min((k for k in summary['models'] if k!='aaru'),key=lambda k:summary['models'][k]['mean_tvd']);st=summary['models'][best]
    lines += ['',f"**{st['label']} has the lowest mean one-shot error among complete runs in this sample: {st['mean_tvd']:.2f} TVD, versus {summary['models']['aaru']['mean_tvd']:.2f} for Aaru.** These are descriptive rankings, with one retained response per question and configuration. The intervals do not adjust for comparing many models or selecting the best result. They do not establish a stable ranking across repeated runs.",'',
        '## Reproduce without inference','',
        'Requires Git and [uv](https://docs.astral.sh/uv/). No credential or model call is needed. The first run downloads locked dependencies.','',
        '```sh','git clone https://github.com/expectedparrot/aaru-edsl-benchmark.git','cd aaru-edsl-benchmark','uv run --frozen aaru-edsl reproduce --check','uv run --frozen pytest -q','```','',
        'Included: [scores](results/scores.csv), [summary](results/summary.json), [response archive](data/forecasts.json), [cost reconciliation](data/cost_audit.json), [examples and source links](results/examples.json), and [PDF plots](figures/benchmark.pdf). Checksums protect file integrity; they are not independent attestations of inference provenance.','',
        '## The EDSL example','',
        '```python','import json','from edsl import Agent, Model, QuestionDistribution, Survey','from aaru_edsl.protocol import SYSTEM, MODELS, context','',
        'sample = json.load(open("data/sample.json"))','questions = [','    QuestionDistribution(','        question_name=row["question_name"],','        question_text=json.dumps(context(row), ensure_ascii=False),','        question_options=[o["label"] for o in row["options"]],','        include_comment=False,','    )','    for row in sample',']',
        'agent = Agent(traits={"role": "forecaster"}, instruction=SYSTEM,','              traits_presentation_template="")','spec = MODELS["astra"]  # choose any configuration in data/protocol.json','model = Model(spec["model"], service_name=spec["service"], **spec["parameters"])','jobs = Survey(questions).by(agent).by(model)','```','',
        'Every model receives identical rendered prompts: population, date, question, instructions, and answer labels. Each interview has empty cross-question memory. Native `QuestionDistribution` validation and an independent archive check require label-keyed probabilities summing to one. No uniform fallback, normalization, retrieval, persona simulation, or fine-tuning is used.','',
        '## Run new forecasts through EP — paid','',
        'From the repository directory, save an EP key in the local `.env` file by completing the browser login:','',
        '```sh','uv run --frozen ep auth login','uv run --frozen ep check','```','',
        'With EDSL already installed, the command is simply `ep auth login`. Only an EP key is needed; no direct provider key is used.','',
        '```sh','uv run --frozen aaru-edsl infer --stage smoke --allow-paid-inference','uv run --frozen aaru-edsl status','# Repeat status until smoke jobs are saved; inspect failures and cost audits.','uv run --frozen aaru-edsl infer --stage full --allow-paid-inference','uv run --frozen aaru-edsl status','# Repeat status until all jobs are saved.','# If needed, retry only failed questions, once:','uv run --frozen aaru-edsl infer --stage retry --allow-paid-inference','uv run --frozen aaru-edsl status','# Repeat status until retry jobs are saved.','uv run --frozen aaru-edsl archive','uv run --frozen aaru-edsl reproduce','```','',
        'Smoke runs ask the first sampled question; full runs ask the other 99. Add `--models astra astra_medium` (or other keys below) to submit a subset. Full submission requires a valid smoke answer for every selected configuration. Existing receipts prevent duplicate submissions; ambiguous submissions block resubmission until reconciled. Raw Results remain under ignored `runs/`. Remote caching is enabled; the single bounded retry bypasses the cache and preserves identical prompts and settings. Successful answers are never rerun based on accuracy. Archiving requires all 100 questions to be accounted for by a valid forecast or a documented final failure after the bounded retry, including retained arms. It replaces the included response archive: use a new branch for a fresh experiment. `--check` verifies the published archive’s numerical results; new stochastic runs need not reproduce those numbers.','',
        '## Sample and settings','',
        f'Selection uses `random.Random({SEED}).sample(sorted(eligible, key=id), 100)` from 2,808 eligible categorical questions. Eligibility excludes 143 noncategorical entries and 42 strict duplicates from the 2,993-record export. The seed fixes question selection, not stochastic inference. These 100 questions span {summary["families"]} operational survey families.','',
        f'The expansion covers {services} EP inference services and multiple model makers. Service is the hosting route, not necessarily the maker. Models were chosen for provider and size diversity from EP’s text-working catalog with non-fallback prices, then checked for compatibility on one question. This is not an exhaustive or random sample of models. Existing Astra, Fable, and Gemini responses were retained.','',
        '| Configuration key | Requested model | EP service | Reasoning setting |','|---|---|---|---|']
    for k,v in MODELS.items():
        p=v['parameters']
        reasoning=p.get('reasoning_effort') or p.get('output_config',{}).get('effort') or ('dynamic (-1)' if p.get('thinking_budget')==-1 else 'provider default')
        lines.append(f"| `{k}` | `{v['model']}` | `{v['service']}` | {reasoning} |")
    lines += ['',
        'All configurations request an 8,192-token output cap; provider enforcement and whether reasoning counts toward the cap can vary. Reasoning controls and temperatures differ across providers: these are not equal compute budgets. Exact parameters and rendered prompts are frozen in [protocol](data/protocol.json) and `data/*_jobs.json` / `data/*_prompts.json`. The [expansion catalog snapshot](data/catalog_expansion.json) records availability and listed prices at selection. EDSL is pinned to public commit `cfef949e4ea87ab0bf6e998ab64b54a1bfcec6e4`.','',
        'Reference shares, Aaru forecasts, source URLs, and other survey answers are excluded from prompts. Audience descriptions come from the benchmark and may contain substantive context; excluding numeric targets does not establish freedom from leakage.','',
        '## Validity and retries','',
        '![Valid answers before and after one retry](figures/validity.png)','',
        '| Model | Valid on first submission | Final valid |','|---|---:|---:|']
    for k in MODELS:
        v=summary['validity'][k]
        lines.append(f"| {LABELS[k]} | {v['first_submission_valid']}/100 | {v['final_valid']}/100 |")
    lines += ['',
        'Only questions missing a valid distribution receive one explicitly requested retry with unchanged settings. EP may also retry internally. Failed attempts are preserved in [failures](data/failures.json), with [error-type diagnostics](data/failure_diagnostics.json); no invalid answer is repaired into a forecast. Missing answers can reflect validation errors, rate limits, or infrastructure failures. Backend reports do not always include raw failed responses. Kimi’s full job was marked completed but omitted 11 validated distributions; the runner checks every question independently of job status.','',
        f'Sensitivity check: **{summary["first_submission_common_questions"]} questions** had valid first-submission answers from every configuration. This small subset is selected by output validity and may differ systematically from other questions. Restricting every method to it gives:','',
        '| Method | Mean TVD on common first-submission questions |','|---|---:|']
    for k,v in summary['first_submission_common_mean_tvd'].items():
        value='—' if v is None else f'{v:.2f}'
        lines.append(f'| {LABELS[k]} | {value} |')
    effort=summary['astra_effort_comparison'];ci=effort['paired_family_bootstrap_95ci']
    if summary['incomplete_models']:
        lines += ['', '### Incomplete runs: excluded from the main ranking','',
            'Observed-only means use each model’s valid questions and may be biased by selective failure. The bounds instead use all 100 targets, allowing each missing answer any TVD from 0 to 100; they do not impute a forecast.','',
            '| Model | Valid / 100 | Observed-only mean TVD | Aaru on same valid questions | Bounds on full-100 mean TVD |',
            '|---|---:|---:|---:|---|']
        for k,v in summary['incomplete_models'].items():
            lo,hi=v['full_sample_mean_tvd_bounds']
            observed='—' if v['mean_tvd_observed_only'] is None else f"{v['mean_tvd_observed_only']:.2f}"
            aaru='—' if v['aaru_mean_on_same_observed_questions'] is None else f"{v['aaru_mean_on_same_observed_questions']:.2f}"
            lines.append(f"| {v['label']} | {v['valid_questions']} | {observed} | {aaru} | [{lo:.2f}, {hi:.2f}] |")
    lines += ['', '### Smoke screening exclusions','',
        'These configurations were not expanded beyond the first question. Decisions used format, infrastructure, and billing checks, without scoring accuracy. One smoke failure does not establish a model’s general inability to answer the task. Smoke answers and cost records are preserved in [screening.json](data/screening.json); settings are in [protocol.json](data/protocol.json).','',
        '| Configuration | Reason for withholding the full batch |','|---|---|']
    for k,reason in SCREENED_OUT.items():lines.append(f"| {ALL_MODELS[k]['label']} (`{k}`) | {reason} |")
    lines += ['', '## GPT-6: medium versus high','',
        f'High minus medium mean TVD is **{effort["mean_difference"]:+.2f}** (95% paired family-bootstrap interval **[{ci[0]:+.2f}, {ci[1]:+.2f}]**). High wins on {effort["high_wins"]}/100 questions, medium on {effort["medium_wins"]}/100, with {effort["ties"]} ties. The interval includes zero; this sample does not establish an accuracy advantage for either effort level.','',
        'The only configured difference is `reasoning_effort`. Medium was added after inspecting high/Fable/Gemini results; retained responses were collected at different times. This is an exploratory paired comparison, not a randomized repeated experiment.','',
        '![Astra effort comparison](figures/astra_effort.png)','',
        'The separate 2,808-question medium-effort analysis used different prompts and reported GPT-6 TVD 10.11 versus Aaru 7.71. Those full-corpus numbers are not the results of this 100-question experiment.','',
        '## Costs and reconciliation','',
        '![Accuracy versus observed cost](figures/cost_accuracy.png)','',
        '| Model | Results cost | Reported EP job cost | Reconciliation |','|---|---:|---:|---|']
    for k in MODELS:
        selected_audits=[a for a in audits if a['model_key']==k]
        low=sum(a['results_cost_usd'] for a in selected_audits);high=sum(a['reported_job_cost_usd'] for a in selected_audits)
        state='discrepancy' if any(a['state']=='discrepancy' for a in selected_audits) else 'matched' if all(a['state']=='matched' for a in selected_audits) else 'unresolved'
        lines.append(f'| {LABELS[k]} | ${low:.5f} | ${high:.5f} | {state} |')
    screening=read(ROOT/'data/screening.json')
    screening_cost=sum(s['cost_audit']['reported_job_cost_usd'] or 0 for s in screening)
    lines += ['',f'Smoke-only excluded configurations add **${screening_cost:.5f}** in reported EP charges.', '',f'Total retained-run accounting: **${sum(a["results_cost_usd"] for a in audits):.2f}** in Results and **${sum(a["reported_job_cost_usd"] for a in audits):.2f}** reported by EP. These are observed costs, not fixed prices for future runs.','',
        'Results costs use recorded token-price metadata and exclude cache hits. Reported EP costs are job-accounting records, not independently verified account debits. `CostReconciliationWarning` flags differences exceeding $0.0002 per job, allowing two billing line items to round to $0.0001. Answers are saved before warning; a billing discrepancy never triggers a paid retry. See [EDSL #2668](https://github.com/expectedparrot/edsl/issues/2668). A separate [token-accounting check](data/token_accounting_concerns.json) flags cases where inclusive completion counts appear to be charged again as reasoning tokens. Results/EP agreement alone cannot detect that shared error. These flags are evidence for investigation, not independently verified account overcharges; provider estimates may also use different prices or discounts.','',
        'A separate Astra `openai_v2` adapter diagnostic returned $0.04981 in Results versus $0.61990 reported by EP. It is excluded from the accuracy comparison and cost table; [its record](data/adapter_diagnostic.json) is preserved. Add that charge when totaling all experiment spending. We chose the catalog-listed `openai` route on billing compatibility before full inference.','',
        '## Interpretation','',
        'TVD is `50 * sum(abs(predicted_share - survey_share))` for shares on a 0–1 scale, expressed in percentage points. Ten TVD points means ten percentage points of probability mass must move to match the reference. It is not a classification error rate. Questions have equal weight; reference and Aaru shares are scored as exported.','',
        'Intervals use 20,000 paired bootstrap draws over survey families (seed 20260927). Connected families share cluster, series, canonical source, or sufficiently long identical question/option text, with a documented Conference Board series join. This operational grouping is not proof of independence. Intervals exclude stochastic reruns, model selection, multiple-comparison adjustment, and human-survey sampling error.','',
        '![Paired mean differences](figures/paired_errors.png)','',
        'This retrospective benchmark uses published surveys that may occur in model training data. It does not establish prospective performance on novel questions, individual-agent validity, joint distributions, or how Aaru generated its predictions. Results apply to this seeded sample.','',
        '![Example marginal distributions](figures/examples.png)','',
        'Examples show Astra medium/high, Fable, and Gemini 3.8 Flash for legibility. They are the first six sampled questions with at most four options and ASCII wording, selected without reference to errors. All model forecasts and original source links are in the archive.','',
        '## Data and provenance','',
        'Source: [Aaru’s publication](https://aaru.com/publications/population-simulation-at-the-replication-floor-research) and [question export](https://aaru.com/data/population-simulation/questions.json). A byte-preserving source snapshot and provenance are included. [NOTICE.md](NOTICE.md) describes third-party materials. MIT covers original code only.','',
        'GitHub Actions runs tests and offline reproduction without credentials. Public response records contain answers, usage, price metadata, cache flags, and prompts; they omit provider/account identifiers and reasoning content. EP job UUIDs identify supporting private accounting records.']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n')

def reproduce(check=False):
    for name,sha in read(ROOT/'data/manifest.json')['files'].items():
        if Path(name).name!=name or hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()!=sha:
            raise ValueError(f'Archive checksum mismatch: {name}')
    questions=read(ROOT/'data/sample.json')
    if questions!=selected(): raise ValueError('Frozen sample differs from seeded reconstruction')
    forecasts=read(ROOT/'data/forecasts.json')
    rows=score(questions,forecasts,allow_incomplete=True);summary=summarize(rows)
    coverage=read(ROOT/'data/coverage.json')
    for k in MODELS:
        actual={r['id'] for r in forecasts if r['model_key']==k}
        missing={q['id'] for q in questions}-actual
        if (set(coverage[k]['missing_ids'])!=missing or coverage[k]['valid_questions']!=len(actual)
                or coverage[k]['attempted_questions']!=len(questions)):
            raise ValueError(f'{k}: coverage does not account for missing forecasts')
    summary['astra_effort_comparison']=summarize_effort(rows)
    summary['validity']={k:{'first_submission_valid':sum(r['model_key']==k and r['stage']!='retry' for r in forecasts),
                            'final_valid':sum(r['model_key']==k for r in forecasts)} for k in MODELS}
    common=set.intersection(*({r['id'] for r in forecasts if r['model_key']==k and r['stage']!='retry'} for k in MODELS))
    summary['first_submission_common_questions']=len(common)
    summary['first_submission_common_mean_tvd']={k:float(np.mean([r['tvd'] for r in rows if r['model']==k and r['id'] in common])) if common else None for k in LABELS}
    expected=ROOT/'results/expected_summary.json'
    if check and (not expected.exists() or not equivalent(read(expected),summary)): raise ValueError('Numerical summary differs from frozen expected results')
    write(ROOT/'results/summary.json',summary)
    if not expected.exists():write(expected,summary)
    with (ROOT/'results/scores.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['id','family','model','tvd'],lineterminator='\n');writer.writeheader();writer.writerows(rows)
    plots(rows,summary,questions,forecasts)
    readme(summary,read(ROOT/'data/cost_audit.json'))
    print(__import__('json').dumps(summary,indent=2))
