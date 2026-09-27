"""Explicit EP-only remote inference, resumable receipts, and cost reconciliation."""
import hashlib
import json
import math
import os
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from .data import ROOT, read, write
from .protocol import MODELS, BENCHMARK_MODELS, SCREENED_OUT, make_job

class CostReconciliationWarning(UserWarning):
    pass

def client(env_file):
    from dotenv import dotenv_values
    cfg = dotenv_values(env_file)
    key = cfg.get('EXPECTED_PARROT_API_KEY') or os.environ.get('EXPECTED_PARROT_API_KEY')
    if not key:
        raise ValueError('Set EXPECTED_PARROT_API_KEY in the environment or --env file')
    os.environ['EXPECTED_PARROT_API_KEY'] = key
    os.environ['EXPECTED_PARROT_URL'] = 'https://www.expectedparrot.com'
    os.environ['EDSL_FETCH_TOKEN_PRICES'] = 'False'
    from edsl import Coop
    coop = Coop(api_key=key, url='https://www.expectedparrot.com')
    resolver = coop._resolve_server_response
    coop._resolve_server_response = lambda response, check_api_key=True: resolver(response, check_api_key=False)
    return coop

def reconcile(results_cost, reported, complete=True, emit_warning=True):
    if not complete or results_cost is None or reported is None:
        return {'state': 'unresolved', 'results_cost_usd': results_cost, 'reported_job_cost_usd': reported}
    delta = reported - results_cost
    state = 'matched' if abs(delta) <= 0.0002 else 'discrepancy'
    record = {'state': state, 'results_cost_usd': results_cost, 'reported_job_cost_usd': reported,
              'difference_usd': delta, 'absolute_tolerance_usd': 0.0002,
              'ratio': reported/results_cost if results_cost else None,
              'actual_debit_verified': False}
    if state == 'discrepancy' and emit_warning:
        warnings.warn(f'Results cost ${results_cost:.5f} differs from reported job cost ${reported:.5f}; answers are preserved.', CostReconciliationWarning)
    return record

def token_accounting_concern(record):
    """Detect inclusive completion tokens recorded again alongside reasoning.

    Restrict to routes where this pattern was observed; provider accounting
    conventions differ. This is evidence to investigate, not a verified debit.
    """
    if MODELS[record['model_key']]['service'] not in ('deep_infra', 'groq'):
        return None
    usage = record.get('usage') or {}
    prompt, completion, total = (usage.get(k) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
    thinking = record.get('recorded_thinking_tokens')
    if not all(isinstance(v, (int, float)) for v in (prompt, completion, total, thinking)):
        return None
    if not (thinking > 0 and total == prompt + completion and record.get('recorded_output_tokens') == completion):
        return None
    price = record.get('output_price_per_million_tokens')
    return {'model_key': record['model_key'], 'id': record['id'],
        'reason': 'Completion tokens reconcile total usage but are recorded alongside additional thinking tokens.',
        'completion_tokens': completion, 'thinking_tokens': thinking,
        'possible_duplicate_token_cost_usd': thinking * price / 1e6 if isinstance(price, (float, int)) else None,
        'actual_debit_verified': False}

def submit(run, env_file, stage, allow_paid, model_keys=None):
    keys = list(BENCHMARK_MODELS) if model_keys is None else list(model_keys)
    if not keys or len(set(keys)) != len(keys) or any(k not in MODELS for k in keys):
        raise ValueError('Select distinct known model configurations')
    if not allow_paid:
        print(f'Plan only: {len(keys)} selected EP configurations ({", ".join(keys)}), private. Add --allow-paid-inference to submit.')
        return
    sample = read(ROOT/'data/sample.json')
    questions = sample[:1] if stage == 'smoke' else sample[1:]
    if stage == 'full':
        for key in keys:
            if key in SCREENED_OUT:
                raise ValueError(f'{key}: full inference withheld: {SCREENED_OUT[key]}')
            path = run / f'{key}_smoke/records.json'
            if not path.exists() or len(read(path)) != 1 or read(path)[0]['id'] != sample[0]['id']:
                raise ValueError('Complete and retrieve smoke runs for selected configurations before submitting the remaining 99 questions.')
    coop = client(env_file)
    coop.get_balance()  # Authenticate before submitting anything.
    for key in keys:
        if stage == 'retry':
            failures_path = run/f'{key}_full/failures.json'
            failures = read(failures_path) if failures_path.exists() else []
            if not failures: continue
            failed_names = {f['question_name'] for f in failures}
            questions = [q for q in sample if q['question_name'] in failed_names]
        folder = run/f'{key}_{stage}'
        folder.mkdir(parents=True, exist_ok=True)
        job = make_job(key, questions)
        if (folder/'submission.json').exists():
            if read(folder/'jobs.json') != job.to_dict():
                raise ValueError(f'{folder}: saved submission uses a different protocol; choose a new run directory')
            print(f'{key}/{stage}: submission already recorded; skipping')
            continue
        if (folder/'submission_pending.json').exists():
            raise RuntimeError(f'{folder}: unresolved submission attempt. Inspect EP jobs and reconcile the receipt before resubmitting.')
        write(folder/'jobs.json', job.to_dict())
        write(folder/'submission_pending.json', {'started_at': datetime.now(timezone.utc).isoformat(),
              'model': key, 'stage': stage, 'questions': len(questions)})
        receipt = coop.remote_inference_create(job, description=f'Aaru EDSL seeded100: {key} {stage}',
            visibility='private', initial_results_visibility='private', fresh=(stage=='retry'), iterations=1, task_timeout=600)
        write(folder/'submission.json', dict(receipt))
        (folder/'submission_pending.json').unlink()
        print(json.dumps({'model': key, 'stage': stage, 'questions': len(questions), 'job_uuid': str(receipt['uuid'])}))

def export_records(data, key, names):
    questions = {q['question_name']: q for q in read(ROOT/'data/sample.json')}
    expected_prompts = {p['question_name']: p for p in read(ROOT/f'data/{key}_prompts.json')}
    assert len(data['data']) == 1
    result = data['data'][0]
    expected_model = read(ROOT/f'data/{key}_jobs.json')['models'][0]
    if result['model']['model'] != expected_model['model'] or result['model']['parameters'] != expected_model['parameters']:
        raise ValueError(f'{key}: returned model configuration differs from frozen job')
    raw = result['raw_model_response']
    records = []
    for name in names:
        q = questions[name]
        for kind in ('user_prompt','system_prompt'):
            rendered = result['prompt'][name+'_'+kind]
            if isinstance(rendered,dict): rendered = rendered['text']
            if rendered != expected_prompts[name][kind]:
                raise ValueError(f'{key}/{name}: returned {kind} differs from frozen prompt')
        answer = result['answer'].get(name)
        if result.get('validated_dict', {}).get(name+'_validated') is not True:
            raise ValueError(f'{key}/{name}: native validation did not succeed')
        labels = [o['label'] for o in q['options']]
        if not isinstance(answer, dict) or set(answer) != set(labels):
            raise ValueError(f'{key}/{name}: missing or invalid distribution; no substitute forecast is used')
        values = [answer[label] for label in labels]
        if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or not 0<=v<=1 for v in values) or abs(sum(values)-1)>1e-6:
            raise ValueError(f'{key}/{name}: invalid probabilities')
        provider = raw.get(name+'_raw_model_response') or {}
        # Whitelist observable usage/model metadata; never export reasoning content or provider IDs.
        usage = provider.get('usage') or provider.get('usage_metadata') or {}
        records.append({'id': q['id'], 'question_name': name, 'model_key': key,
            'requested_model': MODELS[key]['model'], 'returned_model': provider.get('model') or provider.get('model_version'),
            'answer': answer, 'generated_text': result.get('generated_tokens',{}).get(name+'_generated_tokens'),
            'cache_used': result.get('cache_used_dict',{}).get(name),
            'validated': result.get('validated_dict',{}).get(name+'_validated'),
            'usage': usage, 'results_cost_usd': raw.get(name+'_cost'),
            'input_price_per_million_tokens': raw.get(name+'_input_price_per_million_tokens'),
            'output_price_per_million_tokens': raw.get(name+'_output_price_per_million_tokens'),
            'recorded_input_tokens': raw.get(name+'_input_tokens'),
            'recorded_output_tokens': raw.get(name+'_output_tokens'),
            'recorded_thinking_tokens': raw.get(name+'_thinking_tokens'),
            'prompt': {k: v for k,v in result.get('prompt',{}).items() if k.startswith(name+'_')}})
    return records

def status(run, env_file):
    coop = client(env_file)
    for receipt_path in sorted(run.glob('*/submission.json')):
        folder = receipt_path.parent
        key = folder.name.rsplit('_', 1)[0]
        if key not in MODELS:
            raise ValueError(f'Unknown model configuration in receipt folder: {folder.name}')
        if (folder/'records.json').exists():
            print(folder.name + ': saved')
            continue
        receipt = read(receipt_path)
        info = dict(coop.new_remote_inference_get(job_uuid=receipt['uuid']))
        write(folder/'status.json', info)
        print(json.dumps({'run': folder.name, 'status': info['status'], 'details': info.get('latest_job_run_details')}))
        if info['status'] in ('completed','partial_failed','failed') and info.get('results_uuid'):
            result = coop.get(info['results_uuid'], expected_object_type='results')
            data = result.to_dict()
            write(folder/'results.json', data)
            job = read(folder/'jobs.json')
            names = [q['question_name'] for q in job['survey']['questions']]
            valid_names = [name for name in names if data['data'][0].get('validated_dict',{}).get(name+'_validated') is True]
            failures = [{'question_name': name, 'model_key': key, 'reason': 'Missing native validated distribution',
                         'job_uuid': str(receipt['uuid'])} for name in names if name not in valid_names]
            write(folder/'failures.json', failures)
            records = export_records(data, key, valid_names)
            write(folder/'records.json', records)
            reported = info.get('latest_job_run_details',{}).get('cost_usd')
            metadata_complete = all(isinstance(r['cache_used'],bool) and
                isinstance(r['results_cost_usd'],(float,int)) and math.isfinite(r['results_cost_usd']) and
                all(isinstance(r[k],(float,int)) and math.isfinite(r[k]) for k in
                    ('input_price_per_million_tokens','output_price_per_million_tokens')) for r in records)
            audit = reconcile(result.compute_job_cost(), reported, info['status']=='completed' and metadata_complete and not failures, emit_warning=False)
            audit['job_uuid'] = str(receipt['uuid'])
            audit['failed_questions'] = len(failures)
            write(folder/'cost_audit.json', audit)
            if info.get('latest_job_run_details',{}).get('error_report_uuid'):
                try:
                    (folder/'error_report.md').write_text(coop.get_error_report_markdown(receipt['uuid']))
                except Exception as exc:
                    warnings.warn(f'{folder.name}: Results saved, but error report retrieval failed ({type(exc).__name__}).')
            if audit['state']=='discrepancy':
                warnings.warn(f"Job {receipt['uuid']}: Results cost ${audit['results_cost_usd']:.5f} differs from reported job cost ${reported:.5f}; answers and audit saved.", CostReconciliationWarning)

def archive(run):
    forecasts, audits = [], []
    coverage = {}
    sample = read(ROOT/'data/sample.json')
    names_to_ids = {q['question_name']: q['id'] for q in sample}
    for key in BENCHMARK_MODELS:
        final_failures = []
        for stage in ('smoke','full','retry'):
            folder = run/f'{key}_{stage}'
            if stage=='retry' and not folder.exists():
                if final_failures:
                    raise ValueError(f'{key}: complete the one bounded retry before archiving')
                continue
            forecasts.extend({**r, 'stage': stage} for r in read(folder/'records.json'))
            audits.append({'model_key': key, 'stage': stage, **read(folder/'cost_audit.json')})
            stage_failures = read(folder/'failures.json') if (folder/'failures.json').exists() else []
            if stage=='smoke' and stage_failures:
                raise ValueError(f'{key}: smoke screening failed')
            final_failures = stage_failures
            expected_names = set(names_to_ids) - {sample[0]['question_name']} if stage=='full' else None
            if expected_names is not None:
                attempted = {q['question_name'] for q in read(folder/'jobs.json')['survey']['questions']}
                if attempted != expected_names:
                    raise ValueError(f'{key}: full batch did not attempt exactly the remaining 99 questions')
        coverage[key] = {'attempted_questions': len(sample),
            'valid_questions': sum(r['model_key']==key for r in forecasts),
            'missing_ids': sorted(names_to_ids[f['question_name']] for f in final_failures)}
    expected = {(key,q['id']) for key in BENCHMARK_MODELS for q in read(ROOT/'data/sample.json')}
    actual = [(r['model_key'],r['id']) for r in forecasts]
    missing = {(key,qid) for key,c in coverage.items() for qid in c['missing_ids']}
    if set(actual) & missing or set(actual) | missing != expected or len(actual)!=len(set(actual)):
        raise ValueError('Archive must account for every question with a unique forecast or a documented final failure')
    write(ROOT/'data/forecasts.json', forecasts)
    write(ROOT/'data/coverage.json', coverage)
    write(ROOT/'data/cost_audit.json', audits)
    screening = []
    for key, reason in SCREENED_OUT.items():
        folder = run/f'{key}_smoke'
        screening.append({'model_key': key, 'reason': reason,
            'records': read(folder/'records.json'), 'cost_audit': read(folder/'cost_audit.json')})
    write(ROOT/'data/screening.json', screening)
    usage_concerns = [finding for r in forecasts + [r for s in screening for r in s['records']]
                      if (finding := token_accounting_concern(r)) is not None]
    write(ROOT/'data/token_accounting_concerns.json', usage_concerns)
    failures = [r for p in sorted(run.glob('*/failures.json')) for r in read(p)]
    write(ROOT/'data/failures.json', failures)
    diagnostics = []
    for p in sorted(run.glob('*/failures.json')):
        if not read(p): continue
        folder = p.parent
        types = Counter()
        report_path = folder/'error_report.md'
        if report_path.exists():
            for line in report_path.read_text().split('## Exception Details')[0].splitlines():
                cells = [s.strip() for s in line.strip().strip('|').split('|')]
                if line.startswith('|') and len(cells)==5 and cells[-1].isdigit():
                    types[cells[0]] += int(cells[-1])
        info = read(folder/'status.json') if (folder/'status.json').exists() else {}
        key, stage = folder.name.rsplit('_',1)
        diagnostics.append({'model_key':key, 'stage':stage, 'missing_valid_distributions':len(read(p)),
            'job_status':info.get('status'), 'reported_exception_counts':dict(types),
            'error_report_saved':report_path.exists()})
    write(ROOT/'data/failure_diagnostics.json', diagnostics)
    manifest = {'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted((ROOT/'data').iterdir()) if p.is_file() and p.name!='manifest.json'}}
    write(ROOT/'data/manifest.json', manifest)
    print(f'Archived {len(forecasts)} forecasts and {len(audits)} job cost records')
