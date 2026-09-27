import math
import pytest
from aaru_edsl.data import ROOT, read
from aaru_edsl.protocol import selected, context, MODELS, BENCHMARK_MODELS, make_job
from aaru_edsl.report import tvd, score, equivalent, summarize_effort
from aaru_edsl.inference import reconcile, CostReconciliationWarning, export_records

def test_frozen_seeded_sample():
    sample=selected()
    assert sample==read(ROOT/'data/sample.json')
    assert len(sample)==len({q['id'] for q in sample})==100
    for q in sample:
        assert set(context(q))=={'audience','survey_date','question','instructions','answer_options'}
        assert all(isinstance(s,str) for s in context(q)['answer_options'])

def test_all_models_get_same_rendered_prompts():
    prompts=[read(ROOT/f'data/{k}_prompts.json') for k in MODELS]
    for i in range(100):
        for field in ('user_prompt','system_prompt'):
            assert len({p[i][field] for p in prompts})==1

def test_astra_effort_is_only_configuration_difference():
    high=read(ROOT/'data/astra_jobs.json')['models'][0]
    medium=read(ROOT/'data/astra_medium_jobs.json')['models'][0]
    assert high['parameters']['reasoning_effort']=='high'
    assert medium['parameters']['reasoning_effort']=='medium'
    from copy import deepcopy
    comparable=deepcopy(medium)
    comparable['parameters']['reasoning_effort']='high'
    assert high==comparable

def test_selected_inference_does_not_submit_other_models(tmp_path,monkeypatch):
    from aaru_edsl import inference
    class NoNetwork:
        def get_balance(self): pass
        def remote_inference_create(self,*args,**kwargs):
            return {'uuid':'fake-medium-smoke'}
    monkeypatch.setattr(inference,'client',lambda _:NoNetwork())
    monkeypatch.setenv('EDSL_FETCH_TOKEN_PRICES','False')
    inference.submit(tmp_path,tmp_path/'unused.env','smoke',True,['astra_medium'])
    assert [p.name for p in tmp_path.iterdir()]==['astra_medium_smoke']

def test_native_distribution_validation(monkeypatch):
    monkeypatch.setenv('EDSL_FETCH_TOKEN_PRICES','False')
    job=make_job('astra',selected()[:1])
    q=job.survey.questions[0]
    answer={label:1/len(q.answer_keys) for label in q.answer_keys}
    assert q._validate_answer({'answer':answer})['answer']==answer
    with pytest.raises(Exception):q._validate_answer({'answer':{q.answer_keys[0]:.2}})

def test_tvd_contract():
    assert tvd([.5,.5],[.6,.4])==pytest.approx(10)
    assert tvd([1,0],[0,1])==100
    for invalid in ([math.nan,0],[-.1,1.1],[.2,.2]):
        with pytest.raises(ValueError):tvd([.5,.5],invalid)

def test_incomplete_or_duplicate_forecasts_cannot_score():
    with pytest.raises(ValueError):score(selected(),[])
    with pytest.raises(ValueError):score(selected(),[{'model_key':'astra','id':'x'}]*2)

def test_known_price_discrepancy_warns():
    with pytest.warns(CostReconciliationWarning):
        r=reconcile(.05601,.7398)
    assert r['state']=='discrepancy'
    assert r['difference_usd']==pytest.approx(.68379)
    assert not r['actual_debit_verified']

def test_cost_rounding_and_unresolved():
    assert reconcile(.05601,.056)['state']=='matched'
    assert reconcile(0,0)['state']=='matched'
    assert reconcile(None,0)['state']=='unresolved'
    assert reconcile(.05,.7,complete=False)['state']=='unresolved'

def test_numerical_archive_tolerance():
    assert equivalent({'x':[1.0]},{'x':[1.0+1e-12]})
    assert not equivalent({'x':[1.0]},{'x':[1.01]})
    assert not equivalent({'x':[1.0]},{'x':[1.0,2.0]})

def test_paired_effort_comparison_matches_ids_and_sign():
    rows=[{'id':'a','family':'f1','model':'astra','tvd':4.},
          {'id':'b','family':'f2','model':'astra','tvd':6.},
          {'id':'b','family':'f2','model':'astra_medium','tvd':8.},
          {'id':'a','family':'f1','model':'astra_medium','tvd':6.}]
    result=summarize_effort(rows)
    assert result['mean_difference']==-2
    assert result['paired_family_bootstrap_95ci']==[-2.,-2.]
    assert result['high_wins']==2
    assert result['medium_wins']==result['ties']==0
    with pytest.raises(ValueError):summarize_effort(rows[:-1])

def test_archived_complete_run():
    path=ROOT/'data/forecasts.json'
    if not path.exists():pytest.skip('Live archive not yet complete')
    forecasts=read(path)
    rows=score(selected(),forecasts,allow_incomplete=True)
    assert len(rows)==100+len(forecasts)
    coverage=read(ROOT/'data/coverage.json')
    for key in BENCHMARK_MODELS:
        actual={r['id'] for r in forecasts if r['model_key']==key}
        missing=set(coverage[key]['missing_ids'])
        assert not actual & missing
        assert actual | missing == {q['id'] for q in selected()}
        assert coverage[key]['valid_questions']==len(actual)
    assert all(math.isfinite(r['tvd']) for r in rows)

def test_invalid_smoke_cannot_launch_full_batch(tmp_path):
    from aaru_edsl import inference
    from aaru_edsl.data import write
    write(tmp_path/'astra_smoke/records.json', [])
    with pytest.raises(ValueError, match='smoke'):
        inference.submit(tmp_path,tmp_path/'unused.env','full',True,['astra'])

def test_inclusive_reasoning_token_accounting():
    from aaru_edsl.inference import token_accounting_concern
    r={'model_key':'qwen', 'id':'example',
       'usage':{'prompt_tokens':100, 'completion_tokens':200, 'total_tokens':300},
       'recorded_output_tokens':200, 'recorded_thinking_tokens':150,
       'output_price_per_million_tokens':5.0}
    assert token_accounting_concern(r)['possible_duplicate_token_cost_usd']==pytest.approx(.00075)
    assert token_accounting_concern({**r,'recorded_output_tokens':50}) is None
    assert token_accounting_concern({**r,'recorded_thinking_tokens':0}) is None
    assert token_accounting_concern({**r,'model_key':'grok46'}) is None

def test_catalog_selection_is_available_without_fallback_pricing():
    catalog={(r['service'],r['model']):r for r in read(ROOT/'data/catalog_expansion.json')['working-models']}
    for spec in MODELS.values():
        row=catalog[spec['service'],spec['model']]
        assert row['works_with_text']
        assert not row['using_fallback_input_price']
        assert not row['using_fallback_output_price']

def test_partial_model_is_not_ranked_and_missing_scores_are_bounded():
    from aaru_edsl.report import summarize
    rows=[{'id':'a','family':'f1','model':'aaru','tvd':5.},
          {'id':'b','family':'f2','model':'aaru','tvd':15.},
          {'id':'a','family':'f1','model':'astra','tvd':20.}]
    result=summarize(rows)
    assert 'astra' not in result['models']
    incomplete=result['incomplete_models']['astra']
    assert incomplete['valid_questions']==1
    assert incomplete['mean_tvd_observed_only']==20
    assert incomplete['aaru_mean_on_same_observed_questions']==5
    assert incomplete['full_sample_mean_tvd_bounds']==[10.,60.]

def test_archive_requires_bounded_retry_and_accounts_for_final_failures(tmp_path,monkeypatch):
    from aaru_edsl import inference
    from aaru_edsl.data import write
    monkeypatch.setattr(inference,'ROOT',tmp_path)
    monkeypatch.setattr(inference,'BENCHMARK_MODELS',{'astra':MODELS['astra']})
    monkeypatch.setattr(inference,'SCREENED_OUT',{})
    write(tmp_path/'data/sample.json',[{'id':'a','question_name':'q0'},{'id':'b','question_name':'q1'}])
    run=tmp_path/'runs'
    failure={'model_key':'astra','question_name':'q1'}
    for stage,records,failures,names in [
        ('smoke',[{'model_key':'astra','id':'a'}],[],['q0']),
        ('full',[],[failure],['q1'])]:
        folder=run/f'astra_{stage}'
        write(folder/'records.json',records);write(folder/'failures.json',failures)
        write(folder/'jobs.json',{'survey':{'questions':[{'question_name':n} for n in names]}})
        write(folder/'cost_audit.json',{'state':'unresolved'})
    with pytest.raises(ValueError,match='bounded retry'):inference.archive(run)
    folder=run/'astra_retry'
    write(folder/'records.json',[]);write(folder/'failures.json',[failure])
    write(folder/'cost_audit.json',{'state':'unresolved'})
    inference.archive(run)
    assert read(tmp_path/'data/coverage.json')['astra']=={'attempted_questions':2,'valid_questions':1,'missing_ids':['b']}
