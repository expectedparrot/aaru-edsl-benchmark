import math
import pytest
from aaru_edsl.data import ROOT, read
from aaru_edsl.protocol import selected, context, MODELS, make_job
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
    rows=score(selected(),read(path))
    assert len(rows)==100*(1+len(MODELS))
    assert all(math.isfinite(r['tvd']) for r in rows)
