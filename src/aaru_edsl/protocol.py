"""Frozen selection and explicit EDSL model configurations."""
import hashlib
import json
import random
from .data import ROOT, SEED, read, write, reconstruct

SYSTEM = ('Estimate the probability that a randomly selected member of the described survey population selects each answer. '
          'Account for the population, location and survey date. '
          'Return a JSON object mapping each exact answer label to its probability. '
          'Use nonnegative numbers summing to 1.')
MODELS = {
    'astra': {'model': 'gpt-6-astra', 'service': 'openai', 'label': 'GPT-6 Astra',
              'parameters': {'reasoning_effort': 'high', 'temperature': 1, 'top_p': None, 'max_tokens': 8192}},
    'fable': {'model': 'claude-fable-5-1', 'service': 'anthropic', 'label': 'Fable 5.1',
              'parameters': {'thinking': {'type': 'adaptive', 'display': 'omitted'}, 'output_config': {'effort': 'high'}, 'max_tokens': 8192}},
    'gemini': {'model': 'gemini-3.8-flash', 'service': 'google', 'label': 'Gemini 3.8 Flash',
               'parameters': {'temperature': 1, 'topP': None, 'topK': None, 'maxOutputTokens': 8192, 'thinking_budget': -1}},
}

def selected():
    source = read(ROOT / 'data/aaru_questions.json.gz')
    eligible, families, excluded = reconstruct(source)
    assert len(eligible) == 2808
    sample = random.Random(SEED).sample(sorted(eligible, key=lambda q: q['id']), 100)
    groups = {r['id']: r['group'] for r in families}
    return [{**q, 'family': groups[q['id']], 'question_name': f'marginal_{i:03d}'} for i, q in enumerate(sample)]

def context(q):
    return {'audience': q['audience'], 'survey_date': q['date'][:10], 'question': q['prompt'],
            'instructions': q['instructions'], 'answer_options': [o['label'] for o in q['options']]}

def make_job(key, questions):
    from edsl import Agent, Model, QuestionDistribution, Survey
    spec = MODELS[key]
    survey = Survey([QuestionDistribution(question_name=q['question_name'],
                      question_text=json.dumps(context(q), ensure_ascii=False),
                      question_options=[o['label'] for o in q['options']], include_comment=False)
                     for q in questions])
    assert all(not survey.memory_plan.get(q.question_name) for q in survey.questions)
    agent = Agent(traits={'role': 'forecaster'}, instruction=SYSTEM, traits_presentation_template='')
    model = Model(spec['model'], service_name=spec['service'], **spec['parameters'])
    return survey.by(agent).by(model)

def prepare():
    questions = selected()
    write(ROOT / 'data/sample.json', questions)
    write(ROOT / 'data/protocol.json', {'seed': SEED, 'sample_size': 100, 'eligible': 2808,
        'selection': 'random.Random(seed).sample(sorted(eligible, key=id), 100)',
        'models': MODELS, 'system_prompt': SYSTEM, 'question_type': 'distribution',
        'edsl_commit': 'cfef949e4ea87ab0bf6e998ab64b54a1bfcec6e4',
        'seed_scope': 'Question selection only; provider outputs are stochastic.',
        'memory': 'No cross-question memory', 'temperature_note': 'Fable adaptive thinking omits sampling parameters.',
        'reasoning_note': 'Astra and Fable high; Gemini dynamic thinking budget (-1). These are not equal compute budgets.'})
    for key in MODELS:
        job = make_job(key, questions)
        write(ROOT / f'data/{key}_jobs.json', job.to_dict())
        prompts = job.prompts().to_dicts()
        write(ROOT / f'data/{key}_prompts.json', [{k: str(v) if 'prompt' in k else v for k, v in p.items()} for p in prompts])
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'data').iterdir()) if p.is_file() and p.name != 'manifest.json'}
    write(ROOT / 'data/manifest.json', {'files': files})
    print(f'Prepared {len(questions)} questions for {len(MODELS)} models; seed={SEED}')
