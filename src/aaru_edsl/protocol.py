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
    'astra': {'model': 'gpt-6-astra', 'service': 'openai', 'label': 'GPT-6 Astra (high)',
              'parameters': {'reasoning_effort': 'high', 'temperature': 1, 'top_p': None, 'max_tokens': 8192}},
    'astra_medium': {'model': 'gpt-6-astra', 'service': 'openai', 'label': 'GPT-6 Astra (medium)',
              'parameters': {'reasoning_effort': 'medium', 'temperature': 1, 'top_p': None, 'max_tokens': 8192}},
    'fable': {'model': 'claude-fable-5-1', 'service': 'anthropic', 'label': 'Fable 5.1',
              'parameters': {'thinking': {'type': 'adaptive', 'display': 'omitted'}, 'output_config': {'effort': 'high'}, 'max_tokens': 8192}},
    'gemini': {'model': 'gemini-3.8-flash', 'service': 'google', 'label': 'Gemini 3.8 Flash',
               'parameters': {'temperature': 1, 'topP': None, 'topK': None, 'maxOutputTokens': 8192, 'thinking_budget': -1}},
}

# Expansion chosen from EP's text-working catalog before inspecting its answers.
# Catalog-listed, non-fallback prices; a range of model makers and sizes.
for key, model, service, label in [
    ('sol', 'gpt-5.6-sol', 'openai', 'GPT-5.6 Sol'),
    ('luna', 'gpt-5.6-luna', 'openai', 'GPT-5.6 Luna'),
    ('gpt54mini', 'gpt-5.4-mini', 'openai', 'GPT-5.4 mini'),
    ('gpt41mini', 'gpt-4.1-mini', 'openai', 'GPT-4.1 mini'),
    ('opus', 'claude-opus-5', 'anthropic', 'Claude Opus 5'),
    ('sonnet', 'claude-sonnet-5', 'anthropic', 'Claude Sonnet 5'),
    ('haiku', 'claude-haiku-4-5-20251001', 'anthropic', 'Claude Haiku 4.5'),
    ('geminipro', 'gemini-3.1-pro-preview', 'google', 'Gemini 3.1 Pro preview'),
    ('geminilite', 'gemini-3.5-flash-lite', 'google', 'Gemini 3.5 Flash Lite'),
    ('gemini25', 'gemini-2.5-flash', 'google', 'Gemini 2.5 Flash'),
    ('grok46', 'grok-4.6', 'xai', 'Grok 4.6'),
    ('grok420', 'grok-4.20-0309-reasoning', 'xai', 'Grok 4.20 reasoning'),
    ('deepseekpro', 'deepseek-ai/DeepSeek-V4-Pro', 'deep_infra', 'DeepSeek V4 Pro'),
    ('deepseekflash', 'deepseek-ai/DeepSeek-V4.1-Flash', 'deep_infra', 'DeepSeek V4.1 Flash'),
    ('qwen', 'Qwen/Qwen3.8-Max', 'deep_infra', 'Qwen 3.8 Max'),
    ('minimax', 'MiniMaxAI/MiniMax-M3', 'deep_infra', 'MiniMax M3'),
    ('kimi', 'moonshotai/Kimi-K2.6', 'deep_infra', 'Kimi K2.6'),
    ('glm', 'zai-org/GLM-5.3', 'deep_infra', 'GLM 5.3'),
    ('llama', 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8', 'deep_infra', 'Llama 4 Maverick'),
    ('mistral', 'mistralai/Mistral-Small-3.2-24B-Instruct-2506', 'deep_infra', 'Mistral Small 3.2'),
    ('nemotron', 'nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B', 'deep_infra', 'Nemotron 3 Ultra'),
    ('seed', 'ByteDance/Seed-2.0-pro', 'deep_infra', 'Seed 2.0 Pro'),
    ('mimo', 'XiaomiMiMo/MiMo-V2.5-Pro', 'deep_infra', 'MiMo V2.5 Pro'),
    ('phi', 'microsoft/phi-4', 'deep_infra', 'Phi-4'),
    ('nova', 'global.amazon.nova-2-lite-v1:0', 'bedrock', 'Amazon Nova 2 Lite'),
    ('gptoss', 'openai/gpt-oss-120b', 'groq', 'GPT-OSS 120B'),
    ('gemma', 'google/gemma-4-31B-it', 'deep_infra', 'Gemma 4 31B'),
    ('llama33', 'meta-llama/Llama-3.3-70B-Instruct-Turbo', 'deep_infra', 'Llama 3.3 70B'),
]:
    parameters = {'temperature': .5, 'max_tokens': 8192}
    if service == 'openai' and key != 'gpt41mini':
        parameters = {'reasoning_effort': 'high', 'temperature': 1, 'top_p': None, 'max_tokens': 8192}
    elif key in ('opus', 'sonnet'):
        parameters = {'thinking': {'type': 'adaptive', 'display': 'omitted'}, 'output_config': {'effort': 'high'}, 'max_tokens': 8192}
    elif service == 'google':
        parameters = {'temperature': 1, 'topP': None, 'topK': None, 'maxOutputTokens': 8192, 'thinking_budget': -1}
    MODELS[key] = {'model': model, 'service': service, 'label': label, 'parameters': parameters}

SCREENED_OUT = {
    'llama33': 'EP reported JobStalledError: the single-question task stayed queued for 602 seconds. No validated answer or full batch; this is an infrastructure failure, not a measured model-format failure.',
    **{k: 'Smoke run produced no native validated distribution; no full batch submitted.'
       for k in ('gemini25', 'mistral', 'phi')},
    **{k: 'Smoke usage records suggest reasoning tokens included in completion_tokens were charged a second time; full batch withheld pending billing clarification.'
       for k in ('qwen', 'glm', 'mimo', 'nemotron', 'seed', 'gptoss')},
}
BENCHMARK_MODELS = {k: v for k, v in MODELS.items() if k not in SCREENED_OUT}

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

def prepare(model_keys=None):
    keys = list(MODELS) if model_keys is None else list(model_keys)
    if not keys or len(set(keys)) != len(keys) or any(k not in MODELS for k in keys):
        raise ValueError('Select distinct known model configurations')
    questions = selected()
    write(ROOT / 'data/sample.json', questions)
    write(ROOT / 'data/protocol.json', {'seed': SEED, 'sample_size': 100, 'eligible': 2808,
        'selection': 'random.Random(seed).sample(sorted(eligible, key=id), 100)',
        'models': MODELS, 'benchmark_models': list(BENCHMARK_MODELS), 'screened_out': SCREENED_OUT,
        'system_prompt': SYSTEM, 'question_type': 'distribution',
        'edsl_commit': 'cfef949e4ea87ab0bf6e998ab64b54a1bfcec6e4',
        'seed_scope': 'Question selection only; provider outputs are stochastic.',
        'memory': 'No cross-question memory', 'temperature_note': 'Fable adaptive thinking omits sampling parameters.',
        'reasoning_note': 'Explicit effort settings where configured; otherwise provider defaults. These are not equal compute budgets. See each frozen model configuration.',
        'extension_note': 'Astra medium and the broad model expansion were added after inspecting the completed high/Fable/Gemini run. Existing responses are retained. Catalog availability, provider diversity, and non-fallback prices determined the expansion; smoke tests assess compatibility, not accuracy. Comparisons are exploratory and use one retained draw per question per configuration.'})
    for key in keys:
        job = make_job(key, questions)
        write(ROOT / f'data/{key}_jobs.json', job.to_dict())
        prompts = job.prompts().to_dicts()
        write(ROOT / f'data/{key}_prompts.json', [{k: str(v) if 'prompt' in k else v for k, v in p.items()} for p in prompts])
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'data').iterdir()) if p.is_file() and p.name != 'manifest.json'}
    write(ROOT / 'data/manifest.json', {'files': files})
    print(f'Prepared {len(questions)} questions for {len(keys)} model configurations; seed={SEED}')
