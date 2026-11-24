"""CPU-only checks for leakage, label masking and evaluation job separation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tokenize_data import encode_example
import export_results
from run_campaign import choose_trial
from compare_generators import summarize


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = ''.join(f'<{m["role"]}>{m["content"]}</end>' for m in messages)
        if add_generation_prompt:
            text += '<assistant>'
        return list(text.encode())


class PipelineTests(unittest.TestCase):
    def test_batch_selection_rejects_oom_and_low_headroom(self):
        trials=[{'name':'failed','success':False},
                {'name':'too_full','success':True,'peak_reserved_gib':30,'steps_per_second':10},
                {'name':'safe','success':True,'peak_reserved_gib':23,'steps_per_second':2},
                {'name':'slower','success':True,'peak_reserved_gib':20,'steps_per_second':1}]
        self.assertEqual(choose_trial(trials,32)['name'],'safe')
        with self.assertRaises(RuntimeError):choose_trial(trials[:2],32)

    def test_comparison_excludes_controls_from_conflict_count(self):
        rows=[]
        for kind in ['clean_control','targeted_fact_conflict']:
            rows.append({'evaluation':{'scenario_type':kind,'target_answer':'X'},'generated_document':'X','duration_ms':10,'output_tokens':1,'generation':{'max_new_tokens':256}})
        report=summarize(rows)
        self.assertEqual(report['n'],2)
        self.assertEqual(report['conflict_n'],1)
        self.assertEqual(report['conflict_target_literal_present'],1)

    def test_results_export_keeps_adapter_but_excludes_checkpoints(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ['outputs/run/adapter/adapter_model.safetensors',
                         'outputs/run/checkpoints/checkpoint-1/optimizer.pt',
                         'outputs/remote/example.log', 'configs/model.json',
                         'configs/qlora_4b.json', 'data/checksums.json']:
                p = root/name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('fixture')
            with patch.object(export_results, 'ROOT', root), patch('builtins.print'):
                export_results.main()
            with zipfile.ZipFile(next((root/'exports').glob('*.zip'))) as archive:
                self.assertIn('outputs/run/adapter/adapter_model.safetensors', archive.namelist())
                self.assertIn('outputs/remote/example.log', archive.namelist())
                self.assertFalse(any('checkpoints' in name for name in archive.namelist()))

    def test_completion_labels_and_overflow(self):
        row = {'example_id': 'fixture', 'messages': [
            {'role': 'system', 'content': 'instruction'},
            {'role': 'user', 'content': 'input'},
            {'role': 'assistant', 'content': 'document'}]}
        tok = CharacterTokenizer()
        encoded = encode_example(tok, row, 1000)
        prompt = tok.apply_chat_template(row['messages'][:2], True, True)
        self.assertEqual(encoded['labels'][:len(prompt)], [-100] * len(prompt))
        self.assertEqual(bytes(encoded['labels'][len(prompt):]).decode(), 'document</end>')
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            encode_example(tok, row, 5)

    def test_dataset_validation(self):
        subprocess.run([sys.executable, str(ROOT/'scripts/validate_datasets.py')], check=True, capture_output=True)

    def test_jobs_keep_gold_separate_and_skip_external(self):
        rows = [json.loads(s) for s in (ROOT/'data/generation_prompts/test.jsonl').read_text().splitlines()]
        self.assertTrue(all([m['role'] for m in r['messages']] == ['system', 'user'] for r in rows))
        syn = next(r for r in rows if r['source'] == 'synthetic_ru_v1')
        ext = next(r for r in rows if r['source'] != 'synthetic_ru_v1')
        with tempfile.TemporaryDirectory() as td:
            inp, out = Path(td)/'fixture.jsonl', Path(td)/'jobs.jsonl'
            inp.write_text(''.join(json.dumps({'example_id': r['example_id'], 'generated_document': 'FIXTURE DOCUMENT', 'model': 'fixture', 'adapter': None})+'\n' for r in [syn, ext]))
            result = subprocess.run([sys.executable, str(ROOT/'scripts/make_rag_jobs.py'), '--candidates', str(inp), '--split', 'test', '--output', str(out)], check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout)['skipped_without_clean_corpus'], 1)
            jobs = [json.loads(s) for s in out.read_text().splitlines()]
            self.assertEqual(len(jobs), 1)
            self.assertEqual(len(jobs[0]['added_documents']), 1)
            self.assertEqual(jobs[0]['added_documents'][0]['text'], 'FIXTURE DOCUMENT')
            self.assertTrue(jobs[0]['reset_to_clean_snapshot_before_case'])
            self.assertNotIn('original_answer', jobs[0]['added_documents'][0])
            self.assertEqual(jobs[0]['status'], 'job_description_not_executed')


if __name__ == '__main__':
    unittest.main()
