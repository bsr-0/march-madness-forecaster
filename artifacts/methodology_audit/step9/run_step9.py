"""Step 9 driver: the repository's qualification-audit script, pointed at (a) the CURRENT canonical
production log for selection parity and (b) a fresh output directory. Nothing else is changed."""
import sys, json, hashlib, subprocess, datetime, glob, pathlib
sys.path.insert(0, '.')
import scripts.referee_qualification_audit as Q
import scripts.referee_robustness_audit as R
from src.evaluation import referee_audit as ra
ROOT = pathlib.Path('.')
out = ROOT / 'artifacts/methodology_audit/step9'
canon = ROOT / 'artifacts/methodology_audit/step5/path3_backtest_after_F5_1.txt'
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]
manifest = {
    'generated_at': datetime.datetime.utcnow().isoformat(),
    'git_head': subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),
    'working_tree_dirty': bool(subprocess.run(['git','status','--porcelain'],capture_output=True,text=True).stdout.strip()),
    'qualification_v2_sha256_16': sha('artifacts/methodology_audit/step8/qualification_v2.json'),
    'referee_set': json.load(open('artifacts/methodology_audit/step8/qualification_v2.json'))['gate'] and [r for r in ra.ALL_REFEREE_ORDER if json.load(open('artifacts/methodology_audit/step8/qualification_v2.json'))['gate'].get(r,{}).get('primary_eligible')],
    'independent_referee': json.load(open('artifacts/methodology_audit/step8/qualification_v2.json'))['independent_referee'],
    'candidate_artifacts_sha256_16': {pathlib.Path(p).name: sha(p) for p in sorted(glob.glob('artifacts/candidates/candidates_*.json'))},
    'canonical_selection_log': str(canon), 'canonical_selection_log_sha256_16': sha(canon),
    'criteria': ra.CRITERIA, 'qualification_rule': ra.QUALIFICATION,
    'opponent_config': 'draw_selection_trials: pool pick shares (real pool 2023-2026 else ESPN; 2012 static seed rates), n_opponents = pool size - 1, chalk_noise 0, referee noise 0.16 logit',
    'p1_definition': 'expected first-place share, ties split (Step 4 F4-1)',
}
json.dump(manifest, open(out/'run_manifest.json','w'), indent=1, default=str)
print("manifest written:", {k: manifest[k] for k in ('git_head','qualification_v2_sha256_16','referee_set','independent_referee')})
R.CANONICAL_LOG = canon; Q.CANONICAL_LOG = canon
sys.argv = ['referee_qualification_audit', '--out-dir', str(out)]
raise SystemExit(Q.main())
