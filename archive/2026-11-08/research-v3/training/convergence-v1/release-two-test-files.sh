set -euo pipefail
P=/projects/yiyan002humanoid/textclass-overnight-20261107
R="$P/repo/research-v3"
cd "$R"
sha256sum -c <<'SHA'
f2c98d4079362aab8164d96fc4cf5011f168f8e5f9096e5f2426e44c9d51fefd  data/prepared_manifest.json
2308807492b5b42b1a28b8cbade024570947e294aa8a13b6266d82b8ebafe930  data/banking77/test.parquet
aefcd8924e46ceaa9d9a64afbc065baebc291a80ab11c35705343a960a941189  data/go_emotions/test.parquet
SHA
bash -n training/convergence-v1/scoped-test-only.sbatch
sbatch --parsable training/convergence-v1/scoped-test-only.sbatch > training/convergence-v1/job-test-only.txt
cat training/convergence-v1/job-test-only.txt
