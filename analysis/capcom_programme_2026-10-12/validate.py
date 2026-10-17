"""Run the current bounded CapCom validation without overwriting historical evidence.

The 12 October validator and its synthetic fixtures are preserved in git at
504096c. Its saved validation.json/artifact_manifest.json remain historical
records. Current validation uses actual reconciled S1 episodes, all 21 arms,
both evaluation phases and a fresh output directory:

    python analysis/capcom_programme_2026-10-12/validate.py \
        --baseline /path/to/checkout-at-644f8f1 --output /fresh/scratch/path
"""
from validate_fixes import main


if __name__ == "__main__":
    main()
