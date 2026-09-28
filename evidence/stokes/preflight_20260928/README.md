# User-supplied Stokes preflight observations

`user_terminal.txt` preserves the two terminal replies supplied on 28 September 2026. The exact command collection time was not supplied. The assistant has not authenticated to Stokes.

Known: Stokes `normal`, unlimited partition time, 153 nodes / 6,928 CPU slots, user/account MaxJobs=250, account `cenyioha`, 80,000 September CPU-hours remaining, empty user queue, and Anaconda 2024.10 available. The requested normal QoS limit fields have no values. An empty `quota -s` result is not a quota measurement.

`preflight_draft.json` is deliberately **verified=false**. It is a factual draft, not completed operational evidence. Use `--account=cenyioha` explicitly; setting `account_required=true` records this chosen launcher policy, not a claim that the scheduler requires the flag. Ordinary association listings include inherited Max limits unless WOPLimits is requested. Aggregate group/QoS submission limits and storage/runtime prerequisites still need the checks in the runbook before finalizing the launchable JSON.

Do not set network verification or Python version from a login-node module listing alone. Verify them on a compute node after current Gate A passes. Preserve raw output with a timestamp, fill the remaining fields, and keep final preflight separate from this historical draft. Recheck balance and occupancy before the full array and regenerate its budget if the bound preflight changes.
