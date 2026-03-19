"""Live test: 20 Farm notes through the full promotion pipeline with real LLM."""
import sys, os, time, asyncio, json
import numpy as np
sys.path.insert(0, '.')

# Bypass __init__.py import chain
sys.modules['hololoom.apps.server'] = type(sys)('hololoom.apps.server')
sys.modules['hololoom.apps.server'].__path__ = [os.path.join(os.getcwd(), 'hololoom', 'apps', 'server')]

from hololoom import _CoreRedirectFinder
sys.meta_path.insert(0, _CoreRedirectFinder())

import importlib
vault_bridge = importlib.import_module('hololoom.apps.server.vault_bridge')
propose_note = vault_bridge.propose_note
VAULT_ROOT = vault_bridge.VAULT_ROOT

# 20 Farm notes — varying quality levels
# Good (should promote): 7 notes with claims, links, substance
# Medium (should return): 7 notes missing links or weak claims
# Bad (should escalate): 6 notes that are thin or contradictory
NOTES = [
    # === GOOD (7) ===
    ("crimson-clover-nitrogen-confirmed-year-two",
     "Second year of crimson clover trials in field 3 confirms 18% nitrogen increase. "
     "Rhizobium populations up 340% per microbiome analysis. The three-field rotation "
     "(clover-corn-fallow) is producing cumulative benefits. See [[soil-testing-protocol]] "
     "and [[field-rotation-plan]] for methodology. County extension verified results."),

    ("honey-yield-correlates-with-clover-proximity",
     "Hives within 500m of crimson clover fields produced 23% more honey in spring 2026 "
     "compared to hives near fallow fields. This suggests clover serves dual purpose: "
     "nitrogen fixation AND bee forage. See [[hive-placement-strategy]] and "
     "[[crimson-clover-nitrogen-fixation-confirmed]] for the connection."),

    ("soil-ph-affects-clover-nodulation",
     "Soil pH below 5.8 reduced Rhizobium nodulation by 60% in field 7 compared to "
     "field 3 (pH 6.4). Lime application in fall 2025 raised field 7 to 6.1 and "
     "nodulation improved. See [[soil-testing-protocol]] and [[lime-application-schedule]] "
     "for details. Optimal pH range appears to be 6.0-6.8 for crimson clover."),

    ("winter-cover-prevents-erosion-in-slope-fields",
     "Fields 4 and 5 (8-12% slope) showed 70% less topsoil loss over winter with "
     "crimson clover cover compared to bare fallow. Measured via erosion pins installed "
     "October 2025. See [[erosion-monitoring-protocol]] and [[field-rotation-plan]] "
     "for the experimental setup. This justifies cover cropping on all sloped fields."),

    ("compost-tea-boosts-microbial-diversity",
     "Weekly compost tea applications to field 2 increased soil microbial diversity index "
     "from 2.1 to 3.4 (Shannon diversity) over 3 months. Paired with [[soil-testing-protocol]] "
     "samples showing improved nutrient cycling. See [[composting-operations]] for the "
     "tea brewing recipe and application rates."),

    ("rain-gauge-network-validates-irrigation-timing",
     "The 6-station rain gauge network installed in January 2026 shows significant "
     "rainfall variation across the farm (up to 40% difference between fields 1 and 8). "
     "This validates the decision to use zone-based irrigation rather than uniform timing. "
     "See [[irrigation-schedule]] and [[weather-station-data]] for the full dataset."),

    ("chicken-tractor-rotation-improves-pasture",
     "Moving chicken tractors on 3-day rotation through pasture blocks increased grass "
     "density by 35% compared to static coops. The nitrogen from chicken manure plus "
     "the scratching behavior breaks up thatch. See [[pasture-management-plan]] and "
     "[[poultry-housing-design]] for the rotation schedule."),

    # === MEDIUM (7) — missing links or weak claims ===
    ("soil-seems-better-this-year",
     "The soil in field 3 looks darker and feels more friable than last year. "
     "I think the clover is helping but we should do more testing to be sure. "
     "The worms seem more active too which is usually a good sign for soil health."),

    ("thinking-about-adding-goats",
     "Been reading about integrated livestock and thinking goats might be good "
     "for managing the blackberry patches along the fencelines. Need to research "
     "fencing requirements and whether they'd conflict with the bee operations."),

    ("field-3-drainage-issue-noted",
     "Noticed standing water in the northwest corner of field 3 after the March rains. "
     "This could affect the clover trial if roots are waterlogged. Should install "
     "a French drain before next fall planting. See [[field-rotation-plan]] for context."),

    ("mushroom-log-inoculation-results",
     "Inoculated 50 oak logs with shiitake spawn in February. First pins appeared "
     "after 8 months which is within the expected range. Yield data not yet available. "
     "See [[mushroom-log-protocol]] and [[hardwood-sourcing]] for the setup details."),

    ("seed-saving-from-heirloom-tomatoes",
     "Saved seeds from 6 heirloom tomato varieties this season. Fermentation method "
     "for 3 days produced clean seeds with good germination test results (85-95%). "
     "No wiki links yet but should connect to the garden planning notes."),

    ("fence-repair-log-march-2026",
     "Replaced 120 feet of fence along the north boundary. Used 4-strand high-tensile "
     "wire on wooden posts. Total cost $340 in materials. Labor: 2 days with helper."),

    ("weather-has-been-unusual",
     "March was much wetter than average. The extension office says we got 6.2 inches "
     "versus the 30-year average of 3.8 inches. This affected planting schedules."),

    # === BAD (6) — thin, no links, or contradictory ===
    ("clover-doesnt-fix-nitrogen",
     "I don't think clover actually fixes nitrogen. The soil tests might be wrong. "
     "We should stop planting it."),

    ("field-notes-tuesday",
     "Walked the fields today. Everything looks fine."),

    ("random-thought",
     "What if we painted the barn red?"),

    ("todo-list-march",
     "Fix fence. Order seeds. Call vet. Clean coop."),

    ("soil-test-results",
     "Got the soil test results back. Numbers look ok."),

    ("buy-more-chickens",
     "We need more chickens."),
]

async def main():
    print("=" * 70)
    print("  LIVE TEST: 20 Farm Notes Through Full Pipeline")
    print("  Model: qwen3.5:9b (local)")
    print("=" * 70)

    # Step 1: Propose all 20 notes
    print("\n[1] PROPOSING 20 notes to Farm department...")
    proposed = []
    for title, content in NOTES:
        result = propose_note(title=title, content=content, agent="farm-worker", department="farm")
        status = result.get("status", "error")
        quality = result.get("quality_score", 0)
        proposed.append(result)
        icon = "+" if status == "proposed" else "x"
        print(f"  {icon} {title[:50]:50s} q={quality:.2f} {status}")

    n_proposed = sum(1 for r in proposed if r.get("status") == "proposed")
    print(f"\n  Proposed: {n_proposed}/20")

    # Step 2: Run the Head evaluation (REAL LLM)
    print("\n[2] EVALUATING inbox with real LLM (qwen3.5:9b)...")
    print("  This will take a few minutes...")

    start = time.time()

    # Import the evaluation function
    department_head = importlib.import_module('hololoom.apps.server.department_head')

    # We need to call evaluate_inbox which is async and needs model_router + call_model
    # Set up the model router to use local 9b for everything
    os.environ['PROMPTLY_OLLAMA_URL'] = 'http://127.0.0.1:11434'

    model_router_mod = importlib.import_module('hololoom.apps.server.model_router')
    router = model_router_mod.get_router()

    llm_mod = importlib.import_module('hololoom.apps.server.promptly.llm')
    call_model = llm_mod.call_model

    # Get candidates
    rubric = department_head.FARM_RUBRIC
    inbox_dir = VAULT_ROOT / "departments" / "farm" / "inbox"
    candidates = []
    if inbox_dir.is_dir():
        for worker_dir in inbox_dir.iterdir():
            if worker_dir.is_dir():
                for md_file in worker_dir.glob("*.md"):
                    candidates.append(md_file)

    print(f"  Found {len(candidates)} candidates in inbox")

    # Evaluate each one
    results = []
    for i, md_file in enumerate(candidates):
        print(f"  [{i+1}/{len(candidates)}] {md_file.name[:50]}...", end="", flush=True)
        try:
            result = await department_head._evaluate_single(
                md_file, "farm", rubric, router, call_model,
            )
            results.append(result)
            print(f" ->{result.action} (score={result.score:.0f})")
        except Exception as e:
            print(f" ->ERROR: {str(e)[:60]}")
            results.append(department_head.EvaluationResult(
                path=str(md_file), action="skip", score=0, reason=str(e)
            ))

    elapsed = time.time() - start

    # Step 3: Summary
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    promoted = [r for r in results if r.action == "promote"]
    returned = [r for r in results if r.action == "return"]
    escalated = [r for r in results if r.action == "escalate"]
    skipped = [r for r in results if r.action == "skip"]

    print(f"\n  Promoted:  {len(promoted)}")
    for r in promoted:
        print(f"    + {r.path.split('/')[-1][:50]} (score={r.score:.0f})")

    print(f"\n  Returned:  {len(returned)}")
    for r in returned:
        print(f"    ~ {r.path.split('/')[-1][:50]} (score={r.score:.0f})")

    print(f"\n  Escalated: {len(escalated)}")
    for r in escalated:
        print(f"    ! {r.path.split('/')[-1][:50]} (score={r.score:.0f})")

    print(f"\n  Skipped:   {len(skipped)}")

    # Math summary
    print(f"\n  Time: {elapsed:.1f}s ({elapsed/max(1,len(candidates)):.1f}s per note)")

    all_scores = [r.score for r in results if r.score > 0]
    if all_scores:
        print(f"  Scores: min={min(all_scores):.0f} max={max(all_scores):.0f} mean={np.mean(all_scores):.0f}")

        # Kalman filter the scores
        try:
            from hololoom.apps.server.math_extensions_v5 import kalman_score_filter
            filtered = kalman_score_filter(all_scores)
            print(f"  Kalman:  min={min(filtered):.0f} max={max(filtered):.0f} mean={np.mean(filtered):.0f}")
        except:
            pass

    print(f"\n  Pipeline fired: mechanical ->Hensel ->contradiction ->LLM ->causal ->gate")
    print(f"  Model: qwen3.5:9b (local Ollama)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
