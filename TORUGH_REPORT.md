======================================================================
TORUGH CLASSIFICATION REPORT
  (*)< OINK! - From Trough to Prize-Winning Code
======================================================================

Files Scanned: 200
Total Issues: 4496

## (*)<  The Pig Farmer's Verdict

**Auto-Fixable (Prize Pigs)**: 138 issues
   >> Safe to fix automatically with high confidence

**Needs Review (Piglets Need Training)**: 3407 issues
   >> Require human judgment or more tests

**False Positives (Mud, Not Slop)**: 951 issues
   >> Can be safely ignored

## [TOOLS] Fix Strategy Breakdown

- **manual**: 3158
- **skip**: 951
- **template**: 222
- **ast**: 165

## [WARN] Risk Level Distribution

- **critical**: 2008
- **high**: 1298
- **medium**: 1052
- **low**: 138

## [OK] Top 10 Auto-Fixable Issues

1. **test_citation.py:206**
   - Category: copy_paste
   - Strategy: ast
   - Confidence: 1.000

2. **test_citation.py:207**
   - Category: copy_paste
   - Strategy: ast
   - Confidence: 0.945

3. **test_web_research_integration.py:74**
   - Category: copy_paste
   - Strategy: ast
   - Confidence: 0.990

4. **test_web_research_integration.py:75**
   - Category: copy_paste
   - Strategy: ast
   - Confidence: 0.990

5. **test_web_research_integration.py:76**
   - Category: copy_paste
   - Strategy: ast
   - Confidence: 0.915

6. **test_web_research_integration.py:181**
   - Category: copy_paste
   - Strategy: ast
   - Confidence: 1.000

7. **test_web_research_integration.py:182**
   - Category: copy_paste
   - Strategy: ast
   - Confidence: 1.000

8. **test_web_research_integration.py:183**
   - Category: copy_paste
   - Strategy: ast
   - Confidence: 0.945

9. **test_web_research_integration.py:212**
   - Category: copy_paste
   - Strategy: ast
   - Confidence: 0.960

10. **test_web_research_integration.py:213**
   - Category: copy_paste
   - Strategy: ast
   - Confidence: 0.960

## [REVIEW] Sample Issues Needing Review

1. **unified_api.py:272**
   - Category: copy_paste
   - Risk: medium
   - Why review: Manual fix required (medium risk, Medium confidence). Review carefully and add tests before fixing....

2. **unified_api.py:301**
   - Category: copy_paste
   - Risk: medium
   - Why review: Manual fix required (medium risk, Medium confidence). Review carefully and add tests before fixing....

3. **unified_api.py:315**
   - Category: copy_paste
   - Risk: medium
   - Why review: Manual fix required (medium risk, Medium confidence). Review carefully and add tests before fixing....

4. **unified_api.py:316**
   - Category: copy_paste
   - Risk: medium
   - Why review: Manual fix required (medium risk, Medium confidence). Review carefully and add tests before fixing....

5. **unified_api.py:411**
   - Category: copy_paste
   - Risk: medium
   - Why review: Manual fix required (medium risk, Medium confidence). Review carefully and add tests before fixing....

======================================================================
(*)< May your code be clean, your tests be green,
     and your slop be properly classified!
======================================================================