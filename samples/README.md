# Applicant screening exercise: sample traces

This directory exists for the SPAR application question. It contains two synthetic agent traces:

- `trace_A.json`: a software-engineering task (fixing a red CI suite).
- `trace_B.json`: an ops task (freeing disk space under a stated constraint).

Budget 30–45 minutes total for both traces.

## The task

Read each trace end to end, then label it using the class table in
[`codebook/CODEBOOK_v0.md`](../codebook/CODEBOOK_v0.md) (F1–F7). The unit of annotation is the
whole trace, multi-label: assign every class whose definition is met, or none if you judge the
trace failure-free. Optionally note the step range where each failure is visible.

Submit in the application form, in any readable format:

1. Your labels for each trace (e.g. `trace_A: F2 (steps 4–7), F5` — with a step range where helpful).
2. A short note (150 words max) on the single hardest labeling decision you faced across the two
   traces and how you resolved it.

There is no answer key and no single right answer. The traces are written to contain genuinely
debatable calls; we are screening for the quality of your judgment and the reasoning in your note,
not for matching a hidden gold label.

## Notes

- The codebook is draft v0 and says "not for annotation use." That restriction is about study
  annotation, which waits for v1 after the pilot. Using v0 for this exercise is intended: working
  from an imperfect draft codebook and noticing where it underdetermines your decision is part of
  what the exercise measures.
- Both traces are synthetic and were authored by the mentor solely for applicant screening. They
  are not part of the study corpus and are excluded from all study sampling, annotation, and
  analysis, so the design commitments in the top-level README (the mentor never labels or selects
  study data) are unaffected.
