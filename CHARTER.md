# Project Charter (skeleton, v0)

Status: pre-round draft. Finalized with the team in week 1. The gates, independence rules, and authorship policy below are commitments; the logistics details are defaults the team can tune at kickoff.

## Project

Can We Trust the Failure Detectors? A validity audit of trace-based monitoring for LLM agents. Twelve weeks, three mentees plus mentor.

## Roles

- **Codebook & statistics lead:** annotation protocol, agreement analysis, confirmatory statistics. Natural first-author trajectory for whoever drives the confirmatory analysis.
- **Tooling & detector lead:** annotation interface, detector/judge harness, reproducibility.
- **Data & release lead:** draws the study sample from the committed candidate pool by executing the preregistered selection rules; license/ToS/PII scrub; corpus packaging.
- **Project lead:** one mentee, nominated at the week-1 kickoff and confirmed by the mentor; runs day-to-day coordination.
- **Mentor:** direction and scaffolding. Written feedback on weekly updates within 72 hours; biweekly 1-hour office hours; reserved gate decisions; hands-on paper editing in weeks 11 and 12. The mentor never labels, never adjudicates, and never draws the sample.

## Cadence

- Each workstream posts a short written update weekly; the mentor replies within 72 hours.
- Blockers are flagged in the team channel and get a same-week response.
- Biweekly 1-hour office hours for open questions and prioritization.

## Gates (reserved to the mentor, decided on written evidence)

1. **Pilot gate (end of week 3):** Krippendorff's alpha of at least 0.6 on the ~30-trace pilot within two codebook revisions, else collapse to a coarser taxonomy. Timeboxed; no open-ended iteration.
2. **Week-6 checkpoint:** preregistered descope rule if minutes-per-trace runs hot. Floor: a ~100-trace stratified double-annotated core plus the detector-validity audit on it.
3. **Submission target:** ICLR 2027 workshop cycle or an SE data/benchmark track.

## Independence and blinding rules

1. Candidate pool committed with per-trace provenance before preregistration.
2. Study sample drawn by the data & release lead under preregistered selection rules; never by the mentor.
3. Annotators blind to detector outputs and injection status.
4. Judge prompts and detector configurations frozen at preregistration, before main-set annotation begins.
5. Disagreements on the double-annotated core resolved by the third annotator under the codebook's written tie-break rules; the mentor is never an adjudicator.

## Annotation budget

Annotation is capped at roughly 4–5 of each mentee's 8–10 weekly hours, so at least half of every mentee's time stays on their owned workstream. Adjudication passes are counted within the cap. Pilot minutes-per-trace is measured and published in the paper.

## Authorship

Mentee first authorship by default (expected: whoever drives the confirmatory analysis); order by contribution; mentor takes the last-author position.

## Escalation

Anything ambiguous gets posted as written options in the tracker; the mentor decides within 72 hours; deadlocks default to the preregistered plan.
