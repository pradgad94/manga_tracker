Review this pull request and report findings grouped by severity.

<automatic_review_task>
1. Read the full diff carefully before forming any opinion.
2. Group findings by severity:
   - **Must fix**: correctness bugs, security vulnerabilities, broken async
     contracts (e.g. sync I/O inside an async route), missing error handling
     at system boundaries (user input / external APIs).
   - **Should fix**: type errors mypy would catch, missing validation,
     performance issues with a measurable impact.
   - **Consider**: non-blocking suggestions; keep these brief.
3. For each finding: cite the exact file and line number, explain WHY it is
   a problem (not just what), and show a concrete corrected snippet.
4. If the diff is correct and there is nothing significant to flag, say so
   explicitly — a short "LGTM" is more useful than invented nits.
</automatic_review_task>
