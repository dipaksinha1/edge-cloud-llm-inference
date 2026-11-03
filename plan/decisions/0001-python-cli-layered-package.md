# 0001 · Python CLI package with a layered design

**Context.** The deliverable is measurements for a course report, run by students on their own laptops. The work is built across many fresh AI sessions that only share files.

**Decision.** A single Python package `eclm` with four `python -m` commands (data, router, run, analyze). Layers: commands → strategies → measurement → clients. Every model sits behind one `ModelClient` protocol.

**Alternatives.** Jupyter notebooks (hard to test, easy to break between sessions). A web app (unneeded, a non-goal). A LangChain-style framework (extra abstraction that hides timing details we must measure ourselves).

**Consequences.** Mock clients make everything testable in the cloud. Contracts in ARCHITECTURE §5 must stay stable, because tasks rely on them.
