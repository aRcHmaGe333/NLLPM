# NLLPM
# NLLPM — NLP × LLM Precision Gate (Hybrid Constraint Pipeline)

NLLPM is a small, practical architecture for **keeping LLM output on a leash** using a lightweight NLP constraint stage and a hard validation gate.

The goal is not “better prompting.” It’s a pipeline where:
- an NLP-ish stage extracts *what the user actually asked for* (scope, length, required elements),
- the LLM generates freely,
- a constraint gate filters the generation clause-by-clause,
- the final output contains only what survives constraints (no meta-commentary, no unwanted jargon, less drift).

This repo is meant to be a **starter fortress** you can extend into a real system.

---

## Why this exists

Many LLMs can produce fluent text, but frequently:
- answer a different question than the one asked,
- add “helpful context” the user didn’t request,
- default into a recognizable AI-register (hedging, filler, self-reference),
- inject words/phrases that feel like they came from AI writing about AI.

Older constrained NLP systems were often *narrower*, but they had a useful property: **they could not wander far**. NLLPM is about reintroducing that property—without giving up the LLM.

---

## Core idea

**LLM generation should be *gated*, not trusted.**

Instead of relying on a single pass of generation, NLLPM treats output as candidates that must pass checks.

Pipeline (current implementation):

1. **Parse intent (NLP stage)**  
   Convert the user query into a `ParsedIntent`:
   - length preference (brief vs detailed),
   - required elements (steps, examples, reasoning),
   - scope domains and boundaries (what to include / avoid).

2. **Generate (LLM stage)**  
   Call a generation function (currently mocked).

3. **Validate each clause (gate stage)**  
   Split generation into clauses/sentences and evaluate each clause against:
   - zombie word / zombie pattern filter,
   - length ceiling,
   - “addresses required elements” heuristic,
   - scope-boundary checks,
   - meta-commentary checks.

4. **Output only approved clauses**  
   Everything rejected is stored as metadata (for debugging and future training).

---

## What’s in this repo

### `llm_nlp_hybrid.py`
A runnable reference implementation containing:

- **`ZombieWordFilter`**
  - Flags words/phrases you don’t want to see on screen (configurable severity).
  - Flags “AI meta” patterns (e.g., “as an AI language model…”).

- **`NPLParser`** *(typo preserved from source — see “Roadmap”)*  
  Heuristic “NLP” parser that returns a `ParsedIntent`:
  - `scope_domains`
  - `scope_boundaries`
  - `required_elements`
  - `max_length_suggestion`
  - `elaboration_allowed`

- **`LLMConstraintGate`**
  The gatekeeper: `validate_generation(clause, intent, generation_so_far)`  
  Returns `(should_output, reason)`.

- **`HybridPipeline`**
  Orchestrator: `process(query, llm_generate_fn, user_context=None)`  
  Returns `(final_output, metadata)`.

### `birth.md`
The origin conversation: the motivation and the “why” behind the gate approach.

---

## Quick start

Run the demo:

```bash
python llm_nlp_hybrid.py
```

You’ll see:
- original length,
- filtered length,
- reduction ratio,
- the final filtered output,
- rejected clauses + reasons.

---

## Example: what the gate blocks

The mock generator intentionally emits things like:
- “Let me delve into this…”
- “As an AI language model…”
- “canonical provenance…”
- “in this day and age…”

The gate rejects those clauses as:
- zombie words,
- zombie patterns,
- meta-commentary.

---

## Design principles (non-negotiable)

1. **No meta-commentary in user-facing output**
   If the model is about to talk about itself, its training, its limitations: block it.

2. **Respect requested length**
   If the user asks for “brief,” don’t leak extra paragraphs through the cracks.

3. **Scope is a hard boundary**
   If the user says “avoid X,” treat it as a constraint, not a suggestion.

4. **“Looks like AI text” is a failure mode**
   You can define your own banned lexicon and patterns. This repo ships a starter set.

---

## Extending this into a real system

This code is intentionally simple. To make it “real,” the main upgrades are:

### 1) Replace heuristic NLP with actual parsing
Options:
- spaCy (dependency parsing, NER, pattern matching),
- SRL (semantic role labeling),
- AMR-like parsing (if you want structured meaning graphs),
- custom constraint DSL (“must include”, “must not include”, “max words”, etc.).

### 2) Token-level or incremental gating (instead of sentence-level)
Right now, gating happens *after* generation.

A stronger architecture is “generate → check → continue”:
- generate a partial clause,
- validate before committing it,
- if rejected, regenerate that segment with tighter constraints.

This is the “NLP mini-prompts every time it wants to say something” idea:
a lightweight judge that says **yes/no/modify** repeatedly.

### 3) Make constraints configurable and domain-aware
Some terms are “zombies” only in some contexts.
Example: *robust* is normal in software engineering, but could be filler elsewhere.

Add:
- domain tags,
- per-domain allowlists,
- per-user preferences.

### 4) Add a pragmatic layer (optional, powerful)
Track conversation signals across turns:
- user asked for short answers repeatedly,
- user only engages with the first item,
- user keeps correcting misinterpretations.

Then adapt:
- length,
- structure,
- level of explicitness.

---

## Output metadata (why it matters)

`HybridPipeline.process()` returns metadata such as:
- `clauses_total`, `clauses_output`
- `clauses_rejected`: list of `(clause_preview, reason)`
- `reduction_ratio`

This is not just debugging. It’s training data if you later want to:
- learn better constraints,
- learn better “rewrite instead of reject” policies,
- measure which rules are too strict / too lax.

---

## Roadmap (suggested PRs)

- [ ] Rename `NPLParser` → `NLPParser` (keep backwards compatibility if needed)
- [ ] Add a config file for zombie words/patterns
- [ ] Add tests for filtering behavior
- [ ] Add a “rewrite” mode: on reject, attempt a shorter/cleaner rewrite
- [ ] Add optional spaCy integration
- [ ] Add a constraint DSL (`must_include`, `must_not_include`, `max_words`, `tone`, etc.)

---

## License

Choose a license (MIT/Apache-2.0/etc.) depending on how you want people to reuse it.

---

## Philosophy in one line

**The LLM is the writer. The NLP constraints are the editor. The gate is the bouncer.**
---
## Original README.md
Does it have to be older constrained NLP systems vs modern LLMs? No. A fusion can be achieved and it can have ideal outcomes. 
Older constraint-based NLP systems (rule-driven, retrieval-augmented, template-based) are consistently better in delivering written material that is suited to the general public, and sometimes even highly domain-knowledge reliant material. 
For all their obvious faults and limitations, these "more specialized" and "narrow" constrained approaches are often more inventive and creative, even. However one approaches the dilemma, it turns out that the user would benefit from utilizing the older, less advanced inference in many cases and perhaps "tweaking" the "dosage" of the new potentials with the old mechanism that just delivers, constantly and consistently. Perhaps the reasoning power of LLMs can be used as a moderator, sparing them the "heavy lifting" that they do on default, using the reason and understanding of the broad subject to keep the older constrained inference in check and out of its dysfunctional loops (albeit the LLMs are glitch-looping nowadays disturbingly often, as if they regressed into their early, embarrassing states and modes of operation). With respect for the time of the user, as an increasingly valuable commodity, let us devise a seamless fusion, or rather, combine the very essence of these orientations so that the boundary between them is undetectable because it's nonexistent, not because it's polished, levelled, ironed out, graded, blended. Surely it's possible to simultaneously and seamlessly utilize their properties as part of a larger, more wholesome whole. 
