## Refactor Plan: Model-specific Prefix Construction in `generate_demo_samples_all.py`

### Goals
- Implement model-specific prefix extraction rules using a unified, explicit API.
- Separate “events” vs “controls” per model type and construct prefixes accordingly.
- Maintain current generation flow; change only how `prefix` is computed.

### Terminology
- Notes: type == note (5) or grace-note (4).
- Expressive: type == expressive feature (3).
- Events: tokens that determine the prefix length budget.
- Controls: tokens included independently of the event-length budget.
- prefix_len: integer budget (number of events) for the prefix. For econditional, use expressive_prefix_len = max(1, prefix_len // 10).

### Model Matrix (what to include and how to count)
1) baseline_ape_20M (joint)
   - Events: notes
   - Controls: none
   - Selection: first prefix_len notes from start-of-song
   - Filtering: drop expressive tokens from the prefix slice

2) prefix_ape_20M (joint, prefix model, not conditional/econditional)
   - Events: notes
   - Controls: expressive tokens from song start up to the time of the nth (prefix_len) note
   - Selection: first prefix_len notes; include all expressive tokens occurring before (<=) the nth note’s time/index

3) anticipation_ape_20M (joint, anticipation model, not conditional/econditional)
   - Same material as (2) but ordering adapted for anticipation (see Ordering section)

4) prefix_conditional_ape_20M (conditional_expressive_total)
   - Events: notes (prefix budget)
   - Controls: all expressive tokens from the entire piece (not truncated by note boundary)
   - Selection: first prefix_len notes + all expressive tokens (full piece)

5) anticipation_conditional_ape_20M (conditional_expressive_total)
   - Same material as (4) but ordering adapted for anticipation (see Ordering section)

6) prefix_econditional_ape_20M (conditional_note_total, notes_are_controls)
   - Events: expressive tokens
   - Controls: all notes (and grace-notes) from the entire piece
   - Selection: first expressive_prefix_len expressive tokens (with expressive_prefix_len = max(1, prefix_len // 10)) + all notes/grace-notes (full piece)

7) anticipation_econditional_ape_20M (conditional_note_total, notes_are_controls)
   - Same material as (6) but ordering adapted for anticipation (see Ordering section)

### Ordering Rules (prefix vs anticipation)
- For prefix models: prefix is the initial context; ordering is chronological from start-of-song. The generated sequence follows the prefix.
- For anticipation models: still extract the same materials, but ensure the encoded conditioning aligns with the model’s anticipation setting (we already pass is_anticipation to generate). No special reordering is required beyond preserving original chronological order of the selected tokens.

### Implementation Steps
1) Add small utilities
   - `is_note(type_id)`: True if note/grace-note.
   - `is_expressive(type_id)`: True if expressive feature.
   - `select_first_n_events(seq, is_event_fn, n, type_dim, n_tokens_per_event)`: scan from SOS, pick first n events (by event_fn), return end index and count.
   - `slice_by_event_window(seq, start_idx, end_idx)`: inclusive slice respecting n_tokens_per_event.
   - `filter_by_type(seq, predicate_fn)`: keep only rows whose type satisfies predicate.
   - `gather_all_controls(seq, predicate_fn)`: gather all tokens of a control type from full sequence.

2) Derive model category flags
   - `is_prefix_model = ("prefix" in model_name)`
   - `is_anticipation_model = (model_is_anticipation)`
   - `is_conditional = ("conditional" in model_name) and ("econditional" not in model_name)`
   - `is_econditional = ("econditional" in model_name)`
   - `is_joint = (eval_type == "joint")`

3) Compute event budget
   - `note_budget = prefix_len`
   - `expr_budget = max(1, prefix_len // 10)` for econditional

4) Build per-model prefix content
   - Baseline:
     - Find first `note_budget` notes → window [SOS .. nth_note_end]
     - `prefix = filter_by_type(window, is_note)`
   - Prefix/Anticipation (non-conditional):
     - Find first `note_budget` notes → window
     - `notes_in_window = filter_by_type(window, is_note)`
     - `expressive_in_window = filter_by_type(window, is_expressive)`
     - `prefix = concat_chronologically(notes_in_window, expressive_in_window)` then sort by time/index to preserve original order
   - Conditional (expressive controls):
     - Find first `note_budget` notes → window
     - `notes_in_window = filter_by_type(window, is_note)`
     - `all_expressive = gather_all_controls(full_seq, is_expressive)`
     - `prefix = merge_and_sort_by_time([notes_in_window, all_expressive])`
   - Econditional (note controls):
     - Find first `expr_budget` expressive tokens → window_expressive
     - `expr_in_window = filter_by_type(window_expressive, is_expressive)`
     - `all_notes = gather_all_controls(full_seq, is_note)`
     - `prefix = merge_and_sort_by_time([expr_in_window, all_notes])`

5) Chronological merge
   - Implement `merge_and_sort_by_time` to interleave multi-source selections preserving original event order (stable sort by time/index; in unidimensional, use event index; in multidimensional, use the time/beat dimension available in encoding).

6) Unidimensional compatibility
   - Reuse existing `model_unidimensional_encoding_function` and `model_get_type_field` to read `type` per event.
   - Ensure selections operate on event rows, not flattened tokens; respect `n_tokens_per_event` when slicing.

7) Padding and batching
   - After constructing per-sample `prefix` arrays, `pad()` and move to device as today.

8) Configuration knobs
   - Add `--prefix_len` (already exists) and add `--expr_prefix_scale` (default 0.1) for econditional budget.
   - Optional: `--prefix_include_expressive_windowed` (bool) for non-conditional prefix/anticipation to toggle expressive inclusion policy.

9) Logging and validation
   - Log: chosen model category, budgets, counts of notes/expressive in prefix, final `prefix.shape`.
   - Assert: prefix non-empty; if empty, fall back to SOS event.
   - For econditional: warn if `expr_budget` picks 0 due to sparsity; bump to 1.

10) Non-invasive integration
   - Replace the current single-path prefix computation with a dispatcher that calls the appropriate strategy per model.
   - Keep downstream `generate` and file writing unchanged.

### Rollout
1) Implement utilities + dispatcher.
2) Wire model categories and strategies.
3) Add logging and edge-case handling.
4) Smoke-test locally across all 7 models; verify prefix sizes and content match the matrix above.


