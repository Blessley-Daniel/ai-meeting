# Project Documentation

This directory holds the material needed to write up the project for
assessment. It is written to be technically defensible: claims are limited to
what was measured, pretrained components and project contributions are kept
distinct, and limitations are stated rather than buried.

| Document | Contents |
|---|---|
| `evaluation.md` | Measured results, and an honest reading of them |
| `benchmarks.md` | Runtime and model-selection measurements on CPU |

## 1. Problem statement

Meeting recordings are easy to produce and hard to convert into a usable
record. Someone has to watch or re-listen, identify what was discussed, what
was decided and who agreed to do what, and write it up. This is slow, and the
result is inconsistent between people and between meetings.

The goal of this project is a working system that takes a recorded meeting as
input and produces structured Minutes of Meeting automatically, running
entirely on local, open-source models with no paid API dependency.

## 2. Objectives

1. Accept a recorded meeting in common audio or video formats.
2. Extract the audio and transcribe it with a pretrained speech-recognition
   model.
3. Extract structured information from the transcript: title, date, time,
   participants, discussion points, decisions, action items with owner and
   deadline, and a conclusion.
4. Constrain the output to a fixed schema, and represent absent information as
   "Not specified" rather than inventing a value.
5. Render human-readable Minutes of Meeting, viewable in a web interface and
   downloadable as PDF and DOCX.
6. Persist results so past meetings can be revisited.
7. Build a dataset for the extraction task and evaluate the system with
   standard metrics.
8. Investigate whether parameter-efficient fine-tuning improves extraction.

## 3. Scope

**In scope**

* Single-file recordings of one meeting.
* Automatic speech recognition.
* Structured information extraction from the transcript.
* Document generation and local history.

**Out of scope, and why**

* **Speaker diarisation.** Separating "who spoke when" requires an additional
  model and audio-quality assumptions that a college project cannot validate.
  Participants are instead inferred from names used in the transcript, which is
  imperfect and is noted as a limitation.
* **Multi-file or multi-meeting merging.**
* **Live transcription.** The pipeline is batch, not streaming.
* **Speaker-attributed transcripts.** Each action item's owner comes from the
  sentence content ("Rahul will…"), not from knowing who was speaking.

## 4. Methodology

The system is a linear pipeline, built as independent modules with a shared
data model so each stage can be tested and demonstrated on its own.

```
Recording
   ↓  ffmpeg                       [module: audio extraction]
16 kHz mono WAV
   ↓  faster-whisper               [module: speech recognition]
Timestamped transcript
   ↓  language model + rule layer  [module: information extraction]
Validated structured JSON
   ↓  renderer                     [module: minutes generation]
Minutes of Meeting
   ↓  reportlab / python-docx      [module: export]
PDF / DOCX          + SQLite history [module: persistence]
```

### Design choices, with reasons

**Why a hybrid model-plus-rules extraction layer rather than a model alone.**
Language models are good at open-ended synthesis and unreliable at exact
attribution. Measured on this project's data, a 0.5B model missed action items
and sometimes dropped deadlines. Decisions and action items follow recognisable
linguistic patterns ("the team decided to…", "X will… by…"), and a rule layer
handles those deterministically and reproducibly. The measured result is that
rules reach F1 1.00 on those fields while contributing nothing to discussion
points or the overview, where the model is the only source. Each component is
used where it is measurably better. See `evaluation.md`.

**Why a grounding check.** The requirement is that the system must not
hallucinate. Prompt instructions alone are not a guarantee: a model can still
emit a plausible name or date that appears nowhere in the transcript. Every
free-text claim is therefore checked against the transcript and reset to "Not
specified" if it cannot be traced back. This is enforced in code, not requested
in a prompt.

**Why greedy decoding.** `do_sample=False` makes extraction deterministic, so
the same recording produces the same minutes. Reproducibility matters more here
than variety.

**Why SQLite.** Single-user, local, no server to run. Suitable for a project of
this size, and trivially replaceable.

## 5. AI components: pretrained vs. project contribution

This distinction is central to the project and is stated explicitly.

| Component | Origin | Trained here? |
|---|---|---|
| Speech recognition (faster-whisper) | Pretrained on large public corpora | No |
| Text embeddings (MiniLM, evaluation only) | Pretrained | No |
| Base extraction model (Qwen2.5-0.5B-Instruct) | Pretrained | No |
| **LoRA adapter on the extraction model** | **Trained on our dataset** | **Yes (optional)** |
| Extraction prompt and JSON schema | Written for this project | Not learned |
| Grounding and validation logic | Written for this project | Not learned |
| Deterministic rule layer | Written for this project | Not learned |
| MoM renderer, exporters, web app | Written for this project | Not learned |

The project does **not** train a speech or language model from scratch. That is
not feasible at this scale and would not be a sensible use of the resources
available. What the project contributes is the task formulation, the dataset,
the extraction and validation logic that makes model output reliable, and an
optional fine-tuned adapter.

## 6. Dataset

`datasets/processed/` contains 48 synthetic transcript → structured-minutes
examples, generated by `training/prepare_dataset.py` from six hand-written
meeting scenarios, split by scenario (see `datasets/README.md`).

| Split | Examples | Scenarios |
|---|---|---|
| train | 24 | 3 |
| validation | 8 | 1 |
| test | 16 | 2 |

**The dataset is synthetic and must be described as such.** It is generated,
not collected. Splits are grouped by scenario rather than by example: variants
of a scenario paraphrase the same transcript, so a per-example split would put
near-duplicates in training and test, and the evaluation would partly measure
memorisation.

Splitting by scenario tests generalisation to an unseen meeting rather than
memorisation of a seen one. This is the more demanding and more honest test.

Limitations, restated because they bound every result: no disfluencies, no
overlapping speech, no speech-recognition errors, and only six scenarios. See
`datasets/README.md` for the full statement and for how public corpora
(AMI, ICSI) could be used to obtain real-audio numbers.

## 7. Algorithms and models

| Stage | Choice | Reason |
|---|---|---|
| Speech recognition | faster-whisper `base`, int8 | Whisper is the standard open ASR model; `base` runs at ~7.5× real time on this CPU, which makes a demo practical |
| Extraction | Qwen2.5-0.5B-Instruct | Selected by measurement: it emitted schema-valid JSON and handled missing fields better than the alternatives tested, and is small enough for CPU inference (`benchmarks.md`) |
| Fine-tuning | LoRA / PEFT, r=8 | Updates 0.22% of parameters, so it is feasible on a CPU-only machine; full fine-tuning is not |
| Evaluation embeddings | all-MiniLM-L6-v2 | Small sentence-embedding model for optional semantic-similarity scoring |

**Prompting.** The model is given a system instruction defining its role and
the rule that absent values must be "Not specified", followed by the JSON
schema and the transcript, and is asked for JSON only. Output is then parsed,
repaired if malformed, validated against the schema, and grounded against the
transcript.

## 8. Training / fine-tuning process

`training/train.py` implements LoRA fine-tuning:

1. Each dataset record becomes a chat example: system instruction, user prompt
   (the same prompt builder the application uses), assistant reply (the target
   JSON).
2. Loss is computed on the assistant reply only. Prompt tokens are masked with
   `-100`. Training on the prompt would teach the model to reproduce
   transcripts, which is not the task.
3. LoRA adapters are attached to the attention projection layers
   (`q_proj`, `k_proj`, `v_proj`, `o_proj`), which is where task-specific
   behaviour is cheapest to steer.
4. Optimiser state is kept only for trainable parameters.

The same prompt builder is imported from the application rather than copied, so
the training instruction cannot drift from the inference instruction.

**Result: the mechanism works, but improvement is not demonstrated.** A 4-step
smoke run halved the training loss (0.44 → 0.23). That shows the pipeline
learns; it does not show the extracted fields improve. Because the rule layer
already reaches F1 1.00 on decisions and action items, the field where an
adapter could plausibly help is discussion points (F1 0.19). Running that
comparison is the identified next experiment, and a null result would be a
legitimate finding to report.

## 9. Evaluation metrics

| Metric | Stage | Definition and justification |
|---|---|---|
| Word Error Rate | ASR | Word-level edit distance divided by reference length. The standard ASR measure. |
| Precision, recall, F1 | Field extraction | Set comparison between predicted and reference items, so a duplicate is not counted twice. |
| Field accuracy | Action-item attribution | Fraction of matched action items with the correct person / deadline. Graded only where the task was matched, so attribution errors are measured separately from detection errors. |
| ROUGE-L | Overview, conclusion | Longest-common-subsequence overlap with the reference. A proxy for summary quality, and a weak one for paraphrase. |
| Semantic similarity | Overview, conclusion | Cosine similarity of sentence embeddings; catches correct paraphrase that ROUGE penalises. |

Full results and interpretation are in `evaluation.md`. Summary, measured on
the held-out test split:

* Decisions F1 1.00, action items F1 1.00, participants F1 0.95.
* Action-item person and deadline attribution 1.00.
* Discussion points F1 0.19; overview ROUGE-L 0.32.
* WER 0.000 on one clean synthetic utterance (a smoke test, not a real-audio
  result).

## 10. Results and interpretation

The headline result is that the deterministic rule layer is highly accurate on
this data for decisions and action items, and that the language model is what
makes discussion points and the overview possible at all — the rules alone
produce neither. This is the clearest justification for the hybrid design.

The weak result is discussion-point extraction. Two explanations are
consistent with the data — the field is hard for a small model, or the metric
is harsh for accepted paraphrase — and the semantic-similarity metric is the
instrument that distinguishes them. Reporting one explanation without running
that test would be a guess.

Full reading, including the misleading cases (title accuracy, participant
false positives), is in `evaluation.md`.

## 11. Limitations

1. **Synthetic evaluation data.** Clean, templated text; no ASR errors; six
   scenarios. All extraction figures are upper bounds and none describe real
   meeting audio.
2. **No real-audio WER.** Reported WER is from one clean synthetic utterance.
   A meaningful figure needs a corpus such as AMI.
3. **No speaker diarisation.** Participants are inferred from names in the
   text. A mention is not attendance; two people with the same name are not
   separated.
4. **English only.** The extraction prompt and rules are English; Whisper
   itself is multilingual.
5. **Small models.** A 0.5B model was chosen for CPU feasibility. Larger models
   would likely extract better at a cost in runtime and memory.
6. **Small test set.** 16 examples, so small differences between configurations
   are noise rather than signal.
7. **Fine-tuning benefit unproven.** The pipeline runs; an improvement in the
   extracted fields has not been demonstrated.

## 12. Future enhancements

1. Run the semantic-similarity evaluation to settle whether discussion-point
   quality is limited by the model or by the metric.
2. Complete the fine-tuning comparison (more steps, with and without the
   adapter) and report the outcome either way.
3. Add speaker diarisation so statements can be attributed to a speaker track
   rather than to names mentioned in text.
4. Evaluate on a licence-permitting subset of AMI to obtain a real-audio WER
   and extraction figures.
5. Extend the date and time parsing to resolve relative deadlines ("Friday")
   against the meeting date into absolute dates.
6. Add a correction interface so a user can edit the minutes and export the
   edited version, which is how real meeting notes are produced.

## 13. Reproducing the project

See the root `README.md` for installation. To reproduce the reported numbers:

```bash
python training/prepare_dataset.py
python training/evaluate.py --split test --method rules  --no-semantic
python training/evaluate.py --split test --method hybrid --no-semantic
python training/train.py --max-steps 4 --limit 8 --grad-accum 2
```

## 14. Honesty statement

Every figure reported in this project comes from a script in this repository
and can be reproduced with the commands above. The dataset is synthetic and is
labelled as such throughout. Pretrained models are identified as pretrained,
and the parts built for this project are identified separately. Where a result
is weak (discussion points) or unproven (fine-tuning), it is stated as weak or
unproven. No claim is made that the system is fully accurate or that it
outperforms existing work.
