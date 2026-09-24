# Measured Benchmarks (Step 3)

All numbers below were **measured on the development machine**, not estimated.
They exist so the college report can quote real figures.

## Test machine

| Property | Value |
|---|---|
| CPU | AMD EPYC 9B14, 4 cores visible to the process |
| RAM | 15 GB total |
| GPU | none (`torch.cuda.is_available()` → `False`) |
| Python | 3.13.15 |
| PyTorch | 2.6.0+cpu (CPU-only wheel) |
| ffmpeg | 7.1.5 |

## 1. Speech recognition — faster-whisper

Test audio: 11.24 s of synthesised English speech (espeak-ng), resampled to
16 kHz mono WAV. The spoken text was:

> "John said that the project prototype should be completed by Friday.
> Sarah agreed to test the prototype on Thursday.
> The team decided to use Python for the backend."

| Metric | Value |
|---|---|
| Model | `base`, `int8`, CPU |
| Model load time (cached weights) | 2.4 s |
| Transcription time | 1.5 s |
| **Real-time factor** | **7.5× faster than real time** |
| Detected language | `en` (probability 0.871) |

Output (verbatim, no correction applied):

```
[ 0.00 ->  4.60] John said that the project prototype should be completed by Friday.
[ 4.60 ->  7.92] Sarah agreed to test the prototype on Thursday.
[ 7.92 -> 10.92] The team decided to use Python for the backend.
```

The transcript matched the spoken text exactly, so measured WER on this
sample was 0.000. This is a *single clean synthetic utterance*, so it is
**not** evidence of 0% WER on real meeting audio — real meetings have
overlapping speakers, accents, crosstalk and noise. Treat it as a smoke test,
and use the AMI corpus for a meaningful WER figure.

## 2. Zero-shot structured extraction — model comparison

Prompt: fixed system instruction + JSON schema + 3-sentence transcript, asking
for JSON only, with the rule "if a value is not stated, use 'Not specified'".

The transcript deliberately **omits Sarah's deadline** and the meeting date,
making hallucination measurable.

| Model | Params | Load | Speed | Valid JSON | Sarah deadline | Action items found |
|---|---|---|---|---|---|---|
| `google/flan-t5-base` | 248 M | 28.5 s | 0.9 s | **No** | — | 0 of 2 |
| `Qwen2.5-0.5B-Instruct` | 494 M | 17.4 s | 6.0 tok/s, 18 s | Yes | **"Not specified"** ✅ | 1 of 2 |
| `Qwen2.5-1.5B-Instruct` | 1 544 M | 66.0 s | 2.4 tok/s, 40 s | Yes | `null` ❌ | **0 of 2** |

### Findings

1. **FLAN-T5-base is unusable zero-shot for this task.** Asked for JSON it
   returned the bare string `John, Sarah, test_prototype, Python`. It is fast,
   which makes it a good *fine-tuning* candidate (Steps 12–14) — LoRA training
   is far cheaper on 248 M parameters than on 1.5 B — but it cannot be the
   Phase 3 baseline without adaptation.
2. **Larger was worse.** The 1.5 B model was 2.5× slower and returned an
   **empty `action_items` list**, missing both action items, while 0.5 B found
   one. Model size alone did not buy reliability.
3. **Instruction-tuned models are unreliable at strict schemas.** The 0.5 B
   model emitted valid JSON and correctly refused to invent a deadline, but it
   also dropped John's action item, treated the informal temporal phrase
   "Friday" as the meeting `date`, and produced a `conclusion` ("the prototype
   was successfully tested") that **contradicts the transcript**, where testing
   was only *agreed*.
4. **Conclusion:** zero-shot prompting at this model scale is a genuine
   baseline, but it fails on recall and introduces unsupported statements.
   This is the empirical motivation for our own dataset and fine-tuning, and
   for a schema-validation and repair layer around the model.

## Decision taken

Default extraction model: **`Qwen/Qwen2.5-0.5B-Instruct`** — the best
accuracy/speed trade-off measured here. The architecture keeps the model
swappable, and the surrounding validation layer does not trust model output.
