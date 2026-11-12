# Training notes

<!--
Copy this file to TRAINING_NOTES.md and fill it in AS YOU GO, not the night
before. One page is enough. It is marked on being specific and honest, not on
length: "v2 missed cans on their side, so we added 40 frames of that" is worth
more than a paragraph about how YOLO works.

Delete these comments and the example text when you're done.
-->

**Group:** <!-- names -->
**Final model file:** <!-- e.g. redbull_final.pt, and which run it came from -->

## 1. Model

<!-- Which model you fine-tuned (e.g. yolo11n.pt) and why that size.
     If you tried another size, say what you saw. -->

## 2. Settings

<!-- For the FINAL run. The values are in runs/<...>/args.yaml.
     Give the command you ran, then anything you changed from the defaults and why. -->

Command:

```
python train_cans.py ...
```

| Setting          | Value | Why |
| ---------------- | ----- | --- |
| epochs           |       |     |
| imgsz            |       |     |
| batch            |       |     |
| augmentation ... |       |     |

## 3. Data and split

<!-- Where every picture came from, how many, and how they were divided.
     Say how you made sure validation pictures weren't near-copies of training
     pictures, and which pictures (if any) you kept aside for testing only. -->

| Source                         | With a can | Without a can | Train | Valid | Test |
| ------------------------------ | ---------- | ------------- | ----- | ----- | ---- |
| e.g. provided `redbull_v5`     |            |               |       |       |      |
| e.g. our own robot frames, wk2 |            |               |       |       |      |

## 4. Validation results

<!-- Final run: the "best val" line your script printed.
     Then say, in a sentence, how far you trust these numbers and why. -->

| mAP50 | mAP50-95 | Precision | Recall |
| ----- | -------- | --------- | ------ |
|       |          |           |        |

Robot-camera frames the model never trained on:

| Frames with a can: detected / total | Frames without: false box / total | Confidence cut-off |
| ----------------------------------- | --------------------------------- | ------------------ |
|                                     |                                   |                    |

## 5. What we changed after testing on the robot

<!-- The most important section. One row per round. What did the live test
     show, what did you change because of it, and did it help? Include changes
     that DIDN'T help. -->

| Run name | What the robot test showed | What we changed | Result |
| -------- | -------------------------- | --------------- | ------ |
| v1       |                            |                 |        |
| v2       |                            |                 |        |

## 6. Known weaknesses

<!-- What does the final model still get wrong? Naming it here is better than
     the marker finding it. -->
