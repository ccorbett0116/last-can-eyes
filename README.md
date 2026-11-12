# The Last Can — Part 2: The Eyes (15 marks)

> _The rover's camera works. It just doesn't know what it's looking at. So far
> it has tried to drink a fire extinguisher._

Your job: **train a YOLO model that finds the Red Bull can in the robot's
camera image**, and hand your group a weights file their GUI can load.

This is the largest part of the lab (15 of 35 marks), and it is the one part
that needs no robot for most of the work. It is written for people who have
never trained a neural network.

The assignment as a whole (mission, rules, setup, attendance) is in the
[main repository](https://github.com/ccorbett0116/last-can). Do its "Getting
started" section first: you need the virtual environment and a PyTorch that
can see the GPU before anything here will run at a sensible speed.

---

## Contents

1. [What's in this repository](#1-whats-in-this-repository)
2. [How this part is marked](#2-how-this-part-is-marked)
3. [Step by step](#3-step-by-step)
4. [Running the sample](#4-running-the-sample)
5. [Getting the can photos](#5-getting-the-can-photos)
6. [Adding your own photos](#6-adding-your-own-photos)
7. [Testing: saved frames first, then live](#7-testing-saved-frames-first-then-live)
8. [What to hand over, and what to submit](#8-what-to-hand-over-and-what-to-submit)
9. [Use the GPU](#9-use-the-gpu)
10. [A plan for three sessions](#10-a-plan-for-three-sessions)
11. [When it goes wrong](#11-when-it-goes-wrong)

---

## 1. What's in this repository

| File                                                     | What it is                                                                                         |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| [PRIMER.md](PRIMER.md)                                   | What a CNN, a detector, an epoch and a validation set are. **Start here if this is new to you**    |
| [train_mnist.py](train_mnist.py)                         | A complete, working YOLO training script, for handwritten digits. Not for cans                     |
| [FROM_DIGITS_TO_CANS.md](FROM_DIGITS_TO_CANS.md)         | What has to change in that script to train on cans, and why                                        |
| [TRAINING_NOTES_template.md](TRAINING_NOTES_template.md) | The headings for the notes you hand in. Copy it to `TRAINING_NOTES.md` and fill it in as you go    |

**You don't get** a trained model or a training script for cans. Those are the
assignment. The can photos are a separate download: see
[section 5](#5-getting-the-can-photos).

---

## 2. How this part is marked

13 of the 15 marks are given **live, with your model running on the robot's
camera**. The other 2 are for your training script and notes.

| Criterion                      | Marks | Full marks when                                                                                                     |
| ------------------------------ | ----- | ------------------------------------------------------------------------------------------------------------------- |
| Recognises the can             | 5     | A single target can in plain view is detected reliably, with a steady box that fits it, at the distances the robot works at |
| Different angles and positions | 4     | Still detected when turned around, lying on its side, near, far, and toward the edges of the frame                   |
| Several cans at once           | 3     | Each target can in the frame gets its own box; none are missed or merged                                            |
| Rejects the wrong cans         | 1     | Other kinds of Red Bull can, and other objects, get **no** box, while the target can in the same view still does    |
| Training process               | 2     | A working training script for a YOLO detection model, and a `TRAINING_NOTES.md`                                     |

For the four live criteria: works nearly every time = all the marks; works in
the easy cases only, or the box flickers on and off = about half; rarely
works, or the model wasn't trained by your group = none. The complete rubric
is in the [main repository](https://github.com/ccorbett0116/last-can/blob/main/RUBRIC.md).

Read that table as a list of **pictures your training data must contain**:

| The test                         | So your training pictures need                                                                    |
| -------------------------------- | ------------------------------------------------------------------------------------------------- |
| A steady box, at working distance | Cans seen **through the robot's camera**, from the robot's height, at the distances it drives at |
| Turned around, on its side       | The can from every side, upright and lying down                                                   |
| Near, far, edges of the frame    | Close cans, small distant cans, and cans half out of the picture                                  |
| Several cans                     | Pictures with two, three or more cans, some of them touching or overlapping                       |
| No box on the wrong can          | Pictures of other cans and can-like objects, **with no box on them**                              |
| While the robot is moving        | Frames saved while driving, slightly blurred, not only while parked                               |

That last one is in the rubric for Part 3: the model has to detect the can
"both while the robot is moving and once it has stopped".

---

## 3. Step by step

1. **Set up** (main repository, "Getting started"). Check that
   `torch.cuda.is_available()` prints `True` on the PC you'll train on.
2. **Read [PRIMER.md](PRIMER.md)** if you haven't trained a model before.
3. **Run the sample** ([section 4](#4-running-the-sample)) and look at
   everything it produces. Budget twenty minutes. You now have a working
   example of every step.
4. **Get the can photos** ([section 5](#5-getting-the-can-photos)).
5. **Adapt the script** using [FROM_DIGITS_TO_CANS.md](FROM_DIGITS_TO_CANS.md),
   and train a first can model. Get this done in week 1, however rough it is.
6. **Test it** on saved robot frames, then on the live camera
   ([section 7](#7-testing-saved-frames-first-then-live)). Write down what it
   gets wrong.
7. **Fix what it got wrong**, which nearly always means better pictures
   ([section 6](#6-adding-your-own-photos)), and train again. Repeat.
8. **Give your group `best.pt`** early, and again whenever it improves. They
   can't finish the GUI or the grab without it.
9. **Write it down as you go** in `TRAINING_NOTES.md`.

While a model trains, you are free. Start a run, then label pictures, write
notes, or help with the GUI.

---

## 4. Running the sample

From this folder, with the virtual environment active:

```powershell
python train_mnist.py
```

The first run downloads two things, so **do it while you have internet** (not
while connected to the robot's Wi-Fi): the MNIST digits (about 10 MB) and the
pretrained model `yolo11n.pt` (about 5 MB).

On a lab PC's GPU the whole thing takes **about two minutes**. It prints a
line per epoch, then a table of scores per digit, and ends with something
like:

```
best val: {'mAP50': 0.99, 'mAP50-95': 0.97, 'precision': 0.98, 'recall': 0.98}
  valid_00000.jpg: 4 (0.97), 7 (0.92), 3 (0.91), 1 (0.73)
  ...
pictures with boxes drawn: ...\runs\mnist\yolo11n_predict
trained weights: ...\runs\mnist\yolo11n\weights\best.pt
```

Then open these, in this order. Each one is something you will need to
understand for the cans.

| Open                                         | Look for                                                                                                       |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `data/mnist_yolo/train/images/` and `labels/` | A picture and the text file with the same name. Match each line of the file to a digit in the picture. **This is what a detection dataset is** |
| `data/mnist_yolo/data.yaml`                  | Four lines: where the pictures are, how many classes, what they're called                                      |
| `runs/mnist/yolo11n/train_batch0.jpg`        | What the model was actually shown: pictures cut up, glued together and recoloured. That is augmentation        |
| `runs/mnist/yolo11n/results.png`             | Losses falling and scores rising, epoch by epoch                                                               |
| `runs/mnist/yolo11n/val_batch0_pred.jpg`     | The model's boxes on pictures it never trained on. Compare with `val_batch0_labels.jpg`                        |
| `runs/mnist/yolo11n_predict/`                | The trained model used the way your GUI will use it                                                            |
| `runs/mnist/yolo11n/args.yaml`               | Every setting the run used, including the dozens you didn't set                                                |

Then read `train_mnist.py` itself, top to bottom. It is about 200 lines and
four steps: build the dataset, train, validate, predict.

Things to try (each takes a couple of minutes, and each teaches something):

```powershell
python train_mnist.py --epochs 2 --name short        # stop early: how much worse is it?
python train_mnist.py --name flipped --set fliplr=0.5  # mirror the digits: what happens to the scores?
python train_mnist.py --model yolo11s.pt --name small  # a bigger model: better? slower?
```

Each `--name` gets its own folder under `runs/mnist/`, so runs don't
overwrite each other and you can compare them.

---

## 5. Getting the can photos

The data is one ZIP, about 360 MB:

**<https://opencloud.colecorbett.ca/s/WrRfxtOfPprYiSj>**

Download it while you have internet (not on the robot's Wi-Fi), and unzip it
**into this folder**, so that you end up with:

```
last-can-eyes/
├── train_mnist.py
└── data/
    ├── README.md          <- read this: it describes every folder in the download
    ├── redbull_raw/
    ├── positives/
    ├── negatives/
    ├── redbull_v1/ ... redbull_v5/
    └── mnist_yolo/        (made by the sample; it lives here too)
```

If you get `data/data/...`, you unzipped one level too deep: move the inner
folder up.

**The one folder you need is `data/redbull_v5`.** It is a finished dataset:
photos from the web and from this robot's camera, boxes checked by hand,
already split into training, validation and test pictures. You don't have to
label or prepare anything to train your first model.
[FROM_DIGITS_TO_CANS.md](FROM_DIGITS_TO_CANS.md) says how to train on it.

The other folders are the pieces it was built from and some earlier versions.
`data/README.md` describes them, for when you want to add your own photos.

Don't put the data in your submission, and don't email it around the group:
everyone downloads their own copy.

---

## 6. Adding your own photos

The first model you train will fail at something on the robot. The cure is
almost never a setting. It is showing the model the case it failed on.

**Collect.** Frames must come **from the robot's camera**, not from a phone: a
phone photo has a different angle, height, lens and sharpness, and the model
learns those too. `docs/robot_guide.md` in the main repository ("Using the
camera") gives a one-line command that saves the current frame, with no code
at all. A "save frame" key in the GUI is faster once your cockpit builder has
one: ask for it in week 1.

Photograph what the rubric tests and what your model got wrong:

- the can far away, close, at the edge of the frame, half hidden
- lying on its side, at every angle, top toward the camera and bottom toward it
- two and three cans together
- other cans and look-alike objects, with **no** target can in view
- the empty lab: floor, chair legs, bags, bins, shoes
- while the robot is moving

Vary the place and the light. Fifty frames of one can in one spot are one
picture, fifty times.

**Label.** Every new picture needs a label file. Draw the box **tight around
the can**: its edges should touch the can's edges. Any labelling tool that
exports "YOLO format" will do; free ones include Label Studio, CVAT and the
annotation tool on Roboflow's website. Then:

- every line must start with class `0`
- a picture with no target can gets an **empty** `.txt` file
- label **every** target can in the picture. A can you leave unboxed is
  teaching the model that cans like that are background

Labelling is slow. A hundred well-chosen frames beat a thousand
near-duplicates, and cost a tenth of the time.

**Split.** New frames go into training **or** validation, never a frame in
one and its near-twin in the other. Keep a batch aside for testing that is
used for neither. See [FROM_DIGITS_TO_CANS.md, section 7](FROM_DIGITS_TO_CANS.md#7-making-sure-validation-pictures-are-new).

**Check.** After the next training run starts, open `train_batch0.jpg` in the
run folder. If your new pictures appear with boxes that fit, the labels were
read correctly. If the boxes are missing or in the wrong place, stop the run
and fix the labels: no amount of training repairs bad labels.

---

## 7. Testing: saved frames first, then live

**Saved frames (no robot needed).** Have your script run the model over a
folder of robot frames it didn't train on, with and without a can, and count
the hits and the false alarms. Look at the saved pictures. This takes seconds
and you can do it after every training run.

**Live.** Load `best.pt` in the GUI (or in a ten-line script that shows the
camera with boxes) and go through the rubric with a real can:

| Try                                                         | Watch for                                   |
| ----------------------------------------------------------- | ------------------------------------------- |
| Can upright in front of the robot, at 30 cm, 1 m, 2 m       | Is the box steady? Does it fit?             |
| Turn the can around; lay it on its side; point its top at the camera | Does the box disappear at some angle? |
| Slide the can to each edge of the picture                   | Does it vanish before it leaves the frame?  |
| Two or three target cans, apart, then touching              | One box each? Or one big box, or a miss?    |
| A different kind of can next to the target                  | Is only the target boxed?                   |
| No can: point the camera around the room                    | Does anything get a box?                    |
| Drive past the can                                          | Does the box survive the motion?            |

Adjust the confidence cut-off before you conclude the model is bad: a box
that flickers may only need a lower cut-off, and false boxes may only need a
higher one. If no cut-off gives you both, the model needs better pictures.

Save a few frames of every failure. They are your next training pictures, and
they are the "what you changed after testing on the robot" that the notes ask
for.

---

## 8. What to hand over, and what to submit

**To your group**, as early as you can, even if the model is bad:

- `best.pt`, the trained weights. It's a few megabytes: fine for Git, email
  or a USB stick.
- The confidence threshold to use with it.
- Two lines on how to load it and what a result looks like.

An early, mediocre model lets your GUI builder draw boxes while you train a
better one. Swapping in new weights later is a one-file change.

**In the group's submission:**

1. **Your training script.** It must run and train a YOLO detection model on
   a can dataset. Adapted from the sample or written from scratch, either is
   fine.
2. **`best.pt`**: the weights your GUI loads. Rename it if you like
   (`redbull.pt`); say which file it is in `HOW_TO_RUN.md`.
3. **`TRAINING_NOTES.md`**, from [the template](TRAINING_NOTES_template.md).
   The rubric asks for five things in it: the **model**, the **settings**, the
   **data split**, the **validation results**, and **what you changed after
   testing on the robot**. A page is enough. Honest and specific beats long.

Leave out the photos, the `runs/` folders and the virtual environment.

---

## 9. Use the GPU

You train on a **lab PC**. Each one has an NVIDIA GPU, and a detector that
takes hours on a CPU trains on it in minutes. That speed is what lets you try
an idea, test it on the robot, and try again within one lab session, so make
sure you're actually getting it.

**Before you train, check the install.** "Getting started" in the main
repository installs the CUDA build of PyTorch. Then:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`True` and a graphics card name means you're ready. `False` means PyTorch
was installed without CUDA support and every run will crawl; the top of
`requirements.txt` in the main repository says how to fix it. The sample
script refuses to start in that case, and tells you the same thing. That's
deliberate.

**While you train, check it's being used.**

- Ultralytics prints the device on the first line of a run. You want
  `CUDA:0` followed by the card's name, not `CPU`.
- Open a second PowerShell window and run `nvidia-smi -l 2`. It shows, every
  two seconds, how busy the GPU is (**GPU-Util**) and how much of its memory
  is in use. Ctrl+C stops it.

**Then get your money's worth.** A GPU that sits at 30% is waiting for
something. These are the settings that decide how hard it works; find out
what each one does before you change it:

| Setting | Why it matters on a GPU |
|---|---|
| `device` | Which device trains. The sample passes `0`, the first GPU. If your own script leaves it to chance, check what it chose |
| `batch` | How many images go through the GPU at once. Too small wastes the card; too large runs out of GPU memory (a `CUDA out of memory` error). Ultralytics can also pick it for you: look up `batch=-1` |
| `imgsz` | Bigger images cost GPU memory and time, and may or may not find more cans |
| `workers` | The images are loaded and augmented by the **CPU**. If GPU-Util is low while the CPU is flat out, the GPU is being starved, and this is the setting that feeds it |
| `cache` | Keeps the dataset in memory instead of re-reading it from disk every epoch |
| `amp` | Mixed precision: faster, and lighter on GPU memory. It's on by default. Leave it on |
| model size (`n`, `s`, `m`, ...) | A bigger model uses more of the GPU. Whether it finds more cans is a separate question, and one you can now afford to test |

Two things that catch people on Windows:

- **Out of memory** usually means `batch` or `imgsz` is too high, or another
  program (or a teammate's forgotten run) is holding the GPU. `nvidia-smi`
  lists what's using it.
- Training code must sit under `if __name__ == "__main__":`, as it does in
  the sample. Without it, `workers` above 0 makes Windows start your script
  over and over.

**The same goes for the GUI.** Your model also runs on the lab PC's GPU when
it draws boxes on the live feed. If detection is slow there, the first thing
to check is whether that environment has the CUDA build of PyTorch too.

**Don't train on the robot's Jetson.** It's busy running the robot, and robot
time is too scarce to spend on training.

Training doesn't need the robot at all. Start a run, hand the robot to a
teammate, and check back when it finishes.

---

## 10. A plan for three sessions

| Week | By the end of the session                                                                                                                                                  |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | GPU check passes. The sample has run and you understand its output. Data downloaded. **A first can model exists**, and your group has its `best.pt`, however bad it is. You have saved some frames from the robot |
| 2    | The model runs live in the GUI. You've been through the test list in section 7 and know its weak spots. New photos are labelled and a second model is trained              |
| 3    | Final model chosen by live testing. Confidence cut-off set. Notes written. No retraining in the last half hour: a model you haven't tested live is a gamble                |

Between sessions is when this part gets done: training and labelling need no
robot, and saved frames can be labelled anywhere.

---

## 11. When it goes wrong

| What you see                                                 | Likely cause, and what to do                                                                                                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| The sample stops with "PyTorch cannot see an NVIDIA GPU"     | The CPU-only build of PyTorch is installed. Follow the steps it prints. See [Use the GPU](#9-use-the-gpu)                                                                 |
| `CUDA out of memory`                                         | The batch is too big for the graphics card, or another training run is still going. Close other runs; try `--batch 16`                                                    |
| The script hangs or errors at "workers" on Windows           | Run it with `--workers 0`                                                                                                                                                 |
| `No such file` or "images not found" for a dataset           | The `data.yaml` has a `path:` line pointing at someone else's disk, or you passed the wrong folder. The yaml should sit in the dataset folder with no `path:` line        |
| Can't download `yolo11n.pt` or MNIST                         | You're on the robot's Wi-Fi, which has no internet. Switch networks, run it once, switch back                                                                             |
| Warning about label classes, then few or no pictures used    | Label lines start with a class number that `data.yaml` doesn't have. With `nc: 1`, every line must start with `0`                                                         |
| The boxes in `train_batch0.jpg` are in the wrong place       | The labels are wrong: not YOLO format, or pixels where fractions should be, or the label files don't match the picture names                                               |
| Great validation scores, poor on the robot                   | The validation pictures aren't like the robot's view, or are near-copies of training pictures. Add robot-camera frames; split by time                                     |
| Boxes on things that aren't cans                             | Not enough negatives. Add robot frames with no can, especially of whatever it wrongly boxed, with empty label files                                                       |
| The wrong kind of can gets a box                             | It has never seen that can without a box. Add pictures of it as negatives, some with the target can beside it (boxed) so it learns the difference                         |
| Misses the can on its side                                   | No (or few) pictures of that. Photograph it. Rotation augmentation (`degrees`) helps a little; real photos help more                                                      |
| Misses distant cans                                          | `imgsz` too small, or no distant cans in training. Use 640 and add far-away frames                                                                                        |
| Two cans get one box, or one of them is missed               | No multi-can pictures in training. Add some, including cans that touch                                                                                                    |
| Found while parked, lost while driving                       | Motion blur. Save frames while the robot moves                                                                                                                            |
| The box flickers                                             | The confidence is hovering at your cut-off. Lower the cut-off; if false boxes then appear, the model needs more training pictures of that case                            |
| The box is in the right place but red and blue look swapped, or detection is much worse in the GUI than in testing | The GUI is passing an RGB array where the model expects BGR. See [FROM_DIGITS_TO_CANS.md, section 8](FROM_DIGITS_TO_CANS.md#8-using-the-model) |
| The second model is worse than the first                     | It happens. Keep every `best.pt` with a name that says what it was trained on, so you can go back                                                                         |

Generative AI is allowed and is good at this kind of work: explaining an error
message, writing a script that converts labels, reading a `results.png`. Paste
it your actual error and your actual `data.yaml`. It can't see your pictures
or your robot, so it can't tell you what your model gets wrong. Only testing
can.
