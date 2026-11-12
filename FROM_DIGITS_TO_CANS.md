# From digits to cans

`train_mnist.py` trains a detector for handwritten digits. This page goes
through what has to change for it to train a detector for the can, and why.
It doesn't give you the finished script: making these changes is the
assignment. None of them is long.

Copy `train_mnist.py` to a new file (say `train_cans.py`) and work on the
copy, so you keep a sample that runs.

**The short version**

| # | In the digits script                                   | For the cans                                                          |
| - | ------------------------------------------------------ | --------------------------------------------------------------------- |
| 1 | Builds its own dataset out of MNIST                    | Uses a folder of real photos with hand-drawn boxes. Nothing to generate |
| 2 | Ten classes, `'0'` to `'9'`                            | One class: the can                                                    |
| 3 | Pictures are 320 x 320, so `imgsz=320`                 | Camera frames are 640 x 360 and cans can be small: a larger `imgsz`   |
| 4 | 10 epochs are enough                                   | A harder problem and real photos: more epochs                         |
| 5 | Mirroring is switched off (`fliplr=0`)                 | A mirrored can is still a can: switch it back on, and choose the other augmentations |
| 6 | Shows its results on 12 validation pictures            | Test on frames from the robot's camera, with and without a can        |
| 7 | Validation pictures are new by construction            | You have to make sure they are                                        |
| 8 | Nothing uses the model afterwards                      | Your GUI loads `best.pt` and runs it on every frame                   |

The model (`yolo11n.pt`), the training call, the validation call and
everything Ultralytics does inside them **stay exactly the same**. That is the
point of the sample: the workflow doesn't care what the objects are. What
changes is the data and a handful of settings.

---

## 1. The dataset: use photos, don't generate

**What to change.** `build_dataset()` and the lines that call it go. In their
place, the script needs the path of a dataset folder that already exists, and
`data_yaml` must point at the `data.yaml` inside it. Making that path a
command-line argument (as `--model` and `--epochs` already are) lets you train
on different datasets without editing the script.

**What a dataset folder is.** Exactly what `build_dataset()` wrote for the
digits. Run the sample once and look inside `data/mnist_yolo/`: that layout is
the layout of every YOLO dataset.

```
some_dataset/
├── data.yaml            where the pictures are, how many classes, their names
├── train/
│   ├── images/          the pictures
│   └── labels/          one .txt per picture, same name, one line per box
└── valid/
    ├── images/
    └── labels/
```

**Where the can data comes from.** Download the ZIP from
<https://opencloud.colecorbett.ca/s/WrRfxtOfPprYiSj> and unzip it into this
folder so that you have `last-can-eyes/data/README.md`. (The
[README](README.md#5-getting-the-can-photos) shows the layout.)

**Use `data/redbull_v5`.** It is a finished dataset in exactly the layout
above: about 1,700 training pictures, 218 for validation and 127 kept aside
for testing, every box checked by hand, one class. Point your script at that
folder and train. Nothing needs labelling, splitting or converting first.

Do that in week 1: a bad first model on day one is worth more than a perfect
plan.

The download also holds the pieces `redbull_v5` was built from (web photos,
robot-camera frames with and without a can) and four earlier, smaller
versions. `data/README.md` describes them. You only need them if you want to
build a dataset of your own.

It is unlikely to be your last model. The live test covers things the
download can't fully cover for you (see the rubric: angles, distance, several
cans, the wrong kind of can), and the fix for almost every failure you will
see on the robot is **more of the right pictures**. "Adding your own photos"
in the [README](README.md#6-adding-your-own-photos) covers how.

**If you assemble a dataset yourself** (optional), three things in the
download will trip you up if you don't know about them:

- `redbull_raw/` has **four classes** (they describe the can's condition:
  good, dented and so on). You want one. Every label line has to start with
  `0`. Its `data.yaml` also names folders that aren't there, so you can't
  train on `redbull_raw/` as it is.
- The frames in `negatives/` have **no label files**. In a dataset, a picture
  with no can gets a label file that exists and is empty. Ultralytics also
  treats a picture with no label file at all as background, but then you
  can't tell "no can here" from "nobody has labelled this one yet". An empty
  file says you meant it.
- `positives/session1/` and `session2/` are the raw captures, and
  `_similar/` holds near-duplicate frames that were set aside. The labelled
  frames are in `session1_labeled/` and `session2_labeled/` (`images/` and
  `labels/`). Ignore `prelabels/`: those are unchecked machine guesses.

---

## 2. One class

**What to change.** Nothing in the Python. The number of classes and their
names are read from `data.yaml`:

```yaml
train: train/images
val: valid/images
nc: 1
names: ['redbull']
```

The assembled datasets already have this. If you make your own `data.yaml`,
this is all it needs. Leave out any `path:` line and Ultralytics looks for
`train/` and `valid/` next to the file, which keeps the dataset working when
you copy it to another PC.

**Why it matters.** `nc` and `names` must agree with the label files. With
`nc: 1`, every label line must start with `0`. A line that starts with `2`
refers to a class that doesn't exist: Ultralytics throws that picture out
with a warning about label classes, and if that is every picture, there is
nothing left to train on.

**Why only one class**, when the rubric says the model must ignore other
kinds of can? Because "ignore" is not a class. The model learns to ignore
something by seeing it in training pictures **with no box on it**. Pictures of
the wrong can, with an empty label file, are negatives like any other.

---

## 3. Image size

**What to change.** The default of `--imgsz`.

**Why.** Every picture is resized so its longer side is `imgsz` before it goes
into the network. The digit pictures are 320 pixels wide, so 320 loses
nothing. A camera frame is 640 wide. At `imgsz=320` it would be halved, and a
can two metres away, which is only about 10 pixels wide in the original,
becomes 5 pixels: too little to recognise. At 640 the frame goes in at full
size.

Bigger than 640 gains nothing, since the camera doesn't deliver more pixels.
Use the same `imgsz` when you run the model as when you trained it.

---

## 4. Epochs

**What to change.** The default of `--epochs`.

**Why.** White digits on a black background are an easy problem, and ten
passes are enough. Real photos, with shadows, reflections, clutter and cans
at every size, need longer: think **60 to 100 epochs**, not 10. On the lab
GPU that is still minutes.

You don't have to guess the exact number. `patience=20` is already in the
script: training stops by itself if the validation score hasn't improved for
20 epochs, and `best.pt` is always the best epoch, not the last one. Look at
`results.png` afterwards: if the scores were still climbing at the end, train
longer; if they went flat halfway, the extra epochs changed nothing.

---

## 5. Augmentation

**What to change.** Remove the line that sets `fliplr` to 0. Then decide
which other augmentations to turn up, and pass them with `--set`, for example
`--set degrees=10 scale=0.7`.

**Why `fliplr` was off.** Ultralytics mirrors half of the training pictures
left to right. A mirrored 3 is not a 3, so for digits that would teach
nonsense. A mirrored can is a perfectly good can, and you get twice the
variety for nothing.

**The others.** Augmentation is how you tell the model which differences
don't matter. Each setting should be chosen by asking: _does the robot really
see this?_

| Setting           | What it does to each training picture           | Ask yourself                                                              |
| ----------------- | ----------------------------------------------- | ------------------------------------------------------------------------- |
| `degrees`         | Rotates it by up to this many degrees           | Will the can always be upright? (The drop leaves it on its side.)         |
| `scale`           | Zooms in and out by up to this fraction         | How different are "can at 30 cm" and "can at 2 m"? Are both in your photos? |
| `translate`       | Shifts it sideways and up or down               | Will the can always be in the middle of the frame?                        |
| `hsv_s`, `hsv_v`  | Changes colour strength and brightness          | Does the lab's lighting change through the day?                           |
| `fliplr`          | Mirrors left to right (default 0.5)             | See above                                                                 |
| `flipud`          | Mirrors top to bottom (default 0)               | Does the robot ever see the world upside down?                            |
| `mosaic`          | Glues four pictures into one (default on)       | Leave it on. It is why small objects get learned at all                   |

The full list, with defaults, is in the Ultralytics documentation under
"train settings". Every run also writes the settings it actually used to
`args.yaml` in its folder.

Two cautions:

- **More is not better.** Rotating by 180 degrees teaches the model about
  cans on the ceiling. Each augmentation spends some of the model's capacity
  on cases; spend it on cases that happen.
- **Augmentation is not a substitute for photos.** Rotating a picture of an
  upright can gives a tilted picture of an upright can, with tilted shadows
  and a tilted floor. It is not the same as a photo of a can lying down. If a
  case matters, photograph it.

**Change one thing per run**, and give each run its own `--name`. If you
change three settings and the model gets better, you don't know which one did
it. That comparison is what `TRAINING_NOTES.md` asks you to write down.

---

## 6. Testing on what the robot sees

**What to change.** Step 4 of the script runs the model on twelve validation
pictures and saves them with boxes. Point it, as well or instead, at folders
of **frames from the robot's camera**:

- a folder of frames that **all** contain the can. Count how many get a box.
- a folder of frames with **no** target can. Count how many get a box anyway.

No labels are needed for either count: `len(r.boxes)` is the number of boxes
the model drew on a picture. Keep `save=True` and look through the saved
pictures. Ten minutes of looking at what it missed tells you what to
photograph next.

**Why.** The validation score answers "how does it do on pictures like my
validation pictures?". The marks are for "how does it do on the robot?". If
most of your validation pictures are close-ups from the web, the first number
can be 0.99 while the second is poor, and you won't find out until the demo.

For the test to be honest, the frames you count on must **not** be ones the
model trained on. Keep some robot frames aside for this and never train on
them. `redbull_v3` to `v5` have a `test/` folder made that way.

---

## 7. Making sure validation pictures are new

**What to change.** Nothing, if you train on an assembled dataset: its split
has been done with care. Everything, if you add your own photos.

**Why.** The digits were safe by construction: validation digits were written
by different people from training digits. With frames saved from a camera
there is no such guarantee. Frames saved a moment apart are nearly identical.
Shuffle them and deal them out at random, and near-twins land on both sides:
the model is tested on pictures it has effectively memorised.

When you add your own frames:

- **Split by time, not by frame.** Keep whole stretches together: for
  example, everything from one drive past the can goes to training, and a
  different drive goes to validation.
- **Throw away near-duplicates.** Twenty copies of the same view teach the
  model nothing more than one, and they are twenty boxes to draw.
- **Never put a rotated or edited copy of a training picture into
  validation.**

A validation score that jumps to 0.99 the moment you add your own frames is
the usual sign that this went wrong.

---

## 8. Using the model

Training produces `runs/<...>/weights/best.pt`. That file is your model: it is
what the GUI loads, and what you submit. Step 4 of the sample already shows
every call you need:

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")          # once, when the program starts

results = model.predict(frame, conf=0.25, imgsz=640, verbose=False)
r = results[0]                            # one picture in, one result out
for (x1, y1, x2, y2), confidence in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
    ...                                   # corners of one box, in pixels of `frame`
```

Things that catch people out:

- **Load the model once**, not once per frame. Loading takes seconds.
- **The first prediction is slow** (the GPU is warming up). Run one on a blank
  picture at start-up so the first real frame isn't late.
- **Colour order.** Given a file name, Ultralytics reads the picture
  correctly. Given a NumPy array, it assumes the channels are in **BGR** order
  (OpenCV's habit). An image opened with PIL is RGB. Feed it RGB as if it were
  BGR and red and blue are swapped, which matters a great deal for a blue can.
  Reverse the last axis first: `frame[:, :, ::-1]`.
- **Box coordinates are in pixels of the picture you passed in.** If the GUI
  shows the picture at a different size, scale the boxes by the same amount.
- **`conf` is your cut-off** ([PRIMER.md, section 10](PRIMER.md#10-confidence)).
  Make it adjustable in the GUI and tune it on the live picture.
- **Don't run the detector in the GUI's main thread.** A window that waits for
  each prediction freezes. Agree with your cockpit builder how frames go in
  and boxes come out: see `docs/putting_it_together.md` in the main
  repository.

---

## Checklist

- [ ] `train_mnist.py` runs to the end and you have looked at its output pictures.
- [ ] Your copy takes a dataset folder instead of building one.
- [ ] `imgsz`, `epochs` and the augmentation settings are chosen for cans, and
      you can say why for each.
- [ ] `train_batch0.jpg` from a can run shows cans with boxes that fit them.
- [ ] The script reports how the model does on robot-camera frames it never
      trained on, with and without a can.
- [ ] `best.pt` loads in a three-line test and draws a box on a saved robot frame.
- [ ] Each run has a name, and its settings and results are in `TRAINING_NOTES.md`.
