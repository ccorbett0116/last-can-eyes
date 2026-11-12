# Primer: what you are about to train

For people who have never trained a neural network. Ten minutes of reading,
no maths. If you already know what an epoch and a validation set are, skim
the headings and go to [FROM_DIGITS_TO_CANS.md](FROM_DIGITS_TO_CANS.md).

1. [A picture is a grid of numbers](#1-a-picture-is-a-grid-of-numbers)
2. [A neural network is a function with knobs](#2-a-neural-network-is-a-function-with-knobs)
3. [What makes it a CNN](#3-what-makes-it-a-cnn)
4. [Classifying is not detecting](#4-classifying-is-not-detecting)
5. [YOLO, and why you don't start from zero](#5-yolo-and-why-you-dont-start-from-zero)
6. [The data is the program](#6-the-data-is-the-program)
7. [Training, validation and test](#7-training-validation-and-test)
8. [The words on the screen while it trains](#8-the-words-on-the-screen-while-it-trains)
9. [How good is it? Reading the scores](#9-how-good-is-it-reading-the-scores)
10. [Confidence](#10-confidence)
11. [Why the GPU matters](#11-why-the-gpu-matters)

---

## 1. A picture is a grid of numbers

The robot's camera gives a picture 640 pixels wide and 360 high. Each pixel is
three numbers from 0 to 255: how much red, green and blue. So one frame is
about 690,000 numbers. That is all the computer gets. There is no "can" in
there, only numbers, and the job is to turn those numbers into "there is a can
here, in this rectangle".

Nobody knows how to write that as ordinary code. `if pixel is blue and the one
next to it is silver...` falls apart the moment the light changes or the can
rolls over. So we don't write the rule. We show a program thousands of
examples and let it work out a rule for itself.

## 2. A neural network is a function with knobs

A neural network is a function: numbers in (the picture), numbers out (boxes).
In between are millions of adjustable numbers called **weights**. Think of
them as knobs. With the knobs set at random, the output is rubbish.

**Training** is the process of setting the knobs:

1. Show the network a picture it has the right answer for.
2. Compare its answer with the right one. The size of the mistake is a single
   number called the **loss**.
3. Turn every knob a tiny amount in the direction that would have made the
   loss smaller.
4. Repeat, thousands of times, over all the pictures.

That's it. Nothing in the network is told what a can looks like. The knobs
end up in positions that happen to give right answers on the examples, and, if
the examples were good, on new pictures too.

## 3. What makes it a CNN

CNN stands for **convolutional neural network**. It is a neural network built
for pictures, around one idea: look at the picture through small windows.

A **filter** is a tiny grid of weights, perhaps 3 x 3 pixels. It is slid
across the whole picture, and at each position it answers one question, such
as "is there a vertical edge here?". The same filter is used everywhere, so a
can in the corner is found by the same weights as a can in the middle.

The network stacks these in **layers**:

- The first layers find simple things: edges, corners, patches of colour.
- The middle layers combine those into parts: curves, stripes, text.
- The last layers combine parts into objects: "the top of a can", "a can".

You don't design any of this. Which edges and which parts matter is what
training works out.

## 4. Classifying is not detecting

Two different jobs, and the difference matters for this lab.

| Job                | Question                              | Answer                                      |
| ------------------ | ------------------------------------- | ------------------------------------------- |
| **Classification** | What is this picture of?              | One label: "a 7"                            |
| **Detection**      | What is in this picture, and _where_? | A list of boxes, each with a label: "a 7 here, a 3 there" |

The robot needs detection. It has to know where the can is in order to drive
to it, there may be several cans, and often there is none.

MNIST, the famous handwritten-digit dataset, is a classification dataset:
tiny pictures with one digit each. The sample script turns it into a detection
problem by pasting digits onto a larger blank picture and recording where each
one went. That step exists only because of MNIST. Your can photos are already
pictures with objects somewhere in them.

## 5. YOLO, and why you don't start from zero

**YOLO** ("you only look once") is a family of detection networks that are
fast enough to run on live video. **Ultralytics** is the Python library that
packages them. You will use **YOLO11n**: version 11, size "nano", the smallest
and fastest. Larger sizes exist (`s`, `m`, ...). They can be a little more
accurate, and they are slower to train and to run.

The file `yolo11n.pt` is not an empty network. It has already been trained, by
someone else, on hundreds of thousands of photos of everyday objects. Its
early layers already know edges, textures and shapes. You take that network
and continue training it on your pictures. This is called **fine-tuning** or
**transfer learning**, and it is why a few hundred photos and a few minutes
are enough, where starting from random weights would need vastly more of both.

`.pt` files are **weights**: the knob settings, saved. `yolo11n.pt` is the
starting point. `best.pt`, which training produces, is your model.

## 6. The data is the program

With ordinary code, behaviour comes from what you write. Here it comes from
what you show. The model learns exactly what is in the training pictures,
including things you didn't intend.

- If every training photo shows the can upright, it won't find one on its side.
- If every photo was taken close up on a white table, it won't find a small
  can far away on a grey floor.
- If no photo ever shows a similar-looking object _without_ a box, it has no
  reason not to box that object.

So most of the work in this part, and most of the marks, come from choosing
what the model sees, not from the training code.

**Labels.** For detection, every picture comes with a small text file that
lists its boxes, one per line:

```
class  x_centre  y_centre  width  height
```

`class` is a number (0 for the first kind of object, 1 for the second...). The
other four are fractions of the picture's width and height, so they are always
between 0 and 1. A can in the middle of the picture, a tenth as wide as the
picture and a quarter as tall:

```
0 0.5 0.5 0.1 0.25
```

A picture with no object in it has an **empty** label file. Those pictures are
called **negatives**, and they are how the model learns what to leave alone.

Someone has to draw those boxes by hand. The model can't be better than its
labels: a sloppy box teaches a sloppy box.

## 7. Training, validation and test

If you measure the model on the same pictures it learned from, it will look
perfect: it can simply have memorised them. That is called **overfitting**,
and it tells you nothing about a new picture.

So the pictures are divided:

| Set                    | Used for                                                                             |
| ---------------------- | ------------------------------------------------------------------------------------ |
| **Training**           | Turning the knobs. The model sees these again and again                              |
| **Validation** (valid) | Checking progress on pictures it has not learned from. The scores you see come from here |
| **Test** (optional)    | One final, honest check at the very end, on pictures used for nothing else           |

The validation score only means something if the validation pictures are
**really new**. This is the easiest mistake to make with video: two frames
taken a tenth of a second apart are nearly the same picture. Put one in
training and the other in validation, and the model is being tested on
something it has, in effect, already seen. The score looks wonderful and the
robot finds nothing.

## 8. The words on the screen while it trains

| Word                       | Meaning                                                                                                                                                                  |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Epoch**                  | One pass through every training picture. Training runs for many epochs                                                                                                   |
| **Batch**                  | The pictures are processed a group at a time, and the knobs are turned once per group. `batch=32` means 32 pictures per group                                            |
| **Image size** (`imgsz`)   | Pictures are resized to this before going in. Bigger keeps small objects visible and is slower                                                                           |
| **Loss** (`box_loss`, `cls_loss`, `dfl_loss`) | How wrong the model currently is: about where the boxes are, and about which class they are. They should go down as training goes on                 |
| **Augmentation**           | Each time a picture is shown it is altered at random: shifted, resized, brightened, mirrored, rotated. One photo does the work of many, and the model learns to cope with variety |
| **Mosaic**                 | One of those alterations: four training pictures glued into one. It is why the example batches look like collages                                                        |
| **Patience**               | If the validation score hasn't improved for this many epochs, training stops early                                                                                       |
| **`best.pt` / `last.pt`**  | The weights from the epoch with the best validation score, and from the final epoch. You want `best.pt`                                                                  |

## 9. How good is it? Reading the scores

A predicted box counts as **correct** if it overlaps a real box closely enough
and has the right class. "Closely enough" is measured by **IoU**
(intersection over union): the area the two boxes share, divided by the area
they cover together. 1.0 is a perfect fit; 0.5 is a loose one.

| Score         | Question it answers                                                              | Bad when                         |
| ------------- | -------------------------------------------------------------------------------- | -------------------------------- |
| **Precision** | Of the boxes it drew, how many were right?                                       | It boxes things that aren't cans |
| **Recall**    | Of the cans that were there, how many did it find?                               | It misses cans                   |
| **mAP50**     | An overall mark from 0 to 1, counting a box as right at IoU 0.5 or more          | Either of the above              |
| **mAP50-95**  | The same, averaged over stricter and stricter overlaps. Rewards boxes that fit tightly | The boxes are loose         |

Higher is better for all four.

**The trap:** these scores are measured on _your validation pictures_. They
say how the model does on pictures like those. If those pictures are easy, or
are near-copies of training pictures, or look nothing like what the robot's
camera sees, a score of 0.99 means very little. The only test that counts in
this lab is the robot's live camera.

Each run also saves pictures, which are worth more than the numbers:

| File                    | What it shows                                                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------------- |
| `train_batch0.jpg`      | Training pictures as the model sees them, after augmentation, with their labels. **Look at this first.** If the boxes are in the wrong place here, the labels are wrong and nothing else matters |
| `val_batch0_labels.jpg` | Validation pictures with the correct boxes                                                      |
| `val_batch0_pred.jpg`   | The same pictures with the model's boxes. Compare the two                                       |
| `results.png`           | Losses and scores, epoch by epoch. Losses should fall and scores should rise, then level off    |
| `confusion_matrix.png`  | Which classes get mistaken for which. "background" means "nothing there"                        |

## 10. Confidence

Every box the model draws comes with a **confidence** from 0 to 1: how sure it
is. You choose a cut-off and ignore boxes below it.

- Cut-off too **low**: boxes on things that aren't cans.
- Cut-off too **high**: real cans missed, and boxes that flicker on and off as
  the confidence wobbles around the cut-off.

There is no correct value. It is a trade between the two kinds of mistake, and
you set it by watching the live picture. It does not need retraining.

## 11. Why the GPU matters

Training is the same simple arithmetic done billions of times. A graphics card
(GPU) does it tens of times faster than the processor (CPU). On the lab PCs
the sample trains in about two minutes on the GPU. On the CPU it still works,
and you will wait long enough to regret it.

Before any training, check that Python can see the GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`True` and the name of the card is what you want. `False` means the CPU-only
build of PyTorch is installed: fix it with the instructions at the top of
`requirements.txt` in the main repository.
