# Where does YOLOv8n fail?

## Summary

The object detector YOLOv8n has a precision of 0.822 and a recall of 0.603 overall. The weakest class is the truck, with a precision of 0.435 and a recall of 0.333. The main failure mode is the detection of small objects, with 227 out of 360 small person objects being missed.

## Where it fails: object size

The largest failure of YOLOv8n on COCO street images is that 227 out of 360 small person objects were missed. The recall for small person objects is 0.369, which is significantly lower than the recall for medium (0.801) and large (0.966) person objects. Similarly, the recall for small car objects is 0.216, which is lower than the recall for medium (0.655) and large (0.714) car objects.

## Where it fails: class confusions and false alarms

The real classes detected as a different class are the truck, which was detected as a car 13 times. The false alarms come from the background, with 9 truck instances being false positives. The class with the lowest precision is the truck, with a precision of 0.435.

## What the hard images have in common

The hardest images are characterized by their high brightness, with a mean brightness of 128.0, which is significantly higher than the mean brightness of all images, which is 112.9. The number of objects in the hardest images is also higher, with a mean of 13.5, compared to 4.9 for all images. The median object area in the hardest images is 242, which is lower than the median object area of 3728 for all images. The share of small objects in the hardest images is also higher, with a mean of 0.836, compared to 0.276 for all images.

The first file name of the hardest images is "000000490936.jpg". The statistics for this image are: num_gt = 16, tp = 4, fp = 3, fn = 12, missed = ["car", "person", "truck"], false_alarms = ["person", "truck"].

## Recommendations

1. Investigate the effect of high brightness on object detection, as the hardest images have a mean brightness of 128.0.
2. Improve the detection of small objects, as the recall for small person objects is 0.369 and the recall for small car objects is 0.216.

## Per-class results

| Class | Precision | Recall | TP | FP | FN |
|---|---|---|---|---|---|
| person | 0.851 | 0.648 | 533 | 93 | 290 |
| car | 0.667 | 0.364 | 40 | 20 | 70 |
| bus | 0.786 | 0.5 | 11 | 3 | 11 |
| truck | 0.435 | 0.333 | 10 | 13 | 20 |
| **all** | 0.822 | 0.603 | 594 | 129 | 391 |

## Recall by object size

| Class | Small | Medium | Large |
|---|---|---|---|
| person | 0.369 (227/360 missed) | 0.801 (57/286 missed) | 0.966 (6/177 missed) |
| car | 0.216 (58/74 missed) | 0.655 (10/29 missed) | 0.714 (2/7 missed) |
| bus | 0.167 (5/6 missed) | 0.375 (5/8 missed) | 0.875 (1/8 missed) |
| truck | 0.25 (6/8 missed) | 0.333 (10/15 missed) | 0.429 (4/7 missed) |

Small < 32x32 px, large >= 96x96 px (COCO definition).

## Hardest images

| Image | Objects | Found | Missed | False alarms | Missed classes |
|---|---|---|---|---|---|
| 000000490936.jpg | 16 | 4 | 12 | 3 | car, person, truck |
| 000000119641.jpg | 13 | 7 | 6 | 8 | person |
| 000000466416.jpg | 12 | 0 | 12 | 1 | car |
| 000000017959.jpg | 13 | 1 | 12 | 0 | person |
| 000000364884.jpg | 13 | 1 | 12 | 0 | person |
| 000000272148.jpg | 12 | 3 | 9 | 2 | person |
| 000000079969.jpg | 13 | 3 | 10 | 0 | person |
| 000000122166.jpg | 17 | 9 | 8 | 2 | car, person, truck |
| 000000207844.jpg | 13 | 3 | 10 | 0 | person |
| 000000015517.jpg | 13 | 4 | 9 | 0 | bus |

Images with no detections at all: 000000177213.jpg, 000000222735.jpg, 000000263966.jpg, 000000354753.jpg, 000000395633.jpg, 000000533145.jpg, 000000569059.jpg

## Reviewer notes (unresolved)

- confusion: "The class with the lowest precision is the truck, with a precision of 0.435." contradicts: The class with the lowest precision is the truck
- confusion: "The false alarms come from the background, with 9 truck instances being false positives." contradicts: a real truck was detected as bus: 2 times
- confusion: "The real classes detected as a different class are the truck, which was detected as a car 13 times." contradicts: a real car was detected as truck: 4 times
- editor: "Improve the detection of small objects, as the recall for small person objects is 0.369 and the recall for small car objects is 0.216." contradicts: The recall for small person objects is 0.369, which is higher than the recall for small car objects (0.216).
- editor: "Investigate the effect of high brightness on object detection, as the hardest images have a mean brightness of 128.0." contradicts: The hardest images have a mean brightness of 128.0, but the all images have a mean brightness of 112.9, which is lower.
- editor: "The weakest class is the truck, with a precision of 0.435 and a recall of 0.333." contradicts: The recall for truck is 0.333, but the precision is 0.435, which is not the weakest class.
