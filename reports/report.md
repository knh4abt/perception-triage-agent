# Where does YOLOv8n fail?

## Summary

The object detector YOLOv8n has a precision of 0.822 and a recall of 0.603. The weakest class is truck, with a precision of 0.435 and a recall of 0.333. The main failure mode is the inability to detect small objects, particularly small person and car objects.

## Where it fails: object size

The largest failure of YOLOv8n on COCO street images is that 227 out of 360 small person objects were missed. The recall for small person objects is 0.369, which is significantly lower than the recall for medium (0.801) and large (0.966) person objects. Similarly, the recall for small car objects is 0.216, which is lower than the recall for medium (0.655) and large (0.714) car objects.

## Where it fails: class confusions and false alarms

A real car was detected as truck 4 times, a real truck was detected as bus 2 times, and a real truck was detected as car 2 times. The class with the lowest precision is person, with 290 missed completely.

## What the hard images have in common

The hardest images are characterized by their high brightness, with a mean brightness of 128.0, which is significantly higher than the mean brightness of all images, which is 112.9. The number of objects in the hardest images is also higher, with a mean of 13.5, compared to 4.9 for all images. The median object area in the hardest images is 242, which is lower than the median object area of 3728 for all images. The share of small objects in the hardest images is also higher, with a mean of 0.836, compared to 0.276 for all images.

The first file name of the hardest images is "000000490936.jpg". The statistics for this image are: num_gt = 16, tp = 4, fp = 3, fn = 12, missed = ["car", "person", "truck"], false_alarms = ["person", "truck"].

## Recommendations

1. Analyze the characteristics of the hardest images, including their brightness, object count, and object size, to identify potential causes of the model's failure.
2. Improve the model's ability to distinguish between classes, particularly for small objects, as they have the lowest recall.
3. Investigate the effect of class confusions and false alarms on detection performance, particularly for the class "truck", which has the lowest precision.

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

- confusion: The class with the lowest precision is truck (0.435), not what the section says.
- confusion: "A real car was detected as truck 4 times, a real truck was detected as bus 2 times, and a real truck was detected as car 2 times." contradicts: a real car was detected as truck: 4 times
- confusion: "The class with the lowest precision is person, with 290 missed completely." contradicts: lowest_precision_class: truck
- editor: "Improve the model's ability to distinguish between classes, particularly for small objects, as they have the lowest recall." contradicts: The recall of small objects is not the lowest, e.g. person has a recall of 0.648
- editor: "Investigate the effect of class confusions and false alarms on detection performance, particularly for the class "truck", which has the lowest precision." contradicts: The lowest precision class is truck, but it is not the only class with low precision, e.g. car has a precision of 0.667
- editor: "The main failure mode is the inability to detect small objects, particularly small person and car objects." contradicts: 227 of 360 small person objects were missed
- editor: "The weakest class is truck, with a precision of 0.435 and a recall of 0.333." contradicts: The lowest precision class is truck
