# Where does YOLOv8n fail?

## Summary
The overall precision of the detector is 0.822, and the overall recall is 0.603. The weakest class is the truck, with a precision of 0.435 and a recall of 0.333. The main failure mode is the detector's inability to detect small objects, particularly persons, with 227 out of 360 small persons missed.

## Where it fails
The largest failure is the detection of small persons, with 227 out of 360 small persons missed. Other size effects include the detection of small cars, with 58 out of 74 small cars missed, and small buses, with 5 out of 6 small buses missed. Class confusions include cars being detected as trucks (4 times), trucks being detected as buses (2 times), and trucks being detected as cars (2 times). There are 7 images without any detections.

## Recommendations
1.  **Improve small object detection**: The detector struggles to detect small objects, particularly persons. This could be addressed by using a more advanced object detection algorithm or by fine-tuning the current model on a dataset with a focus on small object detection.
2.  **Class confusion reduction**: The detector often confuses cars with trucks and trucks with buses. This could be addressed by increasing the number of training samples for these classes or by using a more advanced classification algorithm.
3.  **Image without detections analysis**: The 7 images without any detections should be analyzed to understand why the detector failed to detect any objects in these images. This could be due to a variety of reasons, including poor image quality, lack of objects in the image, or issues with the detector's configuration.

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
