# Where does YOLOv8n fail?

## Summary
The overall precision of the detector is 0.822, and the overall recall is 0.603. The weakest class is truck, with a precision of 0.435 and a recall of 0.333. The main failure mode is the detector's inability to detect small objects, particularly trucks and cars.

## Per-class results
| Class | Precision | Recall | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| person | 0.851 | 0.648 | 533 | 93 | 290 |
| car | 0.667 | 0.364 | 40 | 20 | 70 |
| bus | 0.786 | 0.5 | 11 | 3 | 11 |
| truck | 0.435 | 0.333 | 10 | 13 | 20 |

## Where it fails
The detector fails to detect small objects, particularly trucks and cars. The recall for small trucks is 0.25, and the recall for small cars is 0.216. The detector also confuses cars with trucks (4 times) and trucks with buses (2 times).

## Hardest images
The hardest images are:

* 000000490936.jpg: 12 missed detections (cars, persons, trucks)
* 000000119641.jpg: 6 missed detections (persons)
* 000000466416.jpg: 12 missed detections (cars)
* 000000017959.jpg: 12 missed detections (persons)
* 000000364884.jpg: 12 missed detections (persons)
* 000000272148.jpg: 9 missed detections (persons)
* 000000079969.jpg: 10 missed detections (persons)
* 000000122166.jpg: 8 missed detections (cars, persons, trucks)
* 000000207844.jpg: 10 missed detections (persons)
* 000000015517.jpg: 9 missed detections (buses)

## Recommendations
1. Increase the number of training samples for small objects, particularly trucks and cars.
2. Improve the detector's ability to detect small objects by using techniques such as data augmentation or object proposal networks.
3. Fine-tune the detector on a dataset with a large number of images of small objects.
