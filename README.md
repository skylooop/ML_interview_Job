## Task 1. Multiplicate
### Requirements
```
$ pip install absl-py
```
### Running script:
```
python Task1Algorithm/multiplicate.py --input_array=1,2,3,4
```
After ``` --input_array``` flag you need to enter values, separated with commas **without any spaces**.

### Description of algorithm
The idea is to solve task in ```O(n)``` space and ```O(1)``` time complexity.
This can be done using running prefix (cumprod until ```i``` element) and suffix (product from ```i = n - 1 to i + 1```) products. That is, multiply left cumprod part (except current element) and right cumprod part (except current element).