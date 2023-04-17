# Lane_Detetction

## Usage

```python=
from img_seg import *

image = "test.jpg"
right_line, left_line, direction, highest_point = img_seg(image)

```

:::danger
right_line, left_line: both are a line, which type is [x1, y1, x2, y2]
direction: type is str
highest_point: type is [x, y]
:::
