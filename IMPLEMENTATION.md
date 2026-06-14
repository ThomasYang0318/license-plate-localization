# License Plate Character Candidate Detection

## Overview

This project implements two image-processing pipelines for license plate character candidate detection:

1. Grayscale + Otsu thresholding
2. Grayscale + Sobel edge detection

Both methods generate intermediate image outputs, draw bounding boxes on the original image, and export detected character candidate coordinates in the required text format.

## Folder Structure

```text
Datasets/             Input JPG images
GrayscaleDatasets/    Grayscale images
OtsuDatasets/         Binary images from Otsu thresholding
OtsuBoxedDatasets/    Original images with Otsu candidate boxes
OtsuResults/          Otsu coordinate output files
SobelDatasets/        Binary images from Sobel edge detection
SobelBoxedDatasets/   Original images with Sobel candidate boxes
SobelResults/         Sobel coordinate output files
```

## Programs

### Otsu Method

```bash
python3 process_otsu_boxes.py
```

Pipeline:

```text
Input image
-> grayscale
-> Otsu binary thresholding
-> adaptive threshold fallback when candidates are too few
-> connected component labeling
-> region property filtering
-> character row selection
-> horizontal group selection
-> x-coordinate sorting
-> boxed image and text output
```

### Sobel Method

```bash
python3 process_sobel_boxes.py
```

Pipeline:

```text
Input image
-> grayscale
-> Gaussian blur
-> Sobel gradient magnitude
-> Otsu thresholding on Sobel response
-> adaptive threshold fallback when candidates are too few
-> connected component labeling
-> region property filtering
-> character row selection
-> horizontal group selection
-> x-coordinate sorting
-> boxed image and text output
```

## Character Candidate Detection Framework

The most important stage is implemented in `character_candidates.py`.

The binary image is processed with connected component labeling:

```python
cv2.connectedComponentsWithStats(...)
```

For each connected component, the following region properties are used:

- `x`
- `y`
- `width`
- `height`
- `area`
- `aspect_ratio = width / height`

The candidate filtering removes noise by checking:

- minimum width
- maximum width ratio
- minimum height
- maximum height ratio
- minimum area
- maximum area ratio
- aspect ratio
- y-position consistency
- image-border margin

Objects such as screws, dash marks, border lines, and background noise are reduced by:

- excluding very small components
- excluding very wide or very tall components
- excluding components with unlikely character aspect ratios
- selecting the densest horizontal character row
- splitting candidates into horizontal groups by x distance
- keeping the highest-scoring character-like group

The final candidates are sorted from left to right by `x` coordinate before output.

## Coordinate Output Format

Each input image produces one `.txt` output file.

Example:

```text
020
6
516 241 15 28
532 241 16 28
549 242 16 28
566 243 15 27
588 243 15 27
604 244 16 27
```

Format:

```text
image_name_without_extension
detected_character_count
x1 y1 width height
x1 y1 width height
...
```

Notes:

- `x1, y1` are the top-left coordinates of the bounding box in the original input image.
- Coordinates are exported using a `(1,1)` image origin, as required by the assignment.
- `width` and `height` are the bounding box dimensions.
- Candidate boxes are sorted by `x1` from left to right.

## Handling Difficult Images

Some images, such as `020.jpg`, contain strong lighting differences and a bright vehicle body. In these cases, whole-image Otsu thresholding can fail because the global threshold includes too much background.

To handle this, both methods use an adaptive-threshold fallback when the primary method produces too few character candidates. The fallback still uses the same connected-component and region-property filtering pipeline.

## Tunable Parameters

Common candidate filtering parameters:

```bash
--min-width
--max-width-ratio
--min-height
--max-height-ratio
--min-area
--max-area-ratio
--min-aspect-ratio
--max-aspect-ratio
--y-tolerance-ratio
--border-margin-ratio
--min-candidates
--adaptive-block-size
--adaptive-c
```

Sobel-specific parameter:

```bash
--sobel-kernel-size
```

## References

1. N. Otsu, "A Threshold Selection Method from Gray-Level Histograms," IEEE Transactions on Systems, Man, and Cybernetics, 1979.
2. I. Sobel and G. Feldman, "A 3x3 Isotropic Gradient Operator for Image Processing," Stanford Artificial Intelligence Project, 1968.
3. R. C. Gonzalez and R. E. Woods, Digital Image Processing, Pearson.
4. OpenCV documentation: thresholding, Sobel derivatives, connected components, contours, and image I/O.
5. `References/Automatic_Vehicle_License_Plate_Recognition_System_Based_on_Image_Processing_and_Template_Matching_Approach.pdf`
