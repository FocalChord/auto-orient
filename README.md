# auto-orient

Detect and fix orientation of JPEG images using deep learning.

Scans a directory of JPEGs, detects which images are rotated (90 CW, 90 CCW, or 180), and applies lossless rotation to correct them. Originally built for film scans from flatbed scanners, but works on any JPEG photos.

## How it works

Uses a fine-tuned Vision Transformer (ViT) from the [Deep-OAD](https://github.com/pidahbus/deep-image-orientation-angle-detection) project. The model predicts a continuous rotation angle for each image, which is rounded to the nearest 90 degrees.

For portrait detections (90/270 degrees), a verification pass rotates the image both ways and re-runs inference to confirm the correct direction. This resolves CW vs CCW ambiguity on images without strong orientation cues.

Rotation is applied by setting the EXIF Orientation tag via `exiftool`, which is truly lossless (no pixel recompression). Alternatively, `jpegtran` can do lossless DCT rotation if you need apps that ignore EXIF orientation to display the image correctly.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [exiftool](https://exiftool.org/) for EXIF rotation (default), or `jpegtran` for pixel rotation

On macOS:

```
brew install exiftool
```

## Setup

```
git clone https://github.com/FocalChord/auto-orient.git
cd auto-orient
uv sync
```

Download the model weights (~990MB) from [Hugging Face](https://huggingface.co/focalchord/film-rotation):

```
uvx huggingface-cli download focalchord/film-rotation weights/model-vit-ang-loss.h5 --local-dir .
```

## Usage

Scan a directory of JPEGs and report detected orientations (dry run):

```
uv run auto-orient ./photos
```

Apply rotations:

```
uv run auto-orient ./photos --apply
```

### Options

```
auto-orient [OPTIONS] [PATH]

Arguments:
  PATH                              Directory to scan [default: ./photos]

Options:
  --apply / --dry-run               Apply rotation or just report [default: dry-run]
  --method [exiftool|jpegtran]      Rotation method [default: exiftool]
  --confidence FLOAT                Min confidence to auto-rotate [default: 0.6]
  --skip-contact-sheets / --include-contact-sheets
                                    Skip contact_sheet*.jpg files [default: skip]
  --output [table|json|csv]         Output format [default: table]
  --output-file PATH                Write results to file
  --batch-size INT                  Inference batch size [default: 32]
```

### Examples

Dry run on a single directory:

```
uv run auto-orient ./photos/vacation
```

Apply with lower confidence threshold (catches more borderline cases):

```
uv run auto-orient ./photos --apply --confidence 0.4
```

Export results as JSON:

```
uv run auto-orient ./photos --output json --output-file results.json
```

## Output

The default table output shows each image with its detected orientation, confidence score, and recommended action:

- **OK**: Image is already upright
- **Rotate 90 CW/CCW**: Portrait image needing rotation
- **Rotate 180**: Upside-down image
- **Review**: Detected as needing rotation but below confidence threshold

## Project structure

```
auto-orient/
  src/auto_orient/
    cli.py              Click CLI
    scanner.py          Recursive JPEG discovery
    rotation.py         exiftool/jpegtran lossless rotation
    report.py           Table, JSON, CSV output
    detectors/
      __init__.py       Orientation enum, DetectionResult dataclass
      oad.py            Deep-OAD ViT model wrapper
      pipeline.py       Detection orchestrator
  weights/                          (downloaded from Hugging Face)
    model-vit-ang-loss.h5   Pre-trained ViT weights (~990MB)
  pyproject.toml
```

## Model

The orientation detection model is a ViT-Base (google/vit-base-patch16-224) with a single Dense(1) regression head, trained with angular loss on COCO. It predicts the rotation angle applied to an image as a continuous value.

Pre-trained weights are from [pidahbus/deep-image-orientation-angle-detection](https://github.com/pidahbus/deep-image-orientation-angle-detection).

## License

Model weights are subject to the [Deep-OAD license](https://github.com/pidahbus/deep-image-orientation-angle-detection).
