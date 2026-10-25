# Data layout

For the baseline training workflow, `scripts/download_flickr8k.py` uses the
Kaggle dataset `adityajn105/flickr8k`. Install/authenticate the Kaggle CLI
(`pip install kaggle`, then place your Kaggle API token in the documented
Kaggle config location) before running:

```powershell
python scripts/download_flickr8k.py --output data/flickr8k
python scripts/train_baseline.py --data data/flickr8k --limit 500 --epochs 2
```

The training script uses up to 500 unique images and their available captions.
The dissertation evaluation set is supplied separately by the researcher:

```text
data/eval/
  annotations.json  # {"image.jpg": ["reference caption", ...]}
  images/
    image.jpg
```

Do not use evaluation images in the baseline training subset. The repository
does not bundle Flickr8k or held-out evaluation images.
