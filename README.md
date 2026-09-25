# ABD3D

ABD3D is a modular single-image 3D generation baseline:

`ViT Encoder -> VAE Latent Space -> DiT Generator -> Triplane Decoder`

## Setup

```powershell
.\abd3d-env\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Training

```powershell
python main.py --batch-size 4 --steps 100000
python main.py --resume checkpoints/latest.pt
```

The default configuration is designed for an RTX 5060 with 8 GB VRAM: batch size 4, gradient accumulation, mixed precision, and a compact model. Checkpoints are written to `checkpoints/latest.pt` every 30 minutes and to `checkpoints/final.pt` at the end of training.

Set `Config.dataset_name` in `config.py` to the exact Arb-Objaverse dataset identifier available to your Hugging Face account. The streaming loader accepts records containing an image under `image`, `render`, or `front_image`.
