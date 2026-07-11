"""离线预计算 DINOv3-SAT493M 教师特征。

对每个 patch 的每个可用高分光学日期做教师推理，输出按日期平均后的
dense feature map（32x32x1024, fp16），作为 V4 蒸馏目标。

用法：
  ASCEND_RT_VISIBLE_DEVICES=6 python scripts/v4/precompute_dinov3_teacher_features.py \
    --regions haidian harbin --device npu:0 --num-shards 2 --shard-id 0
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TEACHER_MEAN = (0.430, 0.411, 0.296)
TEACHER_STD = (0.213, 0.156, 0.143)
INPUT_SIZE = 512  # 427 -> 512, patch16 => 32x32 tokens
GRID = INPUT_SIZE // 16
NUM_PREFIX_TOKENS = 5  # 1 cls + 4 register


def load_teacher(device: str):
    import timm

    model = timm.create_model(
        "vit_large_patch16_dinov3.sat493m",
        pretrained=True,
        pretrained_cfg_overlay=dict(
            file="/data/xuannv_embedding/pretrained/dinov3_vitl16_sat493m_timm/model.safetensors"
        ),
        num_classes=0,
    )
    return model.eval().to(device)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """高分光学 DN 值 -> [0,1] -> 教师归一化。按波段 2%/98% 分位拉伸。"""
    out = np.empty_like(img, dtype=np.float32)
    for b in range(img.shape[0]):
        lo, hi = np.nanpercentile(img[b], [2, 98])
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        out[b] = np.clip((img[b] - lo) / (hi - lo), 0.0, 1.0)
        out[b] = (out[b] - TEACHER_MEAN[b]) / TEACHER_STD[b]
    return out


@torch.no_grad()
def extract(model, img: np.ndarray, device: str) -> torch.Tensor:
    x = torch.from_numpy(img).unsqueeze(0).to(device)
    x = torch.nn.functional.interpolate(
        x, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False
    )
    feats = model.forward_features(x)  # [1, 5+GRID*GRID, 1024]
    patch_tokens = feats[:, NUM_PREFIX_TOKENS:, :]
    return patch_tokens.reshape(GRID, GRID, -1).cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/data/xuannv_embedding/processed"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/xuannv_embedding/teacher_features/dinov3_vitl16_sat493m"))
    parser.add_argument("--regions", nargs="+", default=["haidian", "harbin"])
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    args = parser.parse_args()

    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

    model = load_teacher(args.device)

    for region in args.regions:
        hr_dir = args.data_root / region / "patches" / "highres_optical"
        out_dir = args.output_root / region
        out_dir.mkdir(parents=True, exist_ok=True)

        by_patch = {}
        for f in sorted(hr_dir.glob("highres_optical_*_patch_*.tif")):
            if f.stem.endswith("_mask"):
                continue
            parts = f.stem.split("_")  # highres, optical, date, patch, id
            patch_id = f"patch_{parts[-1]}"
            by_patch.setdefault(patch_id, []).append(f)

        patch_ids = sorted(by_patch)
        patch_ids = [p for i, p in enumerate(patch_ids) if i % args.num_shards == args.shard_id]
        logger.info("%s: %d patches (shard %d/%d)", region, len(patch_ids), args.shard_id, args.num_shards)

        for i, patch_id in enumerate(patch_ids):
            out_path = out_dir / f"{patch_id}_teacher.pt"
            if out_path.exists():
                continue
            acc, n = None, 0
            for f in by_patch[patch_id]:
                with rasterio.open(f) as src:
                    img = src.read().astype(np.float32)
                if not np.isfinite(img).all():
                    img = np.nan_to_num(img, nan=0.0)
                feat = extract(model, normalize_image(img), args.device)
                acc = feat if acc is None else acc + feat
                n += 1
            if n == 0:
                logger.warning("no highres for %s/%s", region, patch_id)
                continue
            mean_feat = (acc / n).to(torch.float16)
            torch.save({"feature": mean_feat, "num_dates": n, "grid": GRID}, out_path)
            if (i + 1) % 40 == 0:
                logger.info("%s: %d/%d done", region, i + 1, len(patch_ids))
        logger.info("%s finished", region)


if __name__ == "__main__":
    main()
