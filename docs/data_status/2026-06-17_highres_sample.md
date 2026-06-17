# 高分辨率样本数据状态（2026-06-17）

## 天仪 SAR（ModelScope）

- 数据集：`WeijieWu/haidian_sar_2025`
- 输出目录：`/data/xuannv_embedding/raw/haidian/highres_sar`
- 状态：**partial**
- 实际下载量：约 **33 GB**，已落盘 `.zip` 文件 **79 个**，目录下还有 ModelScope 元数据文件 `.msc`
- 备注：
  - 登录 ModelScope 成功（token 已生效）。
  - 数据集总文件数超过 600 个（ listing 显示 602+），单文件约 330–340 MB。
  - 受任务 10 分钟超时限制，下载在运行约 5 分钟后被中断；`snapshot_download` 支持断点续传，后续可重新运行同一命令继续下载剩余文件。

## 百度网盘高分辨率光学/SAR

- 远程目录：`/玄女科技/`
- BaiduPCS-Go 登录状态：正常（uid=6511215751）
- links 文件：
  - `scripts/data/baidu_links_harbin.txt`
  - `scripts/data/baidu_links_haidian.txt`
- 状态：**listed + partial downloaded**
- 已下载样本：
  - `/玄女科技/haerbin_patches/patch_000112` → `/data/xuannv_embedding/raw/harbin/highres_optical/patch_000112`
  - 大小：**57 MB**，含 8 个 `.tif` 与 1 个 `.json`，对应哈尔滨 2025-12 与 2026-05 两个时相的高分辨率 patch 光学影像。
- 备注：
  - `/玄女科技/` 目录总览：10 个文件（52.64 GB）+ 7 个子目录。
  - 发现的高分辨率相关资源：
    - `haerbin_patches/`：44 个 patch 子目录，每个约 50–60 MB（已下载 `patch_000112` 作为样本）。
    - `haerbin_patches.zip`：41.22 GB（未下载）。
    - `planetscene-patches.zip`：5.96 GB（未下载）。
    - `beijin_rs_psscene_analytic_sr_udm2.zip`：5.45 GB（未下载）。
    - `哈尔滨_202512/哈尔滨新区_BJ3_202512_DOM_RGB.tif`：22.14 GB（未下载）。
    - `哈尔滨_202605/哈尔滨江北区域.tif`：22.14 GB（未下载）。
    - `海淀区sar/玄女底座/中国北京市点位1_干涉/`：与 ModelScope 数据集结构对应，含 44 个轨道子目录，每子目录约 672 MB。
  - 由于整幅影像和全量压缩包均超过 1 GB，本次仅记录远程路径并下载一个 harbin patch 样本；后续大规模下载可直接使用 `scripts/data/download_baidu.py` 配合已生成的 links 文件。
