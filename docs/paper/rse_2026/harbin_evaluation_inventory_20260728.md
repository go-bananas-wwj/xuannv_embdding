# 哈尔滨论文评测清单审计

日期：2026-07-28  
状态：空间清单、AEF 来源清单及两份官方 52N COG 均已冻结并校验。

## 空间清单

- 原始区域网格：424 patch，来自 `configs/regions/harbin_patches.json`。
- 训练/embedding manifest：424 patch，ID 形式为 `harbin_patch_******`，并通过
  `source_patch_id` 关联原始网格和 OSM mask 的 `patch_******` ID。
- 论文空间划分：`configs/eval/harbin_spatial_5fold_buffer1_20260728.json`。
- 为保证每个测试 cluster 都是完整 2×2 地理块，并保留完整 block 的缓冲区，主评测
  使用 380 patch（95 个完整 2×2 block）。44 个不完整边缘 patch 不进入任何一方的
  train/validation/test/buffer；它们不是被某一个模型单独删除。
- 每折包含 40 test patch、20 validation patch；train 和 buffer 数量因边界形状不同而变。

## ID 与标签绑定

空间 split 已使用 manifest 的 `harbin_patch_*` ID，这是 Xuannv embedding 的输出 ID。
每条评测记录同时保留其 `source_patch_id=patch_*`，作为 OSM 原始 mask 的定位 ID。这样
不会通过字符串替换在运行时临时猜测标签路径。

## AEF annual 2025 来源清单

`configs/eval/harbin_aef_annual_2025_coverage_inventory_20260728.json` 含 380 条无标签
记录。每条记录冻结：

1. `harbin_patch_*` embedding ID；
2. `patch_*` 原始标签 ID；
3. 实际 S2 参考栅格的路径、SHA-256、CRS、transform、bounds、128×128 shape；
4. AEF official annual 2025 的候选 COG URI。

哈尔滨跨越两张官方 52N COG：

| 覆盖 patch 数 | AEF 2025 COG |
| ---: | --- |
| 158 | `xeuzxi87y7s7v28mg-0000008192-0000008192.tiff` |
| 222 | `xodbkdwud1cbasbyr-0000000000-0000008192.tiff` |

## COG 下载与可读性验证

两份 COG 已通过官方 Source Cooperative HTTPS 下载并记录在
`configs/eval/harbin_aef_annual_2025_cog_lock_20260728.json`。下载期间遇到可恢复的 HTTP/2
stream 与连接提前关闭错误；恢复时使用同一官方 HTTPS URI、清洁动态库环境、HTTP/1.1 与
断点续传。没有使用非官方镜像，也没有使用任何 `.part` 文件。

| COG | Bytes | SHA-256 |
| --- | ---: | --- |
| `xeuzxi87y7s7v28mg-0000008192-0000008192.tiff` | 2,671,334,309 | `5e35b40cc36616a937e33311f2d83597fc27504e7c5db2006088ae576483ccf6` |
| `xodbkdwud1cbasbyr-0000000000-0000008192.tiff` | 2,489,175,968 | `bebaa7370c45a0f821f42687e4e221c048814ee26b6e9c46dc54251f39c4daf5` |

2026-07-28 验证：两个文件均可由 rasterio 以 GeoTIFF 打开，具有 EPSG:32652、64 个 int8
band、8192×8192 栅格、nodata=-128；对四个角的 window 分别读取 band 1/32/64 均成功。
环境中未提供 `gdalinfo`，故 GDAL Python/rasterio 驱动验证作为该节点的 GDAL 可读性证据。

允许的下一步是按上述 380 patch 覆盖清单导出 annual 2025 AEF map，并与 scratch P10C 和
冻结海淀 P10C 的 2026-04 embedding 使用同一空间划分、shot 清单、Conv3x3 读取头和
validation-only 阈值选择规则。
