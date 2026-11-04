# 机械臂与静止参考板 `arm/`

[English](README.en.md) · [实验指导](../实验指导.md)


机械臂携带双目组件，使用两个经过测试的预设位姿：命令行使用 `upper` 表示上位姿、`lower` 表示下位姿；扫描清单中的 `arm_pose_id` 则分别为 0、1。静止 ChArUco 板在扫描过程中提供每视角相机相对参考坐标系的位姿，不能随转盘旋转。

## 文件

| 文件 | 作用 |
| --- | --- |
| `arm.py` | 通过串口发送机械臂预设位姿命令，可列出、试算、连接和读取控制器反馈 |
| `generate_charuco_board.py` | 生成固定板打印图及几何配置 |
| `fixed_board.py` | 检测 ChArUco 角点、估计板到相机位姿并计算位姿差 |
| `capture_fixed_board.py` | 在上下位姿重复拍摄固定板，保存原图、角点图与清单 |
| `targets/charuco_7x5_10mm_7mm_config.json` | 固定板尺寸、字典和角点定义；须与实际打印尺寸一致 |
| `targets/*.png`, `targets/*.pdf` | 固定板打印素材 |
| `tests/` | 控制命令和固定板几何测试 |

## 使用示例

在项目根目录执行，先通过 `--dry-run` 检查命令，再接实际串口：

```powershell
python arm\arm.py --list
python arm\arm.py --pose upper --dry-run
python arm\arm.py --port COM7 --pose upper
python arm\arm.py --port COM7 --pose lower --readback
python arm\generate_charuco_board.py
python arm\capture_fixed_board.py --session-id board_check_01 --port COM7 --repeats 3
```

`capture_fixed_board.py` 默认读取 `targets/` 中的板配置，可用 `--board` 指定其他配置；`--left-serial`、`--right-serial`、`--exposure-us` 和 `--min-corners` 可调整采集条件。原始与检测结果默认存于 `raw/arm_fixed_board/<session-id>/`，准确路径以该命令 `--help` 的 `--output-root` 默认值为准。串口发送与等待不能单独证明机械臂实际到位；以固定板检测结果检查位姿稳定性。运行测试：`python -m pytest arm/tests -q`。
