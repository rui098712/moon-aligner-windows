# Moon Aligner 月亮照片自动配准、裁切与多点堆栈 v1.5

程序会自动检测月面，纠正照片之间的位移与小幅旋转，排除缺边或无法配准的照片，并输出尺寸完全一致的 TIFF。它可以进行平均、中位数或多点局部堆栈，生成噪声更低、受大气扰动影响更小的月面照片。

原始照片不会被修改。

## 直接下载Windows版

[下载最新Windows 64位免安装版](https://github.com/rui098712/moon-aligner-windows/releases/latest/download/MoonAligner-Windows-x64.zip)

下载后解压并双击 `MoonAligner.exe` 即可运行，不需要安装Python。推荐使用64位Windows 10或Windows 11。

## 支持格式

- JPG / JPEG
- PNG
- 8位或16位TIFF

NEF等RAW文件请先用尼康NX Studio批量导出为16位TIFF。不要提前逐张裁切、缩放、锐化或降噪。

## 方法一：直接运行源码版

1. 安装[64位Python 3.12](https://www.python.org/downloads/windows/)，安装时勾选 `Add python.exe to PATH`。
2. 双击 `Run-Moon-Aligner.bat`。
3. 第一次运行会自动安装NumPy和OpenCV，需要联网等待几分钟。
4. 依次选择照片文件夹和输出文件夹。

以后再次使用时，只需双击 `Run-Moon-Aligner.bat`。

## 方法二：生成独立EXE

1. 在64位Windows 10或Windows 11电脑上安装Python 3.12。
2. 双击 `Build-Windows-EXE.bat`。
3. 第一次构建需要联网下载NumPy、OpenCV和PyInstaller。
4. 等待窗口显示“构建完成”。
5. 最终程序位于：

```text
dist\月亮裁切与堆栈.exe
```

这个EXE内置Python及依赖，可以复制到其他64位Windows 10/11电脑使用，不需要另外安装Python。

PyInstaller不是跨平台编译器，因此Windows EXE必须在Windows系统中生成，不能直接在macOS上构建。

## 使用方法

1. 双击程序。
2. 在主界面中点击“选择…”指定原始照片文件夹。
3. 点击第二个“选择…”指定输出文件夹；不修改时默认输出到原图目录下的 `moon_aligned`。
4. 选择处理方式：
   - `平均堆栈`：默认推荐，降低随机噪声的效果较好。
   - `中位数堆栈`：更适合排除偶然出现的亮点、飞鸟等干扰。
   - `多点局部堆栈`：实验功能，在月面建立多个对齐点，分别校正局部扰动并择优融合。
   - `不堆栈`：只完成原有的配准与统一裁切。
5. “使用最清晰的”可以控制参与堆栈的照片比例；例如50%表示只使用清晰度排名前一半的照片。
6. 如果只想要最终堆栈图，可以取消“同时保留每张配准裁切后的 TIFF”。
7. 点击“开始处理”，程序才会开始工作。
8. 在主界面查看实时日志和处理状态，完成后点击“打开输出文件夹”。

高级设置通常保持默认即可。处理期间界面会暂时锁定目录和参数，避免误操作；照片处理在后台线程运行，窗口不会失去响应。

输出文件包括：

- `0001_..._aligned.tif`：已经配准并统一裁切的照片。
- `moon_stack_mean.tif`：平均堆栈结果。
- `moon_stack_median.tif`：中位数堆栈结果。
- `moon_stack_local.tif`：多点局部堆栈结果。
- `local_alignment_points.png`：多点模式自动生成的对齐点网格预览。
- `alignment_report.csv`：每张照片的清晰度、相关度、位移、旋转和跳过原因。
- `processing_report.json`：程序版本、处理参数、帧数和裁切范围，方便复现实验。

只会生成当前选择的那一种堆栈结果。如果取消保留单张 TIFF，输出目录中不会出现 `0001_..._aligned.tif` 文件。

## 堆栈说明

- 堆栈不会进行锐化、降噪或色彩增强，输出仍是便于后期处理的 TIFF。
- 输入照片应来自同一拍摄序列，并保持相同的尺寸、颜色通道和位深。
- 程序采用分块堆栈和磁盘临时缓存，能避免一次性把所有照片装入内存；处理时仍需确保系统盘有足够临时空间。
- 多点模式会先建立稳定的平均参考图，再在有纹理的月面自动放置重叠对齐点；每个点分别估计亚像素位移、按局部清晰度选帧，最后使用平滑窗口融合。
- “局部块大小”保持 `0` 时会根据月面尺寸自动选择。数值较小能追踪更细的局部变形，但速度更慢、弱纹理处也更容易误配；数值较大则更稳定。
- 多点模式会在每个对齐点独立选出指定比例的清晰帧，所以不同月面区域可能使用不同的照片组合。
- 该模式借鉴公开的 lucky imaging 和多对齐点工作流程，并不是 AutoStakkert 内部算法的复制品。目前不包含 SER/视频读取、Drizzle、RGB 通道校准、波形锐化等功能。

## GitHub自动测试和构建

项目包含 `.github/workflows/windows-build.yml`。代码推送到GitHub后会自动：

1. 在Windows环境安装Python 3.12、NumPy和OpenCV。
2. 运行平均、中位数、多点局部堆栈以及处理报告测试。
3. 使用PyInstaller构建无需安装Python的 `MoonAligner.exe`。
4. 将 `MoonAligner-Windows-x64.zip` 保存为Actions构建产物。

在GitHub仓库页面打开 `Actions`，选择 `Test and build Windows app`，即可查看运行结果并下载构建产物。推送形如 `v1.5.0` 的Git标签时，工作流还会自动创建对应的GitHub Release。

本地运行测试：

```bash
python -m unittest discover -s tests -v
```

## 常见问题

### 提示“未检测到月面”

确认月亮明显亮于背景、没有严重过曝，而且月亮没有完全超出画面。

### 有照片被跳过

通常是照片过于模糊、云层遮挡、曝光差异过大，或者月亮已经出框。可查看 `alignment_report.csv` 中的原因。

### EXE启动比较慢

单文件EXE第一次启动需要释放内置组件，等待几秒属于正常现象。

### Windows安全提示

自行构建的EXE没有商业代码签名。若Windows SmartScreen提示未知发布者，请确认文件是你自己构建的，再选择“更多信息”→“仍要运行”。

