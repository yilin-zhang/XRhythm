# XRhythm

## 使用方法
整个项目的流程介绍，参考在 FTP 上的 `西安鼓乐项目介绍.pdf` 和 `毕业设计论文.pdf` 。但由于在做完小组内的 presentation 后又调整了模型参数，前者的 `参数设定` 一节内容不再适用。具体的参数设定可以参考毕业设计论文，或是模型目录中包含的 Python 脚本。

本代码在 Python 3.6 下开发调试，但 Python 3 的任何版本都应支持。运行 Python 脚本前需要安装 `keras` 和 `pretty_midi` 两个包。

生成乐曲只需要确定模型。模型文件可在 FTP 上下载，下载后在该工程目录中创建 `models` 目录，并将模型文件解压至该目录下（下文提到的目录 `midi` 和 `datasets` 也都需要自行创建）。该目录下含两个子目录：`2201905310621` 中的模型文件是只经过现代音乐训练后的模型，`201905311043` 中的模型文件是在前一个模型基础上加入鼓乐训练的模型。目录结构为：

```
models
  ├─── 201905310621
  └─── 201905311043
```

在这两个模型子目录中，分别存放了当时运行训练任务所使用的 Python 脚本。由于代码经过一定修改，这两个脚本不一定仍能使用，但是其中定义的参数可供参考。`logs` 目录中是模型训练的日志。`.hdf5` 文件是模型文件。

使用方法参照 `example.py` 的代码使用示例。注意使用时确认模型文件的路径和输出的路径。

### 重新运行数据预处理
如果想要重新运行数据预处理，需要将原始 MIDI 文件放在 `midi` 目录下，目录结构为：

```
midi
  ├─── raw_midi
  └─── xadrum_midi
```

其中 `raw_midi` 是现代音乐数据集，`xadrum_midi` 是鼓乐数据集。前者可以在 Lakh MIDI Dataset 的网站上下载（注意是 Clean MIDI Subset），后者在 FTP 上的 `midi` 文件夹中，名称为 `xadrum_midi.zip` ，解压缩即可。

运行现代音乐预处理代码后，会在 `midi` 目录下多出 `processed_midi` 目录。运行鼓乐音乐预处理代码后，会在 `midi` 目录下多出 `xadrum_processed_midi` 文件夹。目录结构如下所示：

```
midi
  ├─── raw_midi
  ├─── processed_midi
  ├─── xadrum_midi
  └─── xadrum_processed_midi
```
### 重新运行数据导出
如果想要重新运行数据文件导出，需要已存在上述的预处理 MIDI 目录。如果不想从头运行，可以在 FTP 上的 `midi` 文件夹中下载两个对应名称的 zip 文件并解压。

运行对应数据文件导出后，会在 `datasets` 目录创建对应的子目录。需要注意的是，鼓乐数据文件导出后，我实际做了对训练集、验证集、测试集数据文件的手动调整，因此最好直接使用 FTP 上 `datasets` 文件夹提供的 zip 文件进行解压。

### 重新训练模型
如果想要重新进行训练，可以使用 FTP 上 `datasets` 文件夹提供的两个 zip 文件，解压到 `datasets` 目录下，目录结构为：

```
datasets
  ├─── dataset
  └─── xadrum_dataset
```

## 代码说明

### 可执行代码
- `example.py`
包含生成音乐的示例代码。

- `preprocess_midi.py`
运行该代码完成现代音乐 MIDI 文件的预处理。包括去除打击乐、量化和基于 Skyline 算法的旋律提取。

- `preprocess_xadrum.py`
运行该代码完成鼓乐 MIDI 文件的预处理，仅进行量化处理。

- `get_dataset.py`
运行该代码提取经过预处理的现代音乐 MIDI 文件的信息，导出 pickle 文件。

- `get_xadrum_dataset.py`
运行该代码提取经过预处理的西安鼓乐 MIDI 文件的信息，导出 pickle 文件。

- `cal_batches.py`
运行该代码给定 pickle 数据文件的目录和 batch size 计算迭代一轮（epoch）需要的步数。
输出的 `n_train` 和 `n_valid` 需分别对应赋值给 `lstm_model.py` 和 `lstm_model_continue.py` 中的 `steps_per_epoch` 和 `validation_steps` 。

- `train_modern.py`
运行该代码将现代音乐数据送入训练。

- `train_xadrum.py`
运行该代码加载已经使用现代音乐训练后的 LSTM 模型，将鼓乐数据送入训练。

### 函数与类定义代码

所有函数与类定义代码都在 `xrlib` 目录下。

- `model.py`
定义了 `XRModel` 类，其中定义了 LSTM 模型，以及训练相关的类方法。

- `generate.py`
定义了 `RhythmGenerator` 类。主要方法： `generate_from_bar_pitch_list` 和 `generate_from_pitch_list` 。前者接受含有小节信息的音高序列作为参数，后者只接受音高序列作为参数。

- `midi_data.py`
定义了 `MidiData` 类，是基于 `pretty_midi` 提供的 `PrettyMidi` 类的封装，额外定义了一些方法。主要方法是 `dropdrum` 、 `quantize` 和 `skyline` 。

- `utils.py`
定义了一些工具函数。

- `configs.py`
定义了一些常量。
