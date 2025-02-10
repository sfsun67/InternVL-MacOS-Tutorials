#### README: [ZH](https://github.com/sfsun67/InternVL-MacOS-Tutorials/blob/main/README.md) | [EN](https://github.com/sfsun67/InternVL-MacOS-Tutorials/blob/main/README_en.md) 

---
##### 📌 NOTE

**InternVL 官方项目地址**：[代码](https://github.com/OpenGVLab/InternVL);  [文档](https://internvl.readthedocs.io/en/latest/index.html).

**适用平台**: 本指南针对 **Apple M 系列芯片**（如 M4 Pro 24GB）复现 InternVL Chat 推理和训练。也可适配其他 M 型号芯片，尚未在 **Intel** 设备上测试。


InternVL2.5 可在 Apple M 系列芯片（M4 Pro 24GB）上安装、推理、与训练。支持 MPS (float32 / float16 / bfloat16)，但需禁用 Flash Attention 并优化计算。 LoRA 训练可行（显存 22GB/24GB），Full 训练可能溢出。 受内存和算力限制，适用于调试和架构验证。

# 🚀 Installation | 安装

📌 参考官方文档: [InternVL Installation Guide](https://internvl.readthedocs.io/en/latest/get_started/installation.html)

### 1️⃣ 克隆代码库

```bash
git clone https://github.com/sfsun67/InternVL-MacOS-Tutorials.git
```

### 2️⃣ 创建并激活 Conda 虚拟环境
```bash
conda create -n internvl python=3.10
conda activate internvl
```

### 3️⃣ 进入工作目录
```bash
cd ./internvl_chat
```
###  4️⃣ 安装依赖
```bash
pip install -r requirements.txt
```
安装成功后，终端会显示如下内容：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fa567db0a7b0417d821ed630bfe707ab.png#pic_center)

---
# 🖥️ Local Chat Demo | 本地推理

📌 参考官方文档: [Local Chat Demo](https://internvl.readthedocs.io/en/latest/get_started/local_chat_demo.html)

### 🔧 代码修改
####  修改 [model_worker.py](https://github.com/sfsun67/InternVL-MacOS-Tutorials/blob/main/streamlit_demo/model_worker.py#L183)
	
	•	Apple M 系列芯片 需修改 device 为 'mps'，使用 Metal Performance Shaders (MPS) 进行推理。
	•	数据类型支持: torch.float32 兼容性最佳，但 torch.float16 和 torch.bfloat16 也可以加载。
	•	Mac 仅有单 GPU，不使用 split_model 函数分配模型到不同的 GPU 上。。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/30b3bff9b7aa419e89ec300b6e61d05b.png#pic_center)
### 🔧 执行程序

1️⃣ 安装 Streamlit 依赖

进入主目录（InternVL）
```bash
pip install -r requirements/streamlit_demo.txt
```
环境依赖安装完毕后，进入工作目录
```bash
cd ./streamlit_demo
```
2️⃣ 设置环境变量
```bash
export SD_SERVER_PORT=39999
export WEB_SERVER_PORT=10003
export CONTROLLER_PORT=40000
export CONTROLLER_URL=http://127.0.0.1:$CONTROLLER_PORT
export SD_WORKER_URL=http://127.0.0.1:$SD_SERVER_PORT
```
3️⃣ 启动 Streamlit Web 服务器
```bash
streamlit run app.py --server.port $WEB_SERVER_PORT -- --controller_url $CONTROLLER_URL --sd_worker_url $SD_WORKER_URL
```
🚨 注意: 服务器启动后，初次访问网页可能会报错，这是因为控制器尚未启动。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d6c8ffb3af3b473fa99d5766032367b6.png#pic_center)


4️⃣ 启动控制器
运行以下命令以在端口`$CONTROLLER_PORT`上启动控制器：
```bash
python controller.py --host 127.0.0.1 --port $CONTROLLER_PORT
```
5️⃣ 启动模型 Worker
```bash
python model_worker.py --host 127.0.0.1 --controller $CONTROLLER_URL --port 40001 --worker http://127.0.0.1:40001 --device mps --model-path /Users/sf/codeAlgorithms/VLLM/cache/modelscope/hub/OpenGVLab/InternVL2_5-1B
```
📌 内存占用情况:
启动后内存占用：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/273a34742c5c4e9c849a1bc8e2984876.png#pic_center)


运行一段时间后，M4 Pro 24GB 内存占用如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2edefe7f66f043bfaa0b184b58f27581.png#pic_center)
### 运行情况
文字 → 文字：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e1518faea24e47fd8074d0c1c8952f71.png#pic_center)
文字+图片 → 文字：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/94fb4b38d18c4a399619bd638925d2ad.png#pic_center)
*note: 多模态推理出现 Bug，看后台是处理图片时服务端出错。有时间再排查。*


---
# 🔧 Fine-tune | 微调

⚠️ MacOS 训练受限于内存 & MPS 支持，不适用于高效大规模训练，仅用于架构验证 & Debug

1️⃣ 进入工作目录
```bash
cd ./internvl_chat
```
2️⃣ 代码修改

📌 主要修改点：
	•	禁用 Flash Attention
	•	禁用分布式训练
	•	调整位置嵌入方式（双线性插值）
	•	去掉 NVIDIA 相关优化

### 📢代码修改:
#### 修改：[internvl/patch/init.py](https://github.com/sfsun67/InternVL-MacOS-Tutorials/blob/main/internvl_chat/internvl/patch/__init__.py#L7)
注释掉 Flash Attention 相关函数
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/589897c1266b498aa7f4e1a154d47ebb.png#pic_center)

#### 修改：[internvl/model/internvl_chat/modeling_intern_vit.py](https://github.com/sfsun67/InternVL-MacOS-Tutorials/blob/main/internvl_chat/internvl/model/internvl_chat/modeling_intern_vit.py#L158)
位置嵌入函数中，使用双线性插值（bilinear）。mps 双线性插值（bilinear） 已实现并可用，而 双三次插值（bicubic） 尚未实现。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/41d2376d3e2b486d9b40e89f6a4f4053.png#pic_center)

#### 修改：internvl/train/internvl_chat_finetune.py → [internvl/train/internvl_chat_finetune_macos.py](https://github.com/sfsun67/InternVL-MacOS-Tutorials/blob/main/internvl_chat/internvl/train/internvl_chat_finetune_macos.py#L46)

- 注释掉 Flash Attention 相关函数
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5be2c659b8e440f59094cab390adfd78.png#pic_center)

- 不需要PyTorch 的分布式训练。dist.get_rank() 和 dist.get_world_size() 是用于获取当前进程在分布式环境中的信息。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/63a8db2113fe412487ec72c8fd8ebd1c.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6a110f2028004176aecf8f4c5f26e91a.png#pic_center)


#### 修改：注释掉 replace_llama_rmsnorm_with_fused_rmsnorm()。该函数使用猴子补丁对 llama 模型进行均方根层归一化函数替换。使用 Nvidia 库加速 RMSnorm 计算。


#### 修改：这两个参数，用来初始化 torch.distributed 分布式进程组。注释掉。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7d83fd5683d842dfb3f29e070803f7a3.png#pic_center)



#### 修改：这里是 InternVL 模型打的猴子补丁，用到了 Flash Attention，都注释掉。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/70441c735b6245fbb85141ec391d14d5.png#pic_center)


#### 修改：加载预训练模型时，使用 eager （动态计算图）替换 flash_attention_2。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/517193df3c864119b2bee12036c184df.png#pic_center)
*note: 实际测试，eager （动态计算图）和 compile（编译模式）在 M4 Pro 芯片上并没有速度上的差异。需要排查为什么（编译模式）在 M4 Pro 芯片上出现性能未达预期的情况。*  


#### 修改：没有分布式初始化，并且只有一个 GPU，直接去掉 if 就行
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c94801c55cc44f818cf03ea413f51a61.png#pic_center)


#### 新增训练配置文件：
[shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_full_macos.sh](https://github.com/sfsun67/InternVL-MacOS-Tutorials/blob/main/internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_full_macos.sh)

[shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_lora_macos.sh](https://github.com/sfsun67/InternVL-MacOS-Tutorials/blob/main/internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_lora_macos.sh)
- note1: M4 24GB 会在全量训练 1B 模型中爆内存。
- note2: 如果不使用 wandb，请在配置文件中删除这行：  --report_to "wandb" \

#### 新增数据配置文件：
[shell/data/internvl_2_5_fineturn_test_ai2d_train_12k.json](https://github.com/sfsun67/InternVL-MacOS-Tutorials/blob/main/internvl_chat/shell/data/internvl_2_5_fineturn_test_ai2d_train_12k.json)


## 🚀 修改后，启动脚本：
```bash
sh shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_lora_macos.sh
```
### 📦 Model Preparation | 模型准备

InternVL2.5 1B 预训练模型
	•	🔗 下载地址

📌 执行训练脚本将会自动下载模型，或者手动在：https://www.modelscope.cn/models/OpenGVLab/InternVL2_5-1B 下载
### 📂 Data Preparation | 数据准备

InternVL 官方提供数据集:
	•	🔗 [InternVL-Chat-V1-2-SFT-Data](https://huggingface.co/datasets/OpenGVLab/InternVL-Chat-V1-2-SFT-Data/viewer)

数据格式参考:
	•	📄 [Chat Data Format](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html)

### 🔥 Training | 训练

启动训练脚本
- LoRA 训练:
```bash
sh shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_lora_macos.sh
```

- Full 训练（可能内存溢出❗）:
```bash
sh shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_full_macos.sh
```

### ✅ Training Results | 训练结果

📌 实测 M4 Pro 24GB
	•	LoRA（Rank=2）
	•	显存占用: 22GB/24GB
	•	GPU 利用率: >90%
	
📷 训练过程截图:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b331929b9f814a07afbf1dcf832a8481.png#pic_center)



🛠️ TODO：Bug & Debug

📌 文字 + 图片 → 文字 任务 出现 处理图片时错误
	•	可能原因: MacOS MPS 后端对服务端部分操作不支持
	•	解决方案: 需深入排查 image_transform 相关处理

📷 错误截图:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f9341c1f9c5845358a45ebe1a6915dbb.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/89a69f7d43e8484f978f140e464907fa.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6f66f7dc521b471f8a3f72d3224c79a0.png#pic_center)

# 📌 结论

✅ MacOS 可运行 InternVL2.5 1B 本地推理和训练
🚀 MPS 支持 float32 / float16 / bfloat16
⚠️ 不支持 Flash Attention
❗ 全量训练时可能内存溢出，MacOS 仅适用于本地调试，正式训练还需要上云。

📢 Contact & Contribution

💡 欢迎贡献改进意见！
📩 GitHub Issues / Pull Requests

**鸣谢**: 感谢  [InternVL](https://github.com/OpenGVLab/InternVL) 研究团队的研究成果，以及社区对多模态模型的关注与支持。


