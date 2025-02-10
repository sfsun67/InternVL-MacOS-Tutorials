**InternVL 官方项目地址**：[代码](https://github.com/OpenGVLab/InternVL);  [文档](https://internvl.readthedocs.io/en/latest/index.html).

**适用平台**: 本指南针对 **Apple M 系列芯片**（如 M4 Pro 24GB）复现 InternVL Chat 推理和训练。也可适配其他 M 型号芯片，尚未在 **Intel** 设备上测试。

## 🚀 Installation | 安装

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
## 🖥️ Local Chat Demo | 本地推理

📌 参考官方文档: [Local Chat Demo](https://internvl.readthedocs.io/en/latest/get_started/local_chat_demo.html)

🔧 代码修改
	
	•	Apple M 系列芯片 需修改 device 为 'mps'，使用 Metal Performance Shaders (MPS) 进行推理。
	•	数据类型支持: torch.float32 兼容性最佳，但 torch.float16 和 torch.bfloat16 也可以加载。
	•	Mac 仅有单 GPU，不使用 split_model 函数分配模型到不同的 GPU 上。。

1️⃣ 安装 Streamlit 依赖

cd ./ # 确保在主目录
```bash
pip install -r requirements/streamlit_demo.txt
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

4️⃣ 启动控制器
```bash
python controller.py --host 127.0.0.1 --port $CONTROLLER_PORT
```
5️⃣ 启动模型 Worker
```bash
python model_worker.py --host 127.0.0.1 --controller $CONTROLLER_URL --port 40001 --worker http://127.0.0.1:40001 --device mps --model-path /Users/sf/codeAlgorithms/VLLM/cache/modelscope/hub/OpenGVLab/InternVL2_5-1B
```
📌 内存占用情况:

运行一段时间后，M4 Pro 24GB 内存占用如下：

🔧 Fine-tune | 微调

⚠️ MacOS 训练受限于内存 & MPS 支持，不适用于高效大规模训练，仅用于架构验证 & Debug

1️⃣ 进入训练目录
```bash
cd ./internvl_chat
```
2️⃣ 代码修改

📌 主要修改点：
	•	禁用 Flash Attention
	•	禁用分布式训练
	•	调整位置嵌入方式（双线性插值）
	•	去掉 NVIDIA 相关优化

📌 修改文件:
	•	internvl/patch/__init__.py 👉 注释 Flash Attention
	•	internvl/model/internvl_chat/modeling_intern_vit.py 👉 改为双线性插值
	•	internvl/train/internvl_chat_finetune.py 👉 创建 Mac 版 internvl_chat_finetune_macos.py
	•	去掉 torch.distributed
	•	禁用 replace_llama_rmsnorm_with_fused_rmsnorm()
	•	flash_attention_2 替换为 eager 计算

🚀 修改后，启动脚本：
```bash
sh shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_lora_macos.sh
```
📦 Model Preparation | 模型准备

InternVL2.5 1B 预训练模型
	•	🔗 下载地址

📌 手动下载 或者 在训练脚本中自动下载。

📂 Data Preparation | 数据准备

InternVL 官方提供数据集:
	•	🔗 InternVL-Chat-V1-2-SFT-Data

数据格式参考:
	•	📄 Chat Data Format

🔥 Training | 训练

启动训练脚本
	•	LoRA 训练:
```bash
sh shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_lora_macos.sh
```
	•	Full 训练（可能内存溢出❗）:
```bash
sh shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_full_macos.sh
```
✅ Training Results | 训练结果

📌 实测 M4 Pro 24GB
	•	LoRA（Rank=2）
	•	显存占用: 22GB/24GB
	•	GPU 利用率: >90%

📷 训练过程截图:

🛠️ Bug & Debug

📌 文字 + 图片 → 文字 任务 出现 处理图片时错误
	•	错误日志:
	•	可能原因: MacOS MPS 后端对部分 torchvision 操作不支持
	•	解决方案: 需深入排查 image_transform 相关处理

📷 错误截图:

📌 结论

✅ MacOS 可运行 InternVL2.5 本地推理和训练
🚀 MPS 支持 float32 / float16 / bfloat16
⚠️ 不支持 Flash Attention，LoRA 训练较稳定
❗ 训练时可能内存溢出（需优化 batch size）

📢 Contact & Contribution

💡 欢迎贡献改进意见！
📩 GitHub Issues / Pull Requests

鸣谢: 感谢 InternVL 研究团队的研究成果，以及社区对多模态模型的关注与支持。