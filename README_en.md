#### README: [ZH](https://github.com/sfsun67/InternVL-MacOS-Tutorials/blob/main/README.md) | [EN](https://github.com/sfsun67/InternVL-MacOS-Tutorials/blob/main/README_en.md)

---
##### 📌 NOTE

**Official InternVL Project Links**: [Code](https://github.com/OpenGVLab/InternVL);  [Documentation](https://internvl.readthedocs.io/en/latest/index.html).

**Supported Platform**: This guide is for reproducing InternVL Chat inference and training on **Apple M-series chips** (e.g., M4 Pro 24GB). It may also be applicable to other M-series chips but has not been tested on **Intel** devices.

InternVL2.5 can be installed, inferred, and trained on Apple M-series chips (M4 Pro 24GB). It supports MPS (float32 / float16 / bfloat16), but Flash Attention must be disabled, and computation must be optimized. LoRA training is feasible (22GB/24GB VRAM), while full training may cause memory overflow. Due to memory and computational limitations, it is suitable for debugging and architectural validation.

# 🚀 Installation 

📌 Refer to the official documentation: [InternVL Installation Guide](https://internvl.readthedocs.io/en/latest/get_started/installation.html)

### 1️⃣ Clone the repository

```bash
git clone https://github.com/sfsun67/InternVL-MacOS-Tutorials.git
```

### 2️⃣ Create and activate the Conda virtual environment
```bash
conda create -n internvl python=3.10
conda activate internvl
```

### 3️⃣ Navigate to the working directory
```bash
cd ./internvl_chat
```
###  4️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
After successful installation, the terminal will display the following:
![Insert image description here](https://i-blog.csdnimg.cn/direct/fa567db0a7b0417d821ed630bfe707ab.png#pic_center)

---
# 🖥️ Local Chat Demo 

📌 Refer to the official documentation: [Local Chat Demo](https://internvl.readthedocs.io/en/latest/get_started/local_chat_demo.html)

### 🔧 Code Modification
#### Modify [model_worker.py](https://github.com/sfsun67/InternVL-MacOS-Tutorials/blob/main/streamlit_demo/model_worker.py#L183)

- Apple M-series chips require setting `device` to `'mps'`, utilizing Metal Performance Shaders (MPS) for inference.
- Supported data types: `torch.float32` provides the best compatibility, but `torch.float16` and `torch.bfloat16` are also loadable.
- Mac has a single GPU, so the `split_model` function is not used to distribute the model across multiple GPUs.

![Insert image description here](https://i-blog.csdnimg.cn/direct/30b3bff9b7aa419e89ec300b6e61d05b.png#pic_center)

### 🔧 Run the Program

1️⃣ Install Streamlit dependencies

Navigate to the root directory (InternVL):
```bash
pip install -r requirements/streamlit_demo.txt
```
After installation, navigate to the working directory:
```bash
cd ./streamlit_demo
```

2️⃣ Set environment variables
```bash
export SD_SERVER_PORT=39999
export WEB_SERVER_PORT=10003
export CONTROLLER_PORT=40000
export CONTROLLER_URL=http://127.0.0.1:$CONTROLLER_PORT
export SD_WORKER_URL=http://127.0.0.1:$SD_SERVER_PORT
```

3️⃣ Start the Streamlit Web Server
```bash
streamlit run app.py --server.port $WEB_SERVER_PORT -- --controller_url $CONTROLLER_URL --sd_worker_url $SD_WORKER_URL
```
🚨 **Note:** The first webpage access may result in an error because the controller has not started yet.
![Insert image description here](https://i-blog.csdnimg.cn/direct/d6c8ffb3af3b473fa99d5766032367b6.png#pic_center)

4️⃣ Start the Controller
Run the following command to start the controller on port `$CONTROLLER_PORT`:
```bash
python controller.py --host 127.0.0.1 --port $CONTROLLER_PORT
```

5️⃣ Start the Model Worker
```bash
python model_worker.py --host 127.0.0.1 --controller $CONTROLLER_URL --port 40001 --worker http://127.0.0.1:40001 --device mps --model-path /Users/sf/codeAlgorithms/VLLM/cache/modelscope/hub/OpenGVLab/InternVL2_5-1B
```

📌 **Memory Usage:**
After startup:
![Insert image description here](https://i-blog.csdnimg.cn/direct/273a34742c5c4e9c849a1bc8e2984876.png#pic_center)

After running for a while, memory usage on M4 Pro 24GB:
![Insert image description here](https://i-blog.csdnimg.cn/direct/2edefe7f66f043bfaa0b184b58f27581.png#pic_center)

### Running Status
Text → Text:
![Insert image description here](https://i-blog.csdnimg.cn/direct/e1518faea24e47fd8074d0c1c8952f71.png#pic_center)

Text + Image → Text:
![Insert image description here](https://i-blog.csdnimg.cn/direct/94fb4b38d18c4a399619bd638925d2ad.png#pic_center)

*Note: Multimodal inference encountered a bug. The backend crashed while processing images. Needs further debugging.*

---
# 🔧 Fine-tune 

⚠️ **MacOS training is limited by memory & MPS support, making it unsuitable for large-scale training. It is primarily for architectural validation & debugging.**

1️⃣ Navigate to the working directory
```bash
cd ./internvl_chat
```

2️⃣ Code Modifications

📌 Main Modifications:
	• Disable Flash Attention
	• Disable Distributed Training
	• Adjust Position Embedding Method (Bilinear Interpolation)
	• Remove NVIDIA-Related Optimizations

### 📢 Code Modifications:
#### Modification: [internvl/patch/init.py](https://github.com/sfsun67/InternVL-MacOS-Tutorials/blob/main/internvl_chat/internvl/patch/__init__.py#L7)
Comment out Flash Attention-related functions.
![Insert Image Here](https://i-blog.csdnimg.cn/direct/589897c1266b498aa7f4e1a154d47ebb.png#pic_center)

#### Modification: [internvl/model/internvl_chat/modeling_intern_vit.py](https://github.com/sfsun67/InternVL-MacOS-Tutorials/blob/main/internvl_chat/internvl/model/internvl_chat/modeling_intern_vit.py#L158)
Use bilinear interpolation in the position embedding function. MPS supports bilinear interpolation, whereas bicubic interpolation is not yet implemented.
![Insert Image Here](https://i-blog.csdnimg.cn/direct/41d2376d3e2b486d9b40e89f6a4f4053.png#pic_center)

#### Modification: internvl/train/internvl_chat_finetune.py → [internvl/train/internvl_chat_finetune_macos.py](https://github.com/sfsun67/InternVL-MacOS-Tutorials/blob/main/internvl_chat/internvl/train/internvl_chat_finetune_macos.py#L46)

- Comment out Flash Attention-related functions.
![Insert Image Here](https://i-blog.csdnimg.cn/direct/5be2c659b8e440f59094cab390adfd78.png#pic_center)

- PyTorch distributed training is not required. `dist.get_rank()` and `dist.get_world_size()` are used to obtain the current process information in a distributed environment.
![Insert Image Here](https://i-blog.csdnimg.cn/direct/63a8db2113fe412487ec72c8fd8ebd1c.png#pic_center)
![Insert Image Here](https://i-blog.csdnimg.cn/direct/6a110f2028004176aecf8f4c5f26e91a.png#pic_center)

#### Modification: Comment out `replace_llama_rmsnorm_with_fused_rmsnorm()`. This function replaces the RMSNorm function in the LLaMA model via monkey patching using NVIDIA libraries to accelerate RMSNorm computation.

#### Modification: These two parameters are used to initialize the `torch.distributed` process group. Comment them out.
![Insert Image Here](https://i-blog.csdnimg.cn/direct/7d83fd5683d842dfb3f29e070803f7a3.png#pic_center)

#### Modification: The monkey patch for the InternVL model involved Flash Attention. Comment it out.
![Insert Image Here](https://i-blog.csdnimg.cn/direct/70441c735b6245fbb85141ec391d14d5.png#pic_center)

#### Modification: Use eager (dynamic computation graph) instead of `flash_attention_2` when loading the pre-trained model.
![Insert Image Here](https://i-blog.csdnimg.cn/direct/517193df3c864119b2bee12036c184df.png#pic_center)
*note: In actual tests, eager (dynamic computation graph) and compile (compiled mode) showed no speed difference on the M4 Pro chip. Further investigation is needed to determine why the compiled mode did not achieve the expected performance on the M4 Pro chip.*  

#### Modification: Since there is no distributed initialization and only a single GPU, remove the `if` statement directly.
![Insert Image Here](https://i-blog.csdnimg.cn/direct/c94801c55cc44f818cf03ea413f51a61.png#pic_center)

#### New Training Configuration Files:
[shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_full_macos.sh](https://github.com/sfsun67/InternVL-MacOS-Tutorials/blob/main/internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_full_macos.sh)

[shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_lora_macos.sh](https://github.com/sfsun67/InternVL-MacOS-Tutorials/blob/main/internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_lora_macos.sh)
- note1: M4 24GB runs out of memory when training the full 1B model.
- note2: If not using `wandb`, remove this line from the configuration file:  `--report_to "wandb" \`

#### New Data Configuration File:
[shell/data/internvl_2_5_fineturn_test_ai2d_train_12k.json](https://github.com/sfsun67/InternVL-MacOS-Tutorials/blob/main/internvl_chat/shell/data/internvl_2_5_fineturn_test_ai2d_train_12k.json)

## 🚀 Run Script After Modifications:
```bash
sh shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_lora_macos.sh
```

### 📦 Model Preparation
InternVL2.5 1B Pre-trained Model
	• 🔗 Download Link
📌 Running the training script will automatically download the model, or manually download it from: [ModelScope](https://www.modelscope.cn/models/OpenGVLab/InternVL2_5-1B)

### 📂 Data Preparation
InternVL Official Dataset:
	• 🔗 [InternVL-Chat-V1-2-SFT-Data](https://huggingface.co/datasets/OpenGVLab/InternVL-Chat-V1-2-SFT-Data/viewer)

Data Format Reference:
	• 📄 [Chat Data Format](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html)

### 🔥 Training
Run the training script:

- LoRA Training:

```bash
sh shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_lora_macos.sh
```
- Full Training (may run out of memory❗):
```bash
sh shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_full_macos.sh
```

### ✅ Training Results
📌 Tested on M4 Pro 24GB:
	• LoRA (Rank=2)
	• VRAM Usage: 22GB/24GB
	• GPU Utilization: >90%

📷 Training Process Screenshot:
![Insert Image Here](https://i-blog.csdnimg.cn/direct/b331929b9f814a07afbf1dcf832a8481.png#pic_center)

🛠️ TODO: Bug & Debug
📌 Text + Image → Text task encounters an error when processing images
	• Possible Cause: MacOS MPS backend does not support certain server-side operations
	• Solution: Further investigate `image_transform` processing

📷 Error Screenshots:
![Insert Image Here](https://i-blog.csdnimg.cn/direct/f9341c1f9c5845358a45ebe1a6915dbb.png#pic_center)

# 📌 Conclusion
✅ MacOS can run InternVL2.5 1B for local inference and training.
🚀 MPS supports float32 / float16 / bfloat16.
⚠️ Flash Attention is not supported.
❗ Full training may run out of memory; MacOS is suitable for local debugging, but cloud training is recommended for full-scale training.

📢 Contact & Contribution
💡 Suggestions are welcome!
📩 GitHub Issues / Pull Requests

**Acknowledgment**: Thanks to the [InternVL](https://github.com/OpenGVLab/InternVL) research team for their work and the community for their interest and support in multimodal models.

