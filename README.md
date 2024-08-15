# 🌋 Hallucination mitigation through instructive fine tuning and preference alignment

## Install
Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```


```

### CLI Inference

Chat about images using LLaVA without the need of Gradio interface. It also supports multiple GPUs, 4-bit and 8-bit quantized inference. With 4-bit quantization, for our LLaVA-1.5-7B, it uses less than 8GB VRAM on a single GPU.

```Shell
python -m llava.serve.cli \
    --model-path \
    --image-file  \
    --load-4bit
```

<img src="images/demo_cli.gif" width="70%">

## Train

trained if performed on 2 A100 GPUs with 40 hours, 2epochs of instructive finetuing followed by DPO


### Download checkpoints 

link: sdoouyangboya/llava-finetued-dpo


### Visual Instruction Tuning

data:
gdown https://drive.google.com/uc?id=1pWkxE2kqpys1VdwBi99ZXN6-XY5SqhwU

training: finetune.sh

### DPO
data:https://huggingface.co/datasets/juliozhao/hadpo-data/tree/main/hadpo/llava-v1.5

training: dpo.sh


## Evaluation based on Gaive benchamrk

gaive_eval.sh

## Result
| Model  | Gaive accuracy |
|----------|----------|
| llava.1.6 | 5.41 |
| llava.1.6_finetung | 6.19 |
| llava.1.6_finetung_dpo | 6.46 | 

