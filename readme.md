# stable-diffusion-webui-joy-tagger

### 字面意思，joy_caption的webui打标器

模型会自动下载在 `models/joy_caption`目录下

如果使用modelscope下载源，记得先`pip install modelscope`

### 你也可以不当做插件用，直接运行

直接运行，在根目录下运行
```
python -m scripts.app
```

注意，直接运行不要运行install.py，而是先装torch和torchvision，然后pip install -r requirements.txt
