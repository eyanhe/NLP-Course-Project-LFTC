# GitHub与Hugging Face分工

- [GitHub](https://github.com/eyanhe/NLP-Course-Project-LFTC)：源码、配置、研究设计、报告、图表、PPT、Excel、结果CSV/JSON和逐样本预测。
- [Hugging Face](https://huggingface.co/datasets/Eyanhe/LFTC_Dataset_And_Checkpoints)：数据集、统一输入、基础模型、训练权重、特征，以及历史候选检查点。文件在`artifacts/`下保留原仓库相对路径。
- [文件清单](file-manifest.json)：每项的原路径、目标、大小和内容校验值；`both`表示必要的小型元数据在两处保留。

GitHub源码与结果可以单独阅读；运行推理或复现需要下载相应的大资产。Hugging Face下载后，将其中`artifacts/`的内容按原相对路径合并到GitHub项目目录。

发布分两步进行。`huggingface_revision`尚为空、`sha256_status`为`pending_large_asset_upload`的资产，表示尚未完成HF发布与内容核验；不能仅凭清单断言这些文件已在线。最终上传回执将记录远端提交和核验结果。

原始资产锁、模型元数据和冻结文件继续保留。各原始数据集与预训练模型沿用其各自来源与许可，不能把整个混合资产集合当作统一重新许可的数据集。

本机运行环境、工具、凭据、私有集群登录辅助脚本、可重建的MinHash工作缓存和重复传输压缩包留在本地。原始数据和权重不会因发布而从本地删除。

[当前报告与成品](../../report/research-v3/README.md) · [研究沿革](../PROJECT-HISTORY.md)
