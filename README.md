# Meta Chameleon

**[Paper](//arxiv.org/abs/2405.09818) | [Blog](//ai.meta.com/blog/meta-fair-research-new-releases/) | [Model Checkpoint Download](//ai.meta.com/resources/models-and-libraries/chameleon-downloads) | [HuggingFace](//huggingface.co/facebook/chameleon)**

This repository contains artifacts for the Meta Chameleon model from FAIR, Meta AI. In this repository is:
- [Standalone Inference Code](./chameleon/inference) — a fast inference implementation for running [model checkpoints](//ai.meta.com/resources/models-and-libraries/chameleon-downloads)
- [Input-Output Viewing](./chameleon/viewer) — a harness for richly viewing multimodal model inputs and outputs with a browser-based tool
- [Evaluation Prompts](./data) — mixed-modal and text-only prompts for human evaluation

## System Requirements

Running constituent components for inference and the input-output viewer currently requires a CUDA-capable GPU. If you'd like to run inference on other hardware, other inference implementations, including [HuggingFace](//huggingface.co/facebook/chameleon), are platform agnostic.

First, pip install this repository:
```sh
pip install git+https://github.com/fairinternal/chameleon.git
```

If you want access to the full visualizer, you'll need to clone this repository, then pip install:
```sh
git clone https://github.com/facebookresearch/chameleon.git
cd chameleon
pip install -e .
```

Model checkpoints and configs must be downloaded before running inference or the viewer. After [requesting model access](//ai.meta.com/resources/models-and-libraries/chameleon-downloads/), run the following script, adding pre-signed download URL you were emailed when prompted:
```sh
python -m chameleon.download_data [pre-signed URL]
```
(you can also paste the command given in the email containing the download link)

### Running the Viewer

The [viewer](./chameleon/viewer) visualizes multi-modal model input and output. It is most easily run with [`docker-compose`](//docs.docker.com/compose/install/). You'll need to clone the repository, not just a pip install.

The following runs both the service and viewer interface. 
> **By default, this runs the 7B parameter model. You can change the `model_path` variable in [`./config/model_viewer.yaml`](./config/model_viewer.yaml)** to select another model and alter other configuration:
```sh
docker-compose up --build
```

You can open the viewer at http://localhost:7654/

### Running the MiniViewer

The [miniviewer](./chameleon/miniviewer) is a light weight debug visualizer, that can be run with:
```sh
python -m chameleon.miniviewer
```
This runs the 7B parameter model. To run the 30B model, use the following command:
```sh
python -m chameleon.miniviewer --model-size 30b
```

You can open the miniviewer at http://localhost:5000/

### License

Use of this repository and related resources are governed by the [Chameleon Research License](//ai.meta.com/resources/models-and-libraries/chameleon-license) and the [LICENSE](./LICENSE) file.

#### Citation

To cite the paper, model, or software, please use the below:
```
@article{Chameleon_Team_Chameleon_Mixed-Modal_Early-Fusion_2024,
  author = {Chameleon Team},
  doi = {10.48550/arXiv.2405.09818},
  journal = {arXiv preprint arXiv:2405.09818},
  title = {Chameleon: Mixed-Modal Early-Fusion Foundation Models},
  url = {https://github.com/facebookresearch/chameleon},
  year = {2024}
}
```
