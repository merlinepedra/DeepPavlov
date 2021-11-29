from pathlib import Path
import shutil


def initial_setup():
    init_model_path = "/data/models/ner_rus_distilbert_torch_init"
    base_model_path = "/data/models/ner_rus_distilbert_torch"
    model_filename = "/data/models/ner_rus_distilbert_torch/model.pth.tar"
    init_model_filename = "/data/models/ner_rus_distilbert_torch_init/model.pth.tar"
    init_tags_filename = "/data/models/ner_rus_distilbert_torch_init/tag.dict"

    if not Path(model_filename).exists():
        shutil.copy(init_model_filename, base_model_path)
        shutil.copy(init_tags_filename, base_model_path)
