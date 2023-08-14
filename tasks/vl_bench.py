import copy
import datetime
import logging
import os
import os.path as osp
import time
from os.path import join
import json

from tqdm import tqdm
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from dataset import MetaLoader
from models.vindlu import VindLU
from tasks.retrieval_utils import evaluation_wrapper
from models.backbones.bert.tokenization_bert import BertTokenizer
from models.backbones.beit.builder import interpolate_pos_embed_beit
from models.criterions import get_sim
from dataset.vl_bench import setup_loader, process_path
from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed
from utils.config import Config
from utils.config_utils import setup_config, setup_evaluate_config
from utils.distributed import get_rank, is_main_process


logger = logging.getLogger(__name__)

class CustomModel(VindLU):
    def __init__(self, config=None, tokenizer=None, is_pretrain=False):
        super(CustomModel, self).__init__(
            config=config, tokenizer=tokenizer, is_pretrain=False
        )

    def forward(self, image, text, num_texts=None):
        # ================= Dual Encoder ITC loss ================ #
        self.clip_contrastive_temperature()

        vision_embeds, pooled_vision_embeds = self.encode_vision(image)
        text_embeds, pooled_text_embeds = self.encode_text(text)

        sim_v2t, sim_t2v = get_sim(  # sim_v2t: V, T
            pooled_vision_embeds,
            pooled_text_embeds,
            temp=self.temp,
        )

        # TODO: implement vtm in addition to vtc
        return sim_v2t


def setup_model(
    config, model_cls, has_decoder=False, pretrain=False, find_unused_parameters=False
):
    logger.info("Creating model")
    config = copy.deepcopy(config)

    if "bert" in config.model.text_encoder.name:
        tokenizer = BertTokenizer.from_pretrained(config.model.text_encoder.pretrained)
    else:
        raise ValueError(f"Not supported text encoder.")

    model = model_cls(config=config, tokenizer=tokenizer, is_pretrain=pretrain)
    model = model.to(torch.device(config.device))

    assert osp.isfile(config.pretrained_path)
    logger.info(f"Loading checkpoint from {config.pretrained_path}")
    checkpoint = torch.load(config.pretrained_path, map_location="cpu")
    state_dict = checkpoint["model"]

    # interpolate positional embeddings.
    if "beit" in config.model.vision_encoder.name:
        state_dict = interpolate_pos_embed_beit(state_dict, model)
    else:
        raise ValueError(
            f" vision encoder: {config.model.vision_encoder.name} not implelented"
        )

    for key in list(state_dict.keys()):
        if "bert" in key:
            encoder_key = key.replace("bert.", "")
            state_dict[encoder_key] = state_dict[key]
            if not has_decoder:
                del state_dict[key]

        # init text decoder as multimodal encoder (last 6 layers of model.text_encoder)
        # only for generation tasks like VQA
        if has_decoder and "text_encoder" in key:
            if "layer" in key:
                encoder_keys = key.split(".")
                layer_num = int(encoder_keys[4])
                if layer_num < config.model.text_encoder.fusion_layer:
                    del state_dict[key]
                    continue
                else:
                    decoder_layer_num = layer_num - 9
                    encoder_keys[4] = str(decoder_layer_num)
                    encoder_key = ".".join(encoder_keys)
            else:
                encoder_key = key
            decoder_key = encoder_key.replace("text_encoder", "text_decoder")
            state_dict[decoder_key] = state_dict[key]
            del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(msg)
    logger.info(f"Loaded checkpoint from {config.pretrained_path}")
    return model, tokenizer


def main(config):
    logger.info(f"config: \n{config}")
    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)
    cudnn.benchmark = True

    # train_loaders, test_name2loaders, train_media_types = setup_dataloaders(config, mode="ret")
    loader = setup_loader(config)

    model, tokenizer = setup_model(
        config,
        model_cls=CustomModel,
        has_decoder=False,
        pretrain=False,
        find_unused_parameters=True,
    )

    results = dict()
    for i, batch in enumerate(tqdm(loader)):
        video = batch['video'].to(device)
        text_input = tokenizer(
            batch['texts'],
            padding="max_length",
            truncation=True,
            max_length=config.max_txt_l,
            return_tensors='pt',
        ).to(device)
        with torch.cuda.amp.autocast(enabled=config.fp16):
            sim_i2t = model(video, text_input, batch['num_texts'])  # size: V, T
            sim_i2t = sim_i2t.to('cpu')
        processed = post_process_sim(
            sim_i2t,
            batch['ids'],
            batch['num_texts'],
        )
        for item_id, scores in processed:
           results[item_id] = {'scores': scores}

    ann_file = process_path(config.ann_file)
    name = osp.splitext(osp.split(ann_file)[-1])[0]
    output_dir = process_path(config.output_dir)
    _task = 'main' if not config.proficiency else 'prof'
    vtc_file = osp.join(output_dir, f'{name}-vtc-{_task}.json')
    vtm_file = osp.join(output_dir, f'{name}-vtm-{_task}.json')
    with open(vtc_file, 'w') as f:
        json.dump(results, f, sort_keys=False, indent=4)


def post_process_sim(sim_i2t, item_ids, num_texts):
    num_videos = sim_i2t.shape[0]
    num_total_texts = sum(num_texts)
    assert num_total_texts == sim_i2t.shape[1]
    offset = 0
    results = list()
    for i in range(num_videos):
        scores = sim_i2t[i, offset:offset+num_texts[i]].tolist()
        results.append((item_ids[i], scores))
        offset += num_texts[i]
    return results


if __name__ == "__main__":
    cfg = setup_config()
    main(cfg)
