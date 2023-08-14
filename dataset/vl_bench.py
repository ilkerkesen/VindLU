import decord
import torch
import numpy as np
from copy import deepcopy
import logging
import json
import os.path as osp
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from dataset.base_dataset import ImageVideoBaseDataset
from dataset.video_utils import get_frame_indices


logger = logging.getLogger(__name__)


def process_path(path):
    return osp.abspath(osp.expanduser(path))


def read_video(
    video_path,
    start_pts=0,
    end_pts=None,
    pts_unit='pts',
    num_frames=4,
    sample='middle',
    fix_start=None,
    max_num_frames=-1,
):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    fps = video_reader.get_avg_fps()
    vlen = len(video_reader)
    is_full_video = end_pts is None or end_pts == -1
    if pts_unit == 'sec' and not is_full_video:
        ts = video_reader.get_frame_timestamp(np.arange(vlen))
        ts = ts[:, 0]
        start_pts = np.abs(ts-start_pts).argmin()
        end_pts = np.abs(ts-end_pts).argmin()
        vlen = end_pts-start_pts+1
    elif pts_unit == 'sec' and is_full_video:
        pass
    elif pts_unit == 'pts':
        vlen = end_pts-start_pts+1

    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames
    )

    if not is_full_video:
        frame_indices = [i+start_pts for i in frame_indices]
    frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames, frame_indices


class VLBenchDataset(ImageVideoBaseDataset):
    media_type = 'video'

    def __init__(
        self,
        ann_file,
        transform,
        num_frames=4,
        sample_type="rand",
        proficiency=False,
        quva_dir=None,
        something_something_dir=None,
        youtube_dir=None,
    ):
        super().__init__()
        self.ann_file = process_path(ann_file)
        self.transform = transform
        self.num_frames = num_frames
        self.sample_type = sample_type
        self.proficiency = proficiency
        self.load_annotations()        

        # video dirs
        self.quva_dir = None
        self.something_something_dir = None
        self.youtube_dir = None

        if quva_dir is not None:
            quva_dir = process_path(quva_dir)
            assert osp.isdir(quva_dir)
            self.quva_dir = quva_dir

        if something_something_dir is not None:
            video_dir = process_path(something_something_dir)
            assert osp.isdir(video_dir)
            self.something_something_dir = video_dir

        if youtube_dir is not None:
            youtube_dir = process_path(youtube_dir)
            assert osp.isdir(youtube_dir)
            self.youtube_dir = youtube_dir

    def load_annotations(self):
        with open(self.ann_file, 'r') as f:
            raw = json.load(f)

        self.anno_list = list()
        for key, value in raw.items():
            item = deepcopy(value)
            item['key'] = key
            self.anno_list.append(item)

    # TODO: implement this
    # -- I can check what I read using the `read_frames_decord` function.
    def load_and_transform_media_data_video(self, index):
        item = self.anno_list[index]
        dataset = item['dataset']
        video_file = item['video_file']
        video_path = None
        if dataset == 'QUVA':
            normalized = item.get('normalized')
            assert normalized
            video_dir = osp.join(self.quva_dir, 'normalized_videos')
            video_path = osp.join(video_dir, video_file)
        elif dataset == 'something-something-v2':
            video_dir = self.something_something_dir
            video_path = osp.join(video_dir, f'{item["dataset_idx"]}.webm')
        elif dataset == 'RareAct' or dataset == 'VidSitu':
            video_dir = self.youtube_dir
            video_path = osp.join(video_dir, f'{item["youtube_id"]}.mp4')
        else:
            raise NotImplementedError('Not implemented yet.')

        start_pts = item.get('start_time')
        end_pts = item.get('end_time', -1)
        end_pts = end_pts if end_pts != -1 else None

        if item['time_unit'] == 'pts':  # otherwise it returns single frame
            video = read_video(
                video_path,
                start_pts=start_pts,
                end_pts=end_pts,
                pts_unit='pts',
                num_frames=self.num_frames,
            )[0]
        elif item['time_unit'] == 'sec':
            start_pts = float(start_pts) if start_pts is not None else 0.0
            end_pts = float(end_pts) if end_pts is not None else None
            video = read_video(
                video_path,
                start_pts=start_pts,
                end_pts=end_pts,
                pts_unit='sec',
                num_frames=self.num_frames,
            )[0]
        return video                

    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, index):
        item = self.anno_list[index]
        subitem = item if not self.proficiency else item['proficiency']
        video = self.load_and_transform_media_data(index)
        if self.transform is not None:
            video = self.transform(video)
        texts = [subitem['caption']] + subitem['foils']
        return {
            'index': index,
            'item_id': item['key'],
            'video': video,
            'texts': texts,
        }


def create_dataset(config):
    vision_enc_name = config.model.vision_encoder.name
    if "swin" in vision_enc_name:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif "beit" in vision_enc_name:
        mean = (0.5, 0.5, 0.5)  # for all beit model except IN1K finetuning
        std = (0.5, 0.5, 0.5)
    elif "clip" in vision_enc_name:
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    else:
        raise ValueError

    normalize = transforms.Normalize(mean, std)

    type_transform = transforms.Lambda(lambda x: x.float().div(255.))
    transform = transforms.Compose(
        [
            transforms.Resize(
                (config.inputs.image_res, config.inputs.image_res),
                interpolation=InterpolationMode.BICUBIC,
            ),
            type_transform,
            normalize,
        ]
    )

    dataset = VLBenchDataset(
        ann_file=config.ann_file,
        transform=transform,
        num_frames=config.inputs.video_input.num_frames_test,
        sample_type=config.inputs.video_input.sample_type_test,
        quva_dir=config.quva_dir,
        something_something_dir=config.something_something_dir,
        youtube_dir=config.youtube_dir,
        proficiency=config.proficiency,
    )
    return dataset


def create_loader(dataset, batch_size, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
    )


def setup_loader(config):
    logger.info('Creating the dataset/loader.')
    dataset = create_dataset(config)
    loader = create_loader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    return loader


def custom_collate_fn(batch):
    item_ids = [item['item_id'] for item in batch]
    num_texts = [len(item['texts']) for item in batch]
    texts = []
    for item in batch:
        texts.extend(item['texts'])
    video = torch.cat([item['video'].unsqueeze(0) for item in batch], dim=0)
    return {
        'ids': item_ids,
        'video': video,
        'texts': texts,
        'num_texts': num_texts,
    } 