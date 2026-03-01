"""
多模态分类 (Multimodal Classification) 评估脚本
"""
from __future__ import annotations

import os
# 设置 HuggingFace 镜像，必须在导入 transformers/lavis 之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import json
import random
import warnings
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import lavis.common.utils as utils
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.registry import registry
from lavis.common.utils import now
from lavis.datasets.datasets.multimodal_classification_datasets import (
    MultimodalClassificationDataset,
)
from lavis.datasets.datasets.snli_ve_datasets import __DisplMixin
from lavis.processors.blip_processors import BlipImageBaseProcessor
from lavis.processors.clip_processors import _convert_to_rgb
from lavis.processors.randaugment import RandomAugment
from lavis.runners.runner_base import RunnerBase
from lavis_tool.multimodal_classification import MultimodalClassificationTask


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Multimodal Classification Evaluation")
    
    parser.add_argument("--cfg_path", help="配置文件路径")
    parser.add_argument("--cache_path", help="数据集缓存路径")
    parser.add_argument("--json_path", help="测试数据路径 (json)")
    parser.add_argument("--image_path", help="图片数据集路径")
    parser.add_argument("--output_dir", help="结果保存路径")
    
    parser.add_argument(
        "--options",
        nargs="+",
        help="覆盖配置文件中的部分设置，格式为 key=value",
    )

    args = parser.parse_args()
    return args


def setup_seeds(config: Config) -> None:
    """设置随机种子以保证可复现性"""
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


class BlipImageTrainProcessor(BlipImageBaseProcessor):
    """BLIP 图像训练预处理器"""
    def __init__(
        self, 
        image_size: int = 384, 
        transform: Optional[transforms.Compose] = None, 
        mean: Optional[Tuple[float]] = None, 
        std: Optional[Tuple[float]] = None, 
        min_scale: float = 0.5, 
        max_scale: float = 1.0
    ):
        super().__init__(mean=mean, std=std)
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(min_scale, max_scale),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    RandomAugment(
                        2,
                        5,
                        isPIL=True,
                        augs=[
                            "Identity",
                            "AutoContrast",
                            "Brightness",
                            "Sharpness",
                            "Equalize",
                            "ShearX",
                            "ShearY",
                            "TranslateX",
                            "TranslateY",
                            "Rotate",
                        ],
                    ),
                    transforms.ToTensor(),
                    self.normalize,
                ]
            )
        else:
            self.transform = transform

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)
        mean = cfg.get("mean", None)
        std = cfg.get("std", None)
        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )


class BlipImageEvalProcessor(BlipImageBaseProcessor):
    """BLIP 图像评估预处理器"""
    def __init__(
        self, 
        image_size: int = 384, 
        transform: Optional[transforms.Compose] = None, 
        mean: Optional[Tuple[float]] = None, 
        std: Optional[Tuple[float]] = None
    ):
        super().__init__(mean=mean, std=std)
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                    ),
                    transforms.ToTensor(),
                    self.normalize,
                ]
            )
        else:
            self.transform = transform

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)
        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(image_size=image_size, mean=mean, std=std)


class SNLIVisualEntialmentDataset(MultimodalClassificationDataset, __DisplMixin):
    """SNLI Visual Entailment 数据集"""
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.class_labels = self._build_class_labels()

    def _build_class_labels(self):
        return {"contradiction": 0, "neutral": 1, "entailment": 2}

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_id = ann["image"]
        # 处理图像路径
        if image_id.endswith(".jpg"):
            image_path = os.path.join(self.vis_root, image_id)
        else:
            image_path = os.path.join(self.vis_root, "%s" % image_id)
            
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            # 如果找不到直接路径，尝试加 .jpg 后缀
            image_path_jpg = os.path.join(self.vis_root, "%s.jpg" % image_id)
            try:
                image = Image.open(image_path_jpg).convert("RGB")
            except FileNotFoundError:
                 warnings.warn(f"找不到图像: {image_path} 或 {image_path_jpg}")
                 # 返回 None 或抛出异常，这里为了健壮性可能需要处理
                 # 但为了保持原逻辑简单，这里可能会导致后续错误
                 raise

        image = self.vis_processor(image)
        sentence = self.text_processor(ann["sentence"])

        return {
            "image": image,
            "text_input": sentence,
            "label": self.class_labels[ann["label"]],
            "image_id": image_id,
            "instance_id": ann["instance_id"],
        }


class SNLIVisualEntialmentInstructDataset(SNLIVisualEntialmentDataset, __DisplMixin):
    """SNLI Visual Entailment Instruct 数据集"""
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.classnames = ['no', 'maybe', 'yes']
        self.instruct_class_label = {"no": 0, "maybe": 1, "yes": 2}

    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data is not None:
            data["prompt"] = self.text_processor("based on the given the image is {} true?")
            data["answer"] = data["label"]
            # data["label"] = data["label"] # Redundant assignment
            data["question_id"] = data["instance_id"]
        return data


def build(cfg: Config, transform: Optional[transforms.Compose] = None) -> Dict:
    """
    构建数据集
    覆盖默认的 build 方法以支持自定义 transform 和路径
    """
    try:
        image_size = cfg.config['preprocess']['vis_processor']['eval']['image_size']
    except Exception:
        image_size = 384
        
    is_instruct = "instruct" in cfg.config.model.arch
    config = cfg.config['datasets']
    
    # 定义处理器
    vis_processors = {
        'train': BlipImageTrainProcessor(image_size=image_size, transform=transform),
        'eval': BlipImageEvalProcessor(image_size=image_size, transform=transform)
    }
    
    # 文本处理器，这里简化处理，假设都用 blip_caption
    text_processors = {
        'train': registry.get_processor_class('blip_caption').from_config({'name': 'blip_caption'}),
        'eval': registry.get_processor_class('blip_caption').from_config({'name': 'blip_caption'})
    }
    
    retrieval_datasets_keys = list(config.keys())
    dataset_key = retrieval_datasets_keys[0]
    build_info = config[dataset_key]['build_info']

    ann_info = build_info['annotations']
    data_type = config[dataset_key]['data_type']
    vis_info = build_info[data_type]

    datasets = dict()
    for split in ann_info.keys():
        if split not in ["train", "val", "test"]:
            continue

        is_train = split == "train"

        # 选择处理器
        vis_processor = vis_processors["train"] if is_train else vis_processors["eval"]
        text_processor = text_processors["train"] if is_train else text_processors["eval"]
        
        # 处理标注路径
        ann_paths = ann_info.get(split).storage
        if isinstance(ann_paths, str):
            ann_paths = [ann_paths]

        abs_ann_paths = []
        for ann_path in ann_paths:
            if not os.path.isabs(ann_path):
                ann_path = utils.get_cache_path(ann_path)
            abs_ann_paths.append(ann_path)
        ann_paths = abs_ann_paths
        
        # 检查标注文件是否存在
        if not os.path.exists(ann_paths[0]):
             warnings.warn(f"标注路径 {ann_paths[0]} 不存在。跳过 {split} 分割。")
             continue

        # 处理图像存储路径
        vis_path = vis_info.storage
        if not os.path.isabs(vis_path):
            vis_path = utils.get_cache_path(vis_path)

        if not os.path.exists(vis_path):
            warnings.warn(f"存储路径 {vis_path} 不存在。")

        # 创建数据集实例
        dataset_cls = SNLIVisualEntialmentInstructDataset if is_instruct else SNLIVisualEntialmentDataset
        datasets[split] = dataset_cls(
            vis_processor=vis_processor,
            text_processor=text_processor,
            ann_paths=ann_paths,
            vis_root=vis_path,
        )
        
    datasets_retrieval = {dataset_key: datasets}
    return datasets_retrieval


def main():
    print("[进度] 开始多模态分类任务...")
    
    args = parse_args()
    print("[进度] 正在加载配置...")
    
    # 强制将缓存根目录设置为本地目录
    local_cache_root = args.cache_path if args.cache_path else os.path.join(os.getcwd(), "cache")
    os.makedirs(local_cache_root, exist_ok=True)
    registry.mapping["paths"]["cache_root"] = local_cache_root
    
    # 同时设置 torch hub 目录
    torch.hub.set_dir(local_cache_root)
    
    job_id = now()
    cfg = Config(args)

    # 获取数据集名称，用于后续配置更新
    dataset_name = list(cfg.config['datasets'].keys())[0]

    # 根据命令行参数更新配置
    if args.image_path:
        cfg.config['datasets'][dataset_name]['build_info']['images']['storage'] = args.image_path
    if args.output_dir:
        cfg.config['run']['output_dir'] = args.output_dir
    if args.json_path:
        cfg.config['datasets'][dataset_name]['build_info']['annotations']['test']['storage'] = args.json_path

    # 动态过滤 JSON 数据集 (如果提供了 image_path 和 json_path)
    if args.image_path and args.json_path:
        print(f"[检查] 正在检查图像是否存在于 {args.image_path}...")
        try:
            # 只有当 json_path 确实存在时才尝试读取
            if os.path.exists(args.json_path):
                with open(args.json_path, 'r') as f:
                    data = json.load(f) # SNLI-VE json list of dicts

                if os.path.exists(args.image_path):
                    available_images = set(os.listdir(args.image_path))
                    
                    # SNLI-VE 格式通常包含 "image": "xxx.jpg" 或 "image": "xxx"
                    # 这里我们需要更复杂的匹配逻辑，因为 __getitem__ 里有 fallback
                    # 为了简化过滤，我们假设 filenames 匹配
                    
                    # 尝试构建一个 helper function 或直接在这里处理
                    # 考虑到 __getitem__ 里有 .jpg 后缀处理，这里过滤应该宽松一点或者尝试匹配两种情况
                    
                    # 预处理 available_images，同时保存带后缀和不带后缀的版本 (如果原图有后缀)
                    # SNLI-VE 的 image 字段通常是 "flickr30k_000000000000.jpg" 或 id
                    
                    filtered_data = []
                    for item in data:
                        img_id = item.get('image')
                        if not img_id:
                            continue
                            
                        # 检查直接匹配
                        if img_id in available_images:
                            filtered_data.append(item)
                            continue
                            
                        # 检查加 .jpg 后缀匹配 (如果 img_id 没有后缀)
                        if not img_id.endswith('.jpg') and f"{img_id}.jpg" in available_images:
                            filtered_data.append(item)
                            continue
                            
                        # 检查去 .jpg 后缀匹配 (如果 img_id 有后缀，但文件没有? 不太可能，但为了完整性)
                        # 这里主要还是看 __getitem__ 的逻辑:
                        # if image_id.endswith(".jpg"): path = ... else: path = ...
                        # fallback: try adding .jpg
                        
                    if len(filtered_data) < len(data):
                        print(f"[警告] 过滤数据集: 从 {len(data)} 条记录过滤到 {len(filtered_data)} 条，基于可用图像。")
                        
                        # 创建临时过滤后的 JSON 文件
                        output_dir = args.output_dir if args.output_dir else "output"
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True)
                        
                        temp_json_path = os.path.join(output_dir, f"temp_filtered_{os.path.basename(args.json_path)}")
                        with open(temp_json_path, 'w') as f:
                            json.dump(filtered_data, f, indent=4)
                        
                        # 更新配置以使用过滤后的 JSON
                        cfg.config['datasets'][dataset_name]['build_info']['annotations']['test']['storage'] = temp_json_path
                        print(f"[信息] 使用过滤后的数据集: {temp_json_path}")
                    else:
                        print("[信息] 数据集中的所有图像都已找到。")
                else:
                     print(f"[错误] 图像路径 {args.image_path} 不存在。")
            else:
                print(f"[警告] JSON 路径 {args.json_path} 不存在，跳过过滤。")

        except Exception as e:
            print(f"[警告] 过滤数据集失败: {e}")
            # 不中断程序，继续尝试运行

    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger()

    print("[进度] 正在设置任务...")
    task = MultimodalClassificationTask.setup_task(cfg=cfg)
    
    try:
        image_size = cfg.config['preprocess']['vis_processor']['eval']['image_size']
    except Exception:
        image_size = 384

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), 
        (0.26862954, 0.26130258, 0.27577711)
    )
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        _convert_to_rgb,
        transforms.ToTensor(),
        normalize
    ])
    
    print("[进度] 正在构建数据集...")
    datasets = build(cfg, transform=transform)

    print("[进度] 正在构建模型...")
    model = task.build_model(cfg)

    runner = RunnerBase(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )

    # 确保输出目录名称在不同系统下兼容
    output_dir = os.path.join(cfg.run_cfg["output_dir"], os.path.basename(os.path.normpath(args.image_path)) if args.image_path else job_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    registry.mapping["paths"]["output_dir"] = output_dir
    registry.mapping["paths"]["result_dir"] = output_dir

    print("[进度] 开始评估...")
    runner.evaluate(skip_reload=True)
    print("[进度] 评估完成。")


if __name__ == "__main__":
    main()
