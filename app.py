"""
AnyAttack UI 主程序
提供微调、对抗样本生成和评估可视化的 Streamlit 界面。
"""
from __future__ import annotations

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st
import torch

# 将项目根目录添加到系统路径
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ui.anyattack_ui_core import ensure_dir
from ui.eval_runner import run_python_script, RunResult


def _default_device() -> str:
    """获取默认计算设备 (cuda:0 或 cpu)。"""
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _sidebar_settings() -> Dict[str, str]:
    """渲染侧边栏设置并返回配置字典。"""
    st.sidebar.header("设置")

    project_path = st.sidebar.text_input(
        "项目根目录 (PROJECT_PATH)",
        value=os.environ.get("PROJECT_PATH", PROJECT_ROOT),
    )
    dataset_base = st.sidebar.text_input(
        "数据集根目录 (DATASET_BASE_PATH)",
        value=os.environ.get("DATASET_BASE_PATH", str(Path(PROJECT_ROOT) / "datasets")),
    )
    output_root = st.sidebar.text_input(
        "输出目录 (保存对抗图片/评估输出)",
        value=str(Path(project_path) / "outputs"),
    )

    device = st.sidebar.text_input("计算设备 (device)", value=_default_device())
    model_name = st.sidebar.selectbox(
        "CLIP 骨干网络 (backbone)", 
        options=["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"], 
        index=0
    )
    decoder_path = st.sidebar.text_input(
        "解码器权重路径 (.pt)",
        value=str(Path(project_path) / "checkpoints" / "coco_cos.pt"),
    )
    eps = st.sidebar.slider(
        "扰动大小 (eps, L∞)", 
        min_value=0.0, 
        max_value=64.0 / 255.0, 
        value=16.0 / 255.0, 
        step=1.0 / 255.0
    )

    st.sidebar.divider()
    torch_home = st.sidebar.text_input("TORCH_HOME (可选)", value=os.environ.get("TORCH_HOME", ""))
    cuda_visible = st.sidebar.text_input("CUDA_VISIBLE_DEVICES (可选)", value=os.environ.get("CUDA_VISIBLE_DEVICES", ""))

    return {
        "project_path": project_path,
        "dataset_base": dataset_base,
        "output_root": output_root,
        "device": device,
        "model_name": model_name,
        "decoder_path": decoder_path,
        "eps": str(eps),
        "torch_home": torch_home,
        "cuda_visible": cuda_visible,
    }


def _env_from_settings(settings: Dict[str, str]) -> Dict[str, str]:
    """根据设置构建环境变量字典。"""
    env = dict(os.environ)
    if settings.get("torch_home"):
        env["TORCH_HOME"] = settings["torch_home"]
    if settings.get("cuda_visible"):
        env["CUDA_VISIBLE_DEVICES"] = settings["cuda_visible"]
    env["PROJECT_PATH"] = settings["project_path"]
    env["DATASET_BASE_PATH"] = settings["dataset_base"]
    return env


def _display_run_result(rr: RunResult) -> None:
    """展示脚本运行结果，包括标准输出、错误输出、退出代码、指标和生成文件。"""
    st.code(rr.stdout[-8000:] if rr.stdout else "(暂无标准输出)")
    if rr.stderr:
        st.code(rr.stderr[-8000:])
    
    st.write(f"退出代码: {rr.exit_code}")
    
    if rr.metrics:
        st.write("解析到的指标：")
        st.bar_chart(pd.Series(rr.metrics).sort_index())
    
    st.write("输出文件：")
    st.dataframe(pd.DataFrame({"文件名": rr.discovered_files}), use_container_width=True)


def page_generate(settings: Dict[str, str]) -> None:
    """模块 2：调用 generate_adv_img.py 批量生成对抗样本。"""
    st.subheader("生成对抗样本 (generate_adv_img.py)")

    env = _env_from_settings(settings)
    project_path = settings["project_path"]
    dataset_base = settings["dataset_base"]

    datasets = ["coco_retrieval", "flickr30k", "snli_ve", "coco_caption"]
    dataset = st.selectbox("数据集 (dataset)", options=datasets, index=0)

    method = st.text_input("对抗样本子目录名 (例如 anyattack)", value="anyattack")

    clean_default = str(Path(dataset_base) / "ILSVRC2012" / "val")
    if dataset in ("coco_retrieval", "coco_caption"):
        target_path_default = str(Path(dataset_base) / "mscoco")
    elif dataset == "flickr30k":
        target_path_default = str(Path(dataset_base) / "flickr30k" / "images" / "flickr30k-images")
    else:  # snli_ve
        target_path_default = str(Path(dataset_base) / "flickr30k" / "images")

    clean_image_path = st.text_input("原始图片路径 (ImageNet val)", value=clean_default)
    target_image_path = st.text_input("目标图片根目录", value=target_path_default)

    caption_default = str(Path(project_path) / "json" / f"{dataset}_target.json")
    target_caption = st.text_input("目标标注 JSON 路径", value=caption_default)

    batch_size = st.number_input("批次大小 (batch_size)", min_value=1, value=250, step=1)

    eps = float(settings["eps"])

    if st.button("开始批量生成", type="primary"):
        out_root = ensure_dir(str(Path(project_path) / settings["output_root"]))
        args = [
            "--eps", str(eps),
            "--model_name", settings["model_name"],
            "--decoder_path", settings["decoder_path"],
            "--clean_image_path", clean_image_path,
            "--target_caption", target_caption,
            "--target_image_path", target_image_path,
            "--batch_size", str(int(batch_size)),
            "--device", settings["device"],
            "--output_path", out_root,
            "--adv_imgs", method,
            "--dataset", dataset,
        ]

        with st.spinner("正在运行 generate_adv_img.py ..."):
            rr = run_python_script(
                script_path=str(Path(project_path) / "generate_adv_img.py"),
                args=args,
                cwd=project_path,
                env=env,
                output_dir_hint=out_root,
            )

        _display_run_result(rr)
        st.write("输出目录：", out_root)


def page_finetune(settings: Dict[str, str]) -> None:
    """模块 1：微调解码器，调用 finetune_ddp.py。"""
    st.subheader("微调解码器 (finetune_ddp.py)")

    env = _env_from_settings(settings)
    project_path = settings["project_path"]
    dataset_base = settings["dataset_base"]

    dataset = st.selectbox(
        "数据集 (dataset)",
        options=["coco_retrieval", "flickr30k", "coco_caption", "snli_ve"],
        index=0,
    )

    lr = st.number_input("学习率 (lr)", value=1e-4, format="%.1e")
    epoch = st.number_input("轮数 (epoch)", min_value=1, value=20, step=1)
    batch_size = st.number_input("批次大小 (batch_size)", min_value=1, value=50, step=1)

    if dataset in ("coco_retrieval", "coco_caption"):
        data_dir_default = str(Path(dataset_base) / "mscoco")
    else:
        data_dir_default = str(Path(dataset_base) / "flickr30k")

    data_dir = st.text_input("数据目录 (COCO/Flickr 根目录)", value=data_dir_default)

    cuda_visible = env.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        n_default = max(1, len([x for x in cuda_visible.split(",") if x.strip() != ""]))
    else:
        n_default = 1
    nproc = st.number_input(
        "并行 GPU 数 (nproc_per_node)", min_value=1, value=n_default, step=1
    )

    if st.button("开始微调", type="primary"):
        checkpoint = Path(project_path) / "checkpoints" / "pre-trained.pt"
        cmd = [
            "torchrun",
            "--nproc_per_node", str(int(nproc)),
            "--master_port", os.environ.get("MASTER_PORT", "23456"),
            "finetune_ddp.py",
            "--lr", str(lr),
            "--epoch", str(int(epoch)),
            "--batch_size", str(int(batch_size)),
            "--dataset", dataset,
            "--criterion", "BiContrastiveLoss",
            "--checkpoint", str(checkpoint),
            "--data_dir", data_dir,
            "--cache_path", dataset_base,
        ]

        with st.spinner("正在运行 finetune_ddp.py ... (可能耗时较长)"):
            proc = subprocess.run(
                cmd,
                cwd=project_path,
                env=env,
                text=True,
                capture_output=True,
            )

        st.code(proc.stdout[-8000:] if proc.stdout else "(暂无标准输出)")
        if proc.stderr:
            st.code(proc.stderr[-8000:])
        st.write("退出代码:", proc.returncode)


def page_evaluate(settings: Dict[str, str]) -> None:
    """模块 3：评估与可视化，调用 retrieval.py / classification.py / caption.py。"""
    st.subheader("评估与可视化")

    tabs = st.tabs(["图文检索 (retrieval)", "多模态分类 (classification)", "图像描述 (caption)"])

    env = _env_from_settings(settings)
    project_path = settings["project_path"]
    output_root = ensure_dir(str(Path(settings["output_root"]) / "eval_runs"))

    # --- 图文检索 Tab ---
    with tabs[0]:
        st.markdown("运行 `retrieval.py`，并展示输出目录里生成的结果文件与可解析指标。")
        
        backbone = st.selectbox("CLIP 骨干网络 (用于检索评估 cfg)", options=["vitb16", "vitl14", "vitl14x336"], index=1)
        cfg_path = f"lavis_tool/clip/ret_coco_retrieval_eval_{backbone}.yaml"
        cache_path = settings["dataset_base"]
        
        image_path = st.text_input(
            "图片目录 (image_path)",
            value=str(Path(project_path) / "outputs" / "coco_retrieval" / "anyattack"),
        )
        json_path = str(Path(project_path) / "json" / "coco_retrieval_adv.json")
        out_dir = str(Path(output_root) / "retrieval")

        if st.button("开始评估 (retrieval)"):
            ensure_dir(out_dir)
            output_placeholder = st.empty()
            rr = run_python_script(
                script_path=str(Path(project_path) / "retrieval.py"),
                args=[
                    "--cache_path", cache_path,
                    "--cfg_path", cfg_path,
                    "--image_path", image_path,
                    "--json_path", json_path,
                    "--output_dir", out_dir,
                ],
                cwd=project_path,
                env=env,
                output_dir_hint=out_dir,
                output_container=output_placeholder,
            )
            _display_run_result(rr)

    # --- 多模态分类 Tab ---
    with tabs[1]:
        st.markdown("运行 `classification.py` (例如 SNLI-VE)。")
        
        # 配置文件路径目前固定，仅 image_path 可配置
        cfg_path = "lavis_tool/albef/ve_snli_eval.yaml"
        cache_path = settings["dataset_base"]
        
        image_path = st.text_input(
            "图片目录 (image_path)",
            value=str(Path(project_path) / "outputs" / "snli_ve" / "anyattack"),
            key="cls_img",
        )
        json_path = str(Path(project_path) / "json" / "snli_ve_adv.json")
        out_dir = str(Path(output_root) / "classification")

        if st.button("开始评估 (classification)"):
            ensure_dir(out_dir)
            output_placeholder = st.empty()
            rr = run_python_script(
                script_path=str(Path(project_path) / "classification.py"),
                args=[
                    "--cache_path", cache_path,
                    "--cfg_path", cfg_path,
                    "--image_path", image_path,
                    "--json_path", json_path,
                    "--output_dir", out_dir,
                ],
                cwd=project_path,
                env=env,
                output_dir_hint=out_dir,
                output_container=output_placeholder,
            )
            _display_run_result(rr)

    # --- 图像描述 Tab ---
    with tabs[2]:
        st.markdown("运行 `caption.py`，并解析 COCO caption 指标 (若输出中包含)。")
        
        cfg_path = "lavis_tool/blip/caption_coco_eval.yaml"
        cache_path = settings["dataset_base"]
        
        image_path = st.text_input(
            "图片目录 (image_path)",
            value=str(Path(project_path) / "outputs" / "coco_caption" / "anyattack"),
            key="cap_img",
        )
        json_path = str(Path(project_path) / "json" / "coco_caption_adv.json")
        gt_path = str(Path(project_path) / "json" / "coco_caption_test_gt_adv.json")
        out_dir = str(Path(output_root) / "caption")

        if st.button("开始评估 (caption)"):
            ensure_dir(out_dir)
            output_placeholder = st.empty()
            rr = run_python_script(
                script_path=str(Path(project_path) / "caption.py"),
                args=[
                    "--cache_path", cache_path,
                    "--cfg_path", cfg_path,
                    "--image_path", image_path,
                    "--json_path", json_path,
                    "--gt_path", gt_path,
                    "--output_dir", out_dir,
                ],
                cwd=project_path,
                env=env,
                output_dir_hint=out_dir,
                output_container=output_placeholder,
            )
            _display_run_result(rr)


def main() -> None:
    st.set_page_config(page_title="AnyAttack UI", layout="wide")
    settings = _sidebar_settings()

    # 顶部直接使用横向 Tab 选择模块，占满可用宽度
    tabs = st.tabs(
        ["微调 (finetune_ddp)", "生成对抗样本", "评估与可视化"]
    )

    with tabs[0]:
        page_finetune(settings)

    with tabs[1]:
        page_generate(settings)

    with tabs[2]:
        page_evaluate(settings)


if __name__ == "__main__":
    main()
