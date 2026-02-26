from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import torch

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ui.anyattack_ui_core import (
    AnyAttackModels,
    ensure_dir,
    generate_adv_tensor,
    image_bytes_to_tensor,
    load_models,
    sanitize_filename,
    save_tensor_image,
)
from ui.eval_runner import run_python_script


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _get_models_cached(model_name: str, decoder_path: str, device_str: str) -> AnyAttackModels:
    key = (model_name, decoder_path, device_str)
    cached = st.session_state.get("models_key"), st.session_state.get("models_obj")
    if cached[0] == key and cached[1] is not None:
        return cached[1]
    device = torch.device(device_str)
    models = load_models(model_name=model_name, decoder_path=decoder_path, device=device, embed_dim=512)
    st.session_state["models_key"] = key
    st.session_state["models_obj"] = models
    return models


def _sidebar_settings() -> Dict[str, str]:
    st.sidebar.header("设置")

    project_path = st.sidebar.text_input(
        "PROJECT_PATH（项目根目录）",
        value=os.environ.get("PROJECT_PATH", PROJECT_ROOT),
    )
    dataset_base = st.sidebar.text_input(
        "DATASET_BASE_PATH（数据集根目录）",
        value=os.environ.get("DATASET_BASE_PATH", str(Path(PROJECT_ROOT) / "datasets")),
    )
    output_root = st.sidebar.text_input(
        "输出目录（保存对抗图片/评估输出）",
        value=str(Path(project_path) / "ui_outputs"),
    )

    device = st.sidebar.text_input("device", value=_default_device())
    model_name = st.sidebar.selectbox("CLIP backbone", options=["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"], index=0)
    decoder_path = st.sidebar.text_input(
        "decoder 权重路径（.pt）",
        value=str(Path(project_path) / "checkpoints" / "coco_cos.pt"),
    )
    eps = st.sidebar.slider("eps（L∞）", min_value=0.0, max_value=64.0 / 255.0, value=16.0 / 255.0, step=1.0 / 255.0)

    st.sidebar.divider()
    torch_home = st.sidebar.text_input("TORCH_HOME（可选）", value=os.environ.get("TORCH_HOME", ""))
    cuda_visible = st.sidebar.text_input("CUDA_VISIBLE_DEVICES（可选）", value=os.environ.get("CUDA_VISIBLE_DEVICES", ""))

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


def _env_from_settings(s: Dict[str, str]) -> Dict[str, str]:
    env = dict(os.environ)
    if s.get("torch_home"):
        env["TORCH_HOME"] = s["torch_home"]
    if s.get("cuda_visible"):
        env["CUDA_VISIBLE_DEVICES"] = s["cuda_visible"]
    env["PROJECT_PATH"] = s["project_path"]
    env["DATASET_BASE_PATH"] = s["dataset_base"]
    return env


def page_generate(s: Dict[str, str]) -> None:
    st.subheader("生成对抗样本")

    mode = st.radio("模式", options=["单次（1 张 clean + 1 张 target）", "批量（多张 clean + 1 张 target）"], horizontal=True)

    col1, col2 = st.columns(2)
    with col1:
        clean_files = st.file_uploader(
            "上传 clean 图片",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=(mode.startswith("批量")),
        )
    with col2:
        target_file = st.file_uploader("上传 target 图片", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=False)

    if not clean_files or not target_file:
        st.info("请先上传 clean 和 target 图片。")
        return

    out_dir = ensure_dir(str(Path(s["output_root"]) / "adv_images"))
    run_name = st.text_input("本次运行名称（用于文件夹）", value=time.strftime("run_%Y%m%d_%H%M%S"))
    out_dir_run = ensure_dir(str(Path(out_dir) / sanitize_filename(run_name)))

    if st.button("开始生成", type="primary"):
        try:
            models = _get_models_cached(s["model_name"], s["decoder_path"], s["device"])
        except Exception as e:
            st.error(f"模型加载失败：{e}")
            return

        device = torch.device(s["device"])
        eps = float(s["eps"])

        target_tensor = image_bytes_to_tensor(target_file.getvalue(), device=device)

        if not isinstance(clean_files, list):
            clean_files = [clean_files]

        prog = st.progress(0, text="生成中...")
        rows: List[Dict[str, str]] = []
        for idx, f in enumerate(clean_files):
            clean_tensor = image_bytes_to_tensor(f.getvalue(), device=device)
            adv = generate_adv_tensor(models, clean_tensor, target_tensor, eps=eps)
            out_path = str(Path(out_dir_run) / f"{idx:04d}_{sanitize_filename(f.name)}.png")
            save_tensor_image(adv[0].detach().cpu(), out_path)
            rows.append({"clean": f.name, "adv_path": out_path})
            prog.progress((idx + 1) / len(clean_files), text=f"已完成 {idx + 1}/{len(clean_files)}")

        prog.empty()
        st.success(f"已生成 {len(rows)} 张对抗图片，输出目录：{out_dir_run}")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _yaml_options(project_path: str) -> List[str]:
    base = Path(project_path) / "lavis_tool"
    if not base.exists():
        return []
    yamls = [str(p) for p in base.rglob("*.yaml")]
    return sorted(yamls)


def _safe_get_registered_arches() -> Optional[set]:
    """
    Try to read LAVIS registry to know which model.arch are available.
    Returns None if LAVIS is not importable.
    """
    try:
        from lavis.common.registry import registry  # type: ignore

        mapping = registry.mapping.get("model_name_mapping", {})
        if isinstance(mapping, dict):
            return set(mapping.keys())
        return set()
    except Exception:
        return None


def _yaml_meta(yaml_path: str) -> Dict[str, str]:
    """
    Lightweight yaml inspection without requiring a YAML parser.
    We only need:
    - model.arch
    - which dataset keyword appears (coco_retrieval / flickr30k / snli_ve / coco_caption)
    """
    arch = ""
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            txt = f.read()
    except Exception:
        return {"arch": arch, "dataset": ""}

    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("arch:"):
            arch = s.split(":", 1)[1].strip()
            break

    dataset = ""
    for k in ["coco_retrieval", "flickr30k", "snli_ve", "coco_caption", "nocaps"]:
        if k in txt:
            dataset = k
            break
    return {"arch": arch, "dataset": dataset}


def _filter_cfgs_for_task(yamls: List[str], task: str, registered_arches: Optional[set]) -> List[str]:
    """
    task: retrieval | classification | caption
    If registered_arches is provided (not None), filter out yaml whose arch is not registered.
    """
    out: List[str] = []
    for yp in yamls:
        meta = _yaml_meta(yp)
        arch = meta.get("arch", "")
        dataset = meta.get("dataset", "")

        if registered_arches is not None and arch and arch not in registered_arches:
            continue

        name = Path(yp).name.lower()
        if task == "retrieval":
            if not name.startswith("ret_"):
                continue
            if dataset and dataset not in ("coco_retrieval", "flickr30k"):
                continue
        elif task == "classification":
            if dataset and dataset != "snli_ve":
                continue
            if "ve_" not in name and "snli" not in name:
                continue
        elif task == "caption":
            if "caption" not in name:
                continue
            if dataset and dataset not in ("coco_caption", "nocaps"):
                continue
        out.append(yp)
    return out


def _cfg_selectbox(label: str, options: List[str], *, key: Optional[str] = None) -> str:
    def fmt(p: str) -> str:
        meta = _yaml_meta(p)
        # show relative path under lavis_tool for readability
        rel = Path(p).as_posix()
        if "lavis_tool/" in rel:
            rel = rel.split("lavis_tool/")[-1]
        arch = meta.get("arch", "")
        ds = meta.get("dataset", "")
        suffix = " | ".join([x for x in [ds, arch] if x])
        return f"{rel}" + (f"  ({suffix})" if suffix else "")

    if not options:
        st.warning("没有找到可用的 cfg（yaml）。请检查 `lavis_tool/` 是否存在，或当前 LAVIS 是否安装成功。")
        return st.text_input(label, value="", key=key)

    return st.selectbox(label, options=options, format_func=fmt, key=key, index=0)


def page_evaluate(s: Dict[str, str]) -> None:
    st.subheader("评估与可视化")

    tabs = st.tabs(["图文检索（retrieval）", "多模态分类（classification）", "图像描述（caption）"])

    env = _env_from_settings(s)
    project_path = s["project_path"]
    output_root = ensure_dir(str(Path(s["output_root"]) / "eval_runs"))

    yaml_all = _yaml_options(project_path)
    registered_arches = _safe_get_registered_arches()
    if registered_arches is None:
        st.info("提示：当前环境无法导入 LAVIS，UI 将显示全部 cfg；若出现“arch 未注册”，请先修复 LAVIS 安装。")

    with tabs[0]:
        st.markdown("运行 `retrieval.py`，并展示输出目录里生成的结果文件与可解析指标。")
        cfg_candidates = _filter_cfgs_for_task(yaml_all, "retrieval", registered_arches)
        cfg_path = _cfg_selectbox("cfg_path (yaml)", cfg_candidates)
        cache_path = st.text_input("cache_path（数据集缓存根目录）", value=s["dataset_base"])
        image_path = st.text_input("image_path（要评估的图片目录）", value=str(Path(project_path) / "outputs" / "coco_retrieval" / "anyattack"))
        json_path = st.text_input("json_path（对应 annotations json）", value=str(Path(project_path) / "json" / "coco_retrieval_adv.json"))
        out_dir = st.text_input("output_dir（评估输出根目录）", value=str(Path(output_root) / "retrieval"))

        if st.button("开始评估（retrieval）"):
            ensure_dir(out_dir)
            rr = run_python_script(
                script_path=str(Path(project_path) / "retrieval.py"),
                args=[
                    "--cache_path",
                    cache_path,
                    "--cfg_path",
                    cfg_path,
                    "--image_path",
                    image_path,
                    "--json_path",
                    json_path,
                    "--output_dir",
                    out_dir,
                ],
                cwd=project_path,
                env=env,
                output_dir_hint=out_dir,
            )
            st.code(rr.stdout[-8000:] if rr.stdout else "(no stdout)")
            if rr.stderr:
                st.code(rr.stderr[-8000:])
            st.write("exit_code:", rr.exit_code)
            if rr.metrics:
                st.write("解析到的指标：")
                st.bar_chart(pd.Series(rr.metrics).sort_index())
            st.write("输出文件：")
            st.dataframe(pd.DataFrame({"file": rr.discovered_files}), use_container_width=True)

    with tabs[1]:
        st.markdown("运行 `classification.py`（例如 SNLI-VE）。")
        cfg_candidates = _filter_cfgs_for_task(yaml_all, "classification", registered_arches)
        cfg_path = _cfg_selectbox("cfg_path (yaml)", cfg_candidates, key="cls_cfg")
        cache_path = st.text_input("cache_path", value=s["dataset_base"], key="cls_cache")
        image_path = st.text_input("image_path", value=str(Path(project_path) / "outputs" / "snli_ve" / "anyattack"), key="cls_img")
        json_path = st.text_input("json_path", value=str(Path(project_path) / "json" / "snli_ve_adv.json"), key="cls_json")
        out_dir = st.text_input("output_dir", value=str(Path(output_root) / "classification"), key="cls_out")

        if st.button("开始评估（classification）"):
            ensure_dir(out_dir)
            rr = run_python_script(
                script_path=str(Path(project_path) / "classification.py"),
                args=[
                    "--cache_path",
                    cache_path,
                    "--cfg_path",
                    cfg_path,
                    "--image_path",
                    image_path,
                    "--json_path",
                    json_path,
                    "--output_dir",
                    out_dir,
                ],
                cwd=project_path,
                env=env,
                output_dir_hint=out_dir,
            )
            st.code(rr.stdout[-8000:] if rr.stdout else "(no stdout)")
            if rr.stderr:
                st.code(rr.stderr[-8000:])
            st.write("exit_code:", rr.exit_code)
            if rr.metrics:
                st.write("解析到的指标：")
                st.bar_chart(pd.Series(rr.metrics).sort_index())
            st.write("输出文件：")
            st.dataframe(pd.DataFrame({"file": rr.discovered_files}), use_container_width=True)

    with tabs[2]:
        st.markdown("运行 `caption.py`，并解析 COCO caption 指标（若输出中包含）。")
        cfg_candidates = _filter_cfgs_for_task(yaml_all, "caption", registered_arches)
        cfg_path = _cfg_selectbox("cfg_path (yaml)", cfg_candidates, key="cap_cfg")
        cache_path = st.text_input("cache_path", value=s["dataset_base"], key="cap_cache")
        image_path = st.text_input("image_path", value=str(Path(project_path) / "outputs" / "coco_caption" / "anyattack"), key="cap_img")
        json_path = st.text_input("json_path", value=str(Path(project_path) / "json" / "coco_caption_adv.json"), key="cap_json")
        gt_path = st.text_input("gt_path（COCO GT json）", value=str(Path(project_path) / "json" / "coco_caption_test_gt_adv.json"), key="cap_gt")
        out_dir = st.text_input("output_dir", value=str(Path(output_root) / "caption"), key="cap_out")

        if st.button("开始评估（caption）"):
            ensure_dir(out_dir)
            rr = run_python_script(
                script_path=str(Path(project_path) / "caption.py"),
                args=[
                    "--cache_path",
                    cache_path,
                    "--cfg_path",
                    cfg_path,
                    "--image_path",
                    image_path,
                    "--json_path",
                    json_path,
                    "--gt_path",
                    gt_path,
                    "--output_dir",
                    out_dir,
                ],
                cwd=project_path,
                env=env,
                output_dir_hint=out_dir,
            )
            st.code(rr.stdout[-8000:] if rr.stdout else "(no stdout)")
            if rr.stderr:
                st.code(rr.stderr[-8000:])
            st.write("exit_code:", rr.exit_code)
            if rr.metrics:
                st.write("解析到的指标：")
                st.bar_chart(pd.Series(rr.metrics).sort_index())
            st.write("输出文件：")
            st.dataframe(pd.DataFrame({"file": rr.discovered_files}), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="AnyAttack UI", layout="wide")
    st.title("AnyAttack 本地 UI")

    s = _sidebar_settings()

    page = st.sidebar.radio("页面", options=["生成对抗样本", "评估与可视化"])
    if page == "生成对抗样本":
        page_generate(s)
    else:
        page_evaluate(s)


if __name__ == "__main__":
    main()

