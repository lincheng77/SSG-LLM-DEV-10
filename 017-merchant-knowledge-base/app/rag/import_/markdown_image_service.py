"""
Markdown 图片增强服务模块，负责图片总结、上传与引用替换。
"""
import base64
import mimetypes
import os
import re
from pathlib import Path
from typing import Dict

from langchain.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject

from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log
from app.infra.llm import llm_provider
from app.infra.object_storage import minio_gateway
from app.shared.utils.rate_limit_utils import apply_api_rate_limit





@step_log("enrich_markdown_images")
def enrich_markdown_images(state: dict) -> dict:
    # 1. 拿到md内容、md Path对象、md图片 Path对象
    md_content, md_path_obj, images_path_obj = load_markdown_and_image_dir(state)

    # 2. 判断md文档是否有需要处理的图片
    if not images_path_obj.is_dir() or len(list(images_path_obj.iterdir())) == 0:
        logger.warning("图片文件夹为空或者没有图片，无需后续处理")

    # 3. 扫描图片文件夹，获取所有图片文件
    images_context_list = scan_images(md_content, images_path_obj)
    logger.info(f"已经获取上下文信息:{images_context_list}")

    # 4. 对图片上下文中的图片进行总结 (stem 用文档名, MinIO 里每篇文档图片各归各目录)
    image_summaries_dict = summarize_images(images_context_list, md_path_obj.stem)

    # 5. 上传图片到minio并替换图片描述和引用
    new_md_content = upload_images_and_replace(images_context_list, image_summaries_dict, md_content,
                                               md_path_obj.stem)

    # 6. 保存新的MD内容
    state["md_content"] = new_md_content
    state["md_path"] = save_enhanced_markdown(new_md_content, md_path_obj)
    return state


@step_log("load_markdown_and_image_dir")
def load_markdown_and_image_dir(state: dict) -> tuple[str, Path, Path]:
    md_content = state.get("md_content", "")
    md_path = state.get("md_path")

    if not md_path:
        logger.error("md_path核心参数为空,无法继续!!")
        raise ValueError("md_path核心参数为空,无法继续!!")
    md_path_obj = Path(md_path)

    if not md_content:
        logger.warning("md_content为空,根据地址读取!")
        md_content = md_path_obj.read_text(encoding="utf-8-sig")
        state["md_content"] = md_content

    images_path_obj = md_path_obj.parent / "images"

    return md_content, md_path_obj, images_path_obj


def is_supported_image(image_name: str) -> bool:
    # 用标准库 mimetypes 判定: 后缀 -> MIME 类型, image/* 即图片
    mime, _ = mimetypes.guess_type(image_name)
    return mime is not None and mime.startswith("image/")


@step_log("scan_images")
def scan_images(md_content, images_path_obj):
    """
    Returns:
        image_context_list: 文件名、文件路径str、（前上下文、后上下文）
    """
    # 1. 图片上下文的列表
    image_context_list: list[tuple[str, str, tuple[str, str]]] = []

    # 2. 遍历图片目录下的所有文件
    for image_file in images_path_obj.iterdir():
        image_name = image_file.name

        # 2.1 过滤不支持的图片
        if not is_supported_image(image_name):
            logger.warning(f"不支持的图片格式:{image_name},跳过处理")
            continue

        # 2.2 正则匹配：在MD内容中查找是否 引用了当前图片
        rep = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        match_obj = rep.search(md_content)
        if not match_obj:
            logger.warning(f"图片:{image_name}在MD内容中未被引用,跳过处理")
            continue

        # 2.3 获取图片在MD 中的位置（起始，结束）的下标
        start, end = match_obj.span()
        pre_context = md_content[max(start - 100, 0):start]
        post_context = md_content[end:min(end + 100, len(md_content))]

        # 2.4 装入列表
        # 文件名、文件路径str、（前上下文、后上下文）
        image_context_list.append((image_name, str(image_file), (pre_context, post_context)))

    return image_context_list


@step_log("summarize_images")
def summarize_images(image_context_list: list[tuple[str, str, tuple[str, str]]], stem: str) -> Dict[str, str]:
    image_summaries_dict: Dict[str, str] = {}

    vision_model = llm_provider.vision_chat()
    for image_name, image_path_str, context in image_context_list:
        apply_api_rate_limit()

        image_context_prompt = load_prompt("image_summary", root_folder=stem, image_content=context)
        image_path_obj = Path(image_path_str)
        image_date = base64.b64encode(image_path_obj.read_bytes()).decode("utf-8")
        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mimetypes.guess_type(image_name)[0]};base64,{image_date}"
                    }
                },
                {
                    "type": "text",
                    "text": image_context_prompt
                }

            ]
        )

        chain = vision_model | StrOutputParser()
        summary = chain.invoke([message])
        image_summaries_dict[image_name] = summary

    return image_summaries_dict


@step_log("upload_images_and_replace")
def upload_images_and_replace(images_context_list, image_summaries_dict, md_content, stem):
    # 1. 获取minio客户端
    minio_client = minio_gateway.client()

    # 2. 清空该文档在minio服务器中的旧图片
    old_images = minio_client.list_objects(
        bucket_name=minio_gateway.bucket_name,
        prefix=f"{minio_gateway.image_dir[1:]}/{stem}/",  # /kb-images 去掉斜杠               # TODO 记得有机会看看存的情况
        recursive=True  # 递归遍历
    )
    delete_object_list = [DeleteObject(obj.object_name) for obj in old_images]
    errors = minio_client.remove_objects(bucket_name=minio_gateway.bucket_name, delete_object_list=delete_object_list)
    for error in errors:
        logger.warning(f"删除失败，失败原因：{error}")

    # 3. 上传所有新图片到Minio
    # TODO: 图片公网URL当前仅随替换写进md文本, 之后随切块进入向量库; 真正被消费是在查询链(当前还没有出现)
    image_url_dict: dict[str, str] = {}
    for image_name, image_path_str, _ in images_context_list:
        minio_client.fput_object(
            bucket_name=minio_gateway.bucket_name,
            object_name=f"{minio_gateway.image_dir[1:]}/{stem}/{image_name}",
            file_path=image_path_str,
            content_type=mimetypes.guess_type(image_name)[0],
        )
        image_url_dict[image_name] = minio_gateway.build_image_url(stem=stem, image_name=image_name)
    if not image_url_dict:
        logger.warning("图片上传全部失败")
        return md_content

    # 4. 替换MD内容中的图片引用
    for image_name, image_url in image_url_dict.items():
        image_summary = image_summaries_dict[image_name]

        # 替换图片的描述为大模型根据原式图片上下文进行的总结内容
        rep = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        md_content = rep.sub(lambda _: f"![{image_summary}]({image_url})", md_content)


    return md_content


@step_log("save_enhanced_markdown")
def save_enhanced_markdown(new_md_content, md_path_obj):

    new_md_path_obj = md_path_obj.with_name(f"{md_path_obj.stem}_new.md")

    new_md_path_obj.write_text(new_md_content, encoding="utf-8")


    return str(new_md_path_obj)