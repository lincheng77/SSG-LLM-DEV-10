"""
PDF 解析服务模块，负责调用 MinerU 完成 PDF 到 Markdown 的转换。
"""
import shutil
import time
from pathlib import Path

import requests
from app.infra.document_parse.mineru_gateway import mineru_gateway
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import (
    PARSE_PDF_OUTPUT_DIR,
    MINERU_MODEL_VERSION,
    MINERU_POLL_TIMEOUT_SECONDS,
    MINERU_POLL_INTERVAL_SECONDS,
    MINERU_DOWNLOAD_TIMEOUT_SECONDS)
from app.shared.runtime.logger import step_log, logger, PROJECT_ROOT


@step_log("resolve_pdf_paths")
def resolve_pdf_paths(state: ImportGraphState) -> tuple[Path, Path]:
    # 步骤1： 获取参数 pdf_path,local_dir
    pdf_path = state.get("pdf_path")
    local_dir = state.get("local_dir")

    # 步骤2： 非空校验
    if not pdf_path:
        logger.error(f"pdf_path为空,没有文件可以解析,业务无法继续,提前终止!")
        raise ValueError(f"pdf_path为空,没有文件可以解析,业务无法继续,提前终止!")
    if not local_dir:
        local_dir = PROJECT_ROOT / PARSE_PDF_OUTPUT_DIR
        logger.warning(f"local_dir为空,为了业务继续进行,给与默认值:{str(local_dir)}")

    # 步骤3： 将字符串转为Path
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)

    # 步骤4： 存在性校验
    if not pdf_path_obj.is_file():
        logger.error(
            f"{str(pdf_path_obj)}有地址,但是没有具体的文件,业务无法继续,提前终止!!"
        )
        raise FileNotFoundError(
            f"{str(pdf_path_obj)}有地址,但是没有具体的文件,业务无法继续,提前终止!!"
        )
    if not local_dir_obj.is_dir():
        logger.warning(f"{str(local_dir_obj)}地址没有对应的文件夹,我们提前创建!!")
        local_dir_obj.mkdir(parents=True, exist_ok=True)

    # 步骤5： 返回结果
    return pdf_path_obj, local_dir_obj


@step_log("parse_pdf_to_markdown")
def parse_pdf_to_markdown(state: ImportGraphState) -> ImportGraphState:
    # 1. 获取请求参数并校验
    pdf_path_obj, local_dir_obj = resolve_pdf_paths(state)

    # 2. 上传PDF到MinerU，并轮训直到服务端返回最终压缩包地址
    zip_url = upload_pdf_and_poll(pdf_path_obj)
    logger.info(f"minerU返回的zip地址:{zip_url}")

    # 3. 下载压缩包到本地
    md_path_obj = download_zip_and_extract_md(zip_url, local_dir_obj, pdf_path_obj.stem)

    # 4. 更新state状态
    state["md_path"] = str(md_path_obj)
    state["md_content"] = md_path_obj.read_text(encoding="utf-8")
    return state


@step_log("upload_pdf_and_poll")
def upload_pdf_and_poll(pdf_path_obj: Path) -> str:
    # 1. 校验Miner服务配置
    if not mineru_gateway.api_key or not mineru_gateway.base_url:
        logger.error("minerU配置错误,请检查minerU配置!")
        raise ValueError("minerU配置错误,请检查minerU配置!")

    # 2. 构造请求参数
    url = f"{mineru_gateway.base_url}/file-urls/batch"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {mineru_gateway.api_key}",
    }
    payload = {
        "files": [{"name": pdf_path_obj.name}],
        "model_version": MINERU_MODEL_VERSION,
    }

    # 3. 请求得到pdf上传地址
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(
            f"请求minerU上传地址失败,状态码:{response.status_code},响应内容:{response.text}，请检查minerU配置是否正确")

    # 4. 解析响应内容,获取上传地址
    result = response.json()
    if result["code"] != 0:
        raise RuntimeError(f"请求minerU上传地址成功，返回错误信息,业务错误码:{result['code']},错误信息:{result['msg']}")

    upload_url = result["data"]['file_urls'][0]
    batch_id = result["data"]['batch_id']
    logger.info(f"minerU任务提交成功,batch_id:{batch_id},开始上传PDF文件:{pdf_path_obj.name}")

    # 5. 上传PDF文件到MinerU的存储地址
    with requests.Session() as session:
        session.trust_env = False  # 别用系统里自动带来的代理、证书、账号配置，requests其实会自动用session，但是这里用session是为了设置trust_env
        upload_response = session.put(upload_url, data=pdf_path_obj.read_bytes())
        if upload_response.status_code != 200:
            raise RuntimeError(
                f"上传minerU文件失败,状态码:{upload_response.status_code},响应内容:{upload_response.text}，请检查minerU配置是否正确")

    # 6. 轮训查询
    start_time = time.time()
    while True:
        # 6.1 超时判断
        if time.time() - start_time > MINERU_POLL_TIMEOUT_SECONDS:
            raise RuntimeError(f"minerU任务轮询超时, 请检查minerU配置是否正确")

        # 6.2 查询任务
        try:
            poll_response = session.get(f"{mineru_gateway.base_url}/extract-results/batch/{batch_id}", headers=headers)
        except Exception as e:
            logger.warning(f"minerU任务轮询出现异常,将会再稍后重试,异常信息:{e}")
            time.sleep(MINERU_POLL_INTERVAL_SECONDS)
            continue

        # 6.3 服务器原因
        status = poll_response.status_code
        if status // 100 == 4:
            raise RuntimeError(f"minerU任务轮询失败,客户端请求错误,状态码:{status},响应内容:{poll_response.text}")

        if status // 100 == 5:
            logger.error(f"minerU任务轮询失败,服务器故障,将会再稍后重试,状态码:{status},响应内容:{poll_response.text}")
            time.sleep(MINERU_POLL_INTERVAL_SECONDS)
            continue

        # 6.4 解析返回结果）
        poll_result = poll_response.json()
        if poll_result["code"] != 0:
            raise RuntimeError(
                f"minerU任务轮询成功，返回错误信息,业务错误码:{poll_result['code']},错误信息:{poll_result['msg']}")
        extract_result = poll_result["data"]["extract_result"][0]
        extract_result_state = extract_result["state"]

        if extract_result_state == "done":
            extract_result_zip_url = extract_result["full_zip_url"]
            if not extract_result_zip_url:
                raise RuntimeError(f"minerU任务轮询成功,但是没有返回压缩包地址,地址为空")
            return extract_result_zip_url

        if extract_result_state == "failed":
            raise RuntimeError(f"minerU任务轮询成功,但是解析任务失败，错误信息:{extract_result['err_msg']}")

        logger.info(f"minerU任务轮询成功,但是解析任务状态为:{extract_result_state},将会再稍后重试")
        time.sleep(MINERU_POLL_INTERVAL_SECONDS)


@step_log("download_zip_and_extract_md")
def download_zip_and_extract_md(zip_url: str, local_dir_obj: Path, stem: str) -> Path:
    """
      params:
        stem: DPF原始文件名(不带后缀)

    """
    # 1. 下载压缩包到本地
    response = requests.get(zip_url, timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS)
    zip_path_obj = local_dir_obj / f"{stem}_result.zip"
    zip_path_obj.write_bytes(response.content)

    # 2. 解压压缩包到本地
    extract_path_obj = local_dir_obj / stem
    if extract_path_obj.exists():
        shutil.rmtree(extract_path_obj)  # 先删除旧目录，再创建新目录（避免文件名冲突）
    extract_path_obj.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(zip_path_obj, extract_path_obj)

    # 3. 查找所有MD文件
    md_file_list = list(extract_path_obj.rglob("*.md"))
    if not md_file_list:
        raise FileNotFoundError(f"文件解压失败,在:{extract_path_obj}没有任何md文件!")

    # 4. 按优先级选择MD文件
    # 4.1 优先级1 找到和PDF 同名的MD
    for md_file in md_file_list:
        if md_file.stem == stem:
            return md_file

    # 4.2 优先级2 找到名称为"full.md"的MD（MinerU默认生成的MD文件）
    target_md_obj = None
    for md_file in md_file_list:
        if md_file.name.lower() == "full.md":
            target_md_obj = md_file
            break

    # 4.3 优先级3 找到第一个MD文件
    if not target_md_obj:
        target_md_obj = md_file_list[0]

    # 5. 重命名为统一规范名称
    new_md_obj = target_md_obj.with_name(f"{stem}.md")  # 重命名为统一规范名称（新的Path对象）
    return target_md_obj.rename(new_md_obj)  # 重命名文件（磁盘更新）
