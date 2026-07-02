"""使用 MinerU 精准解析 API（vlm 模型）批量将研究报告 PDF 转换为 Markdown。

支持目录结构：
- research/*.pdf → research/*.md
- regulatory/attachments/*.pdf → regulatory/attachments/*.md
- regulatory/html/*.html → 直接转为 markdown

使用方法：
    # token 已配置在项目根目录 .env，变量名 MinerU_TOKEN
    python scripts/pdf2md.py

可选参数：
    --src   原始 PDF 目录（默认 data/public_dataset_upload/raw/research）
    --dst   输出 Markdown 目录（默认 data/public_dataset_upload/md/research）
    --batch 单批次上传文件数量上限（API 限制 ≤ 50，默认 50）
    --workers 并发上传/下载线程数（默认 8）
    --cooldown 批间冷却秒数（默认 5，避免触发 429）
    --token MinerU API token（默认从 .env / 环境变量 MinerU_TOKEN 读取，
            兼容 MINERU_TOKEN）
    --overwrite 已存在的输出 markdown 也重新解析
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

import requests
from dotenv import load_dotenv

# 加载项目根目录 .env（脚本位于 scripts/ 下，故向上一级）
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
# 兼容当前工作目录下的 .env
load_dotenv()


def _read_token_from_env() -> str:
    """按优先级读取 MinerU token 环境变量。"""
    for key in ("MinerU_TOKEN", "MINERU_TOKEN", "mineru_token"):
        val = os.environ.get(key)
        if val:
            return val.strip()
    return ""

# ---------- 常量 ----------
MINERU_BASE = "https://mineru.net/api/v4"
BATCH_UPLOAD_URL = f"{MINERU_BASE}/file-urls/batch"
BATCH_RESULT_URL = f"{MINERU_BASE}/extract-results/batch"

DEFAULT_SRC = Path("data/public_dataset_upload/raw/research")
DEFAULT_DST = Path("data/public_dataset_upload/md/research")
MAX_BATCH_SIZE = 50  # MinerU 单次上传申请上限
POLL_INTERVAL = 5     # 轮询间隔（秒）— VLM 任务通常 <30s，5s 即可
POLL_TIMEOUT = 60 * 60  # 单批轮询最长时间（秒）
MAX_PAGES_PER_SPLIT = 180  # 单拆分文件最大页数（留 20 页余量避免刚好 200）
DOWNLOAD_MAX_RETRIES = 3   # zip 下载失败重试次数


@dataclass
class UploadItem:
    src_path: Path           # 原始文件完整路径
    file_name: str           # 上传时使用的文件名（用于 API）
    data_id: str             # 业务 ID，用于回查映射
    md_target: Path          # 输出 Markdown 路径
    is_split_part: bool = False  # 是否为拆分后的片段
    split_index: int = 0          # 拆分段序号
    original_path: Path | None = None  # 原始文件路径（仅拆分文件时使用）


def split_large_pdf(pdf_path: Path, tmp_dir: Path) -> list[Path]:
    """将超过页数限制的 PDF 拆分为多个小文件，返回拆分后的文件路径列表。"""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    if total_pages <= MAX_PAGES_PER_SPLIT:
        return [pdf_path]

    split_paths: list[Path] = []
    num_parts = (total_pages + MAX_PAGES_PER_SPLIT - 1) // MAX_PAGES_PER_SPLIT

    for i in range(num_parts):
        start_page = i * MAX_PAGES_PER_SPLIT
        end_page = min((i + 1) * MAX_PAGES_PER_SPLIT, total_pages)

        new_doc = fitz.open()
        for page_idx in range(start_page, end_page):
            new_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)

        split_path = tmp_dir / f"{pdf_path.stem}_part_{i + 1:02d}.pdf"
        new_doc.save(split_path)
        new_doc.close()
        split_paths.append(split_path)
        print(f"  [split] {pdf_path.name} 第 {i + 1}/{num_parts} 部分: 页 {start_page + 1}-{end_page}")

    doc.close()
    return split_paths


def merge_markdown_parts(part_paths: list[Path], output_path: Path) -> None:
    """按顺序合并多个拆分段的 Markdown 为一个完整文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, part_path in enumerate(part_paths):
            if part_path.exists():
                content = part_path.read_text(encoding="utf-8")
                if idx > 0:
                    out_f.write("\n\n")
                out_f.write(content)
            else:
                print(f"  [warning] {output_path.name} 第 {idx + 1} 部分缺失")
    print(f"  [merge] 已合并 {len(part_paths)} 部分 -> {output_path}")


class HtmlToMarkdown(HTMLParser):
    """简单 HTML 转 Markdown，用于 regulatory/html 目录文件。"""
    def __init__(self) -> None:
        super().__init__()
        self._md: list[str] = []
        self._last_was_block: bool = True

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._md.append("# ")
        elif re.match(r"h[1-6]", tag):
            level = int(tag[1])
            self._md.append("\n" + "#" * level + " ")
        elif tag == "p":
            self._md.append("\n\n")
        elif tag == "br":
            self._md.append("\n")
        elif tag == "li":
            self._md.append("\n- ")
        elif tag in ("strong", "b"):
            self._md.append("**")
        elif tag in ("em", "i"):
            self._md.append("*")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("strong", "b"):
            self._md.append("**")
        elif tag in ("em", "i"):
            self._md.append("*")

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if stripped:
            self._md.append(stripped)

    def get_markdown(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "".join(self._md)).strip()


def html_to_markdown(html_text: str) -> str:
    parser = HtmlToMarkdown()
    parser.feed(html_text)
    return parser.get_markdown()


def _make_session(token: str) -> requests.Session:
    sess = requests.Session()
    sess.headers.update(
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "Accept": "*/*",
        }
    )
    return sess


def _retry_request(fn, max_retries: int = 8, base_delay: int = 5):
    """带指数退避的请求重试，用于处理 429 限流。"""
    for attempt in range(max_retries):
        try:
            return fn()
        except requests.HTTPError as e:
            if e.response.status_code == 429:
                delay = base_delay * (2**attempt)
                print(f"[429] 限流，等待 {delay} 秒后重试 (第 {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"重试 {max_retries} 次后仍失败")


def _iter_files(src_dir: Path) -> tuple[list[Path], list[Path]]:
    """递归扫描目录，返回 (pdf_list, html_list)。"""
    pdfs: list[Path] = []
    htmls: list[Path] = []

    for root, _dirs, files in os.walk(src_dir):
        for name in files:
            path = Path(root) / name
            if path.suffix.lower() == ".pdf":
                pdfs.append(path)
            elif path.suffix.lower() == ".html":
                htmls.append(path)

    if not pdfs and not htmls:
        raise FileNotFoundError(f"未在 {src_dir} 找到 PDF 或 HTML 文件")
    return sorted(pdfs), sorted(htmls)


def _plan_items(
    files: Iterable[Path], src_root: Path, dst_root: Path, overwrite: bool
) -> list[UploadItem]:
    items: list[UploadItem] = []
    for f in files:
        rel = f.relative_to(src_root)
        target = dst_root / rel.with_suffix(".md")
        if target.exists() and not overwrite:
            try:
                rel_path = target.relative_to(Path.cwd())
            except ValueError:
                rel_path = target
            print(f"[skip] {rel} -> 已存在 {rel_path}")
            continue
        items.append(
            UploadItem(
                src_path=f,
                file_name=f.name,
                data_id=str(rel.with_suffix("")),  # 用相对路径保证唯一性
                md_target=target,
            )
        )
    return items


def _request_upload_urls(sess: requests.Session, items: list[UploadItem]) -> tuple[str, list[str]]:
    """申请批量上传链接，返回 (batch_id, urls)，带 429 重试。"""
    payload = {
        "model_version": "vlm",
        "files": [{"name": it.file_name, "data_id": it.data_id} for it in items],
    }

    def _call():
        resp = sess.post(BATCH_UPLOAD_URL, json=payload, timeout=60)
        resp.raise_for_status()
        return resp

    resp = _retry_request(_call)
    body = resp.json()
    if body.get("code") != 0:
        raise RuntimeError(f"申请上传链接失败: {body}")
    return body["data"]["batch_id"], body["data"]["file_urls"]


def _upload_files(items: list[UploadItem], urls: list[str], workers: int = 8) -> None:
    """并发上传文件到 OSS。"""
    if len(items) != len(urls):
        raise RuntimeError(f"上传链接数量({len(urls)})与文件数量({len(items)})不一致")

    def _upload_one(it: UploadItem, url: str) -> tuple[UploadItem, Exception | None]:
        try:
            with open(it.src_path, "rb") as f:
                # 注意：上传 OSS 不需要带 Authorization / Content-Type
                put = requests.put(url, data=f, timeout=600)
            if put.status_code not in (200, 201):
                return it, RuntimeError(
                    f"HTTP {put.status_code}: {put.text[:200]}"
                )
            return it, None
        except Exception as exc:  # noqa: BLE001
            return it, exc

    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_upload_one, it, url) for it, url in zip(items, urls)]
        for fut in as_completed(futures):
            it, err = fut.result()
            if err is None:
                print(f"[upload] {it.file_name} OK")
            else:
                errors.append(f"{it.file_name}: {err}")
                print(f"[upload-err] {it.file_name}: {err}")
    if errors:
        raise RuntimeError(f"上传失败 {len(errors)}/{len(items)}: {errors[:3]}")


def _poll_batch(sess: requests.Session, batch_id: str) -> dict[str, dict]:
    """轮询直到全部任务结束，返回 {data_id: extract_result_dict}，带 429 重试。"""
    url = f"{BATCH_RESULT_URL}/{batch_id}"
    start = time.time()
    done_states = {"done", "failed"}
    last_status_line = ""
    while True:

        def _call():
            resp = sess.get(url, timeout=60)
            resp.raise_for_status()
            return resp

        resp = _retry_request(_call)
        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"查询批任务失败: {body}")
        results = body["data"].get("extract_result", []) or []
        # 优先用 data_id 做 key（提交时已设置 = rel_path_stem）
        by_id: dict[str, dict] = {}
        for item in results:
            key = item.get("data_id") or Path(item.get("file_name", "")).stem
            by_id[key] = item

        states = [r.get("state", "?") for r in results]
        line = (
            f"[poll] batch={batch_id} elapsed={int(time.time()-start)}s "
            f"states={dict((s, states.count(s)) for s in sorted(set(states)))}"
        )
        if line != last_status_line:
            print(line)
            last_status_line = line

        if results and all(r.get("state") in done_states for r in results):
            return by_id

        if time.time() - start > POLL_TIMEOUT:
            raise TimeoutError(
                f"轮询超时（{POLL_TIMEOUT}s），batch_id={batch_id}，请稍后手动查询"
            )
        time.sleep(POLL_INTERVAL)


def _download_md_from_zip(zip_url: str) -> str:
    """下载解析结果 zip，提取 full.md 文本，带重试避免 IncompleteRead。"""
    last_exc: Exception | None = None
    for attempt in range(DOWNLOAD_MAX_RETRIES):
        try:
            r = requests.get(zip_url, timeout=600, stream=False)
            r.raise_for_status()
            content = r.content
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                # 优先 full.md，否则取任意 .md
                names = zf.namelist()
                target = None
                for n in names:
                    if n.endswith("full.md"):
                        target = n
                        break
                if target is None:
                    for n in names:
                        if n.lower().endswith(".md"):
                            target = n
                            break
                if target is None:
                    raise RuntimeError(f"zip 中未找到 markdown 文件: {names}")
                return zf.read(target).decode("utf-8", errors="replace")
        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            zipfile.BadZipFile,
        ) as exc:
            last_exc = exc
            wait = 2 ** attempt
            print(f"  [retry-dl] 下载失败 ({exc.__class__.__name__})，{wait}s 后重试")
            time.sleep(wait)
    raise RuntimeError(f"下载 zip 失败（{DOWNLOAD_MAX_RETRIES} 次重试后）: {last_exc}")


def _save_markdown(md_text: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(md_text, encoding="utf-8")


def process_pdf_batch(sess: requests.Session, items: list[UploadItem], workers: int = 8) -> None:
    """批量处理 PDF：申请 URL -> 并发上传 -> 轮询 -> 并发下载 -> 保存。"""
    print(f"\n== 提交批次：{len(items)} 个 PDF ==")
    batch_id, urls = _request_upload_urls(sess, items)
    print(f"[batch] id={batch_id}")
    _upload_files(items, urls, workers=workers)
    results_by_id = _poll_batch(sess, batch_id)

    # 收集 done 的任务，准备并发下载
    to_download: list[tuple[UploadItem, str]] = []
    failed: list[str] = []
    for it in items:
        info = results_by_id.get(it.data_id)
        if info is None:
            failed.append(f"{it.file_name}（未返回结果）")
            continue
        state = info.get("state")
        if state != "done":
            failed.append(f"{it.file_name}（state={state}, err={info.get('err_msg')}）")
            continue
        zip_url = info.get("full_zip_url")
        if not zip_url:
            failed.append(f"{it.file_name}（缺少 full_zip_url）")
            continue
        to_download.append((it, zip_url))

    succeeded = 0

    def _dl_one(it: UploadItem, zip_url: str) -> tuple[UploadItem, str | Exception]:
        try:
            md_text = _download_md_from_zip(zip_url)
            _save_markdown(md_text, it.md_target)
            return it, "OK"
        except Exception as exc:  # noqa: BLE001
            return it, exc

    if to_download:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_dl_one, it, url) for it, url in to_download]
            for fut in as_completed(futures):
                it, result = fut.result()
                if result == "OK":
                    print(f"[done] {it.file_name} -> {it.md_target}")
                    succeeded += 1
                else:
                    failed.append(f"{it.file_name}（下载/保存失败: {result}）")

    print(f"\n批次完成：成功 {succeeded}/{len(items)}")
    if failed:
        print("失败列表：")
        for line in failed:
            print(f"  - {line}")


def process_html_files(items: list[UploadItem]) -> None:
    """本地直接转换 HTML 文件，不需要调用 API。"""
    print(f"\n== 转换 HTML：{len(items)} 个文件 ==")
    succeeded = 0
    failed: list[str] = []
    for it in items:
        try:
            html_text = it.src_path.read_text(encoding="utf-8", errors="replace")
            md_text = html_to_markdown(html_text)
            _save_markdown(md_text, it.md_target)
            print(f"[done] {it.file_name} -> {it.md_target}")
            succeeded += 1
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{it.file_name}（转换失败: {exc}）")

    print(f"\nHTML 完成：成功 {succeeded}/{len(items)}")
    if failed:
        print("失败列表：")
        for line in failed:
            print(f"  - {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument("--batch", type=int, default=MAX_BATCH_SIZE,
                        help=f"单批次上传上限（≤{MAX_BATCH_SIZE}，默认 {MAX_BATCH_SIZE}）")
    parser.add_argument("--workers", type=int, default=8,
                        help="并发上传/下载线程数（默认 8）")
    parser.add_argument("--cooldown", type=int, default=5,
                        help="批间冷却秒数（默认 5）")
    parser.add_argument("--token", type=str, default=_read_token_from_env())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.token:
        print(
            "[error] 未提供 MinerU token：请在项目根目录 .env 中配置 MinerU_TOKEN，"
            "或通过 --token 显式传入",
            file=sys.stderr,
        )
        return 2
    if args.batch <= 0 or args.batch > MAX_BATCH_SIZE:
        print(f"[error] --batch 必须在 1..{MAX_BATCH_SIZE} 之间", file=sys.stderr)
        return 2
    if not args.src.exists():
        print(f"[error] 源目录不存在：{args.src}", file=sys.stderr)
        return 2

    pdfs, htmls = _iter_files(args.src)
    html_items = _plan_items(htmls, args.src, args.dst, args.overwrite)

    # ---- 处理 PDF：先检测超页数文件并拆分 ----
    import fitz

    # 创建临时目录存放拆分后的文件
    tmp_dir = args.dst / ".split_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 记录需要合并的拆分文件映射: original_path -> [(split_index, md_target), ...]
    merge_map: dict[Path, list[tuple[int, Path]]] = {}

    # 逐个检查 PDF，超页数的拆分后生成新的 UploadItem
    all_pdf_items: list[UploadItem] = []
    for pdf_path in pdfs:
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        doc.close()

        rel = pdf_path.relative_to(args.src)
        target = args.dst / rel.with_suffix(".md")

        if target.exists() and not args.overwrite:
            print(f"[skip] {rel} -> 已存在")
            continue

        if page_count > MAX_PAGES_PER_SPLIT:
            print(f"[split] {pdf_path.name} ({page_count} 页 > {MAX_PAGES_PER_SPLIT}) 将拆分处理")
            split_paths = split_large_pdf(pdf_path, tmp_dir)
            # 为每个拆分段创建 UploadItem
            for idx, split_path in enumerate(split_paths):
                split_target = tmp_dir / f"{split_path.stem}.md"
                all_pdf_items.append(
                    UploadItem(
                        src_path=split_path,
                        file_name=split_path.name,
                        data_id=str(split_path.stem),
                        md_target=split_target,
                        is_split_part=True,
                        split_index=idx,
                        original_path=pdf_path,
                    )
                )
            merge_map[pdf_path] = []
        else:
            all_pdf_items.append(
                UploadItem(
                    src_path=pdf_path,
                    file_name=pdf_path.name,
                    data_id=str(rel.with_suffix("")),
                    md_target=target,
                )
            )

    if not all_pdf_items and not html_items:
        print("没有需要处理的文件（全部已存在，使用 --overwrite 可强制重跑）")
        return 0

    if html_items:
        process_html_files(html_items)

    if all_pdf_items:
        sess = _make_session(args.token)
        # 切分为多个批次（每批 ≤ args.batch）
        for i in range(0, len(all_pdf_items), args.batch):
            chunk = all_pdf_items[i : i + args.batch]
            try:
                process_pdf_batch(sess, chunk, workers=args.workers)
            except Exception as exc:  # noqa: BLE001
                print(f"[batch-error] {exc}", file=sys.stderr)
                # 继续下一个批次
            # 批间冷却，避免 API 限流（最后一批跳过）
            if i + args.batch < len(all_pdf_items) and args.cooldown > 0:
                print(f"[cooldown] 批次结束，等待 {args.cooldown} 秒避免限流...")
                time.sleep(args.cooldown)

        # ---- 合并拆分后的 Markdown ----
        if merge_map:
            print("\n==== 合并拆分文件 ====")
            # 先从 all_pdf_items 收集每个原始文件对应的拆分段 md 路径
            for item in all_pdf_items:
                if item.is_split_part and item.original_path is not None:
                    if item.md_target.exists():
                        merge_map[item.original_path].append((item.split_index, item.md_target))

            for original_path, parts in merge_map.items():
                if not parts:
                    print(f"  [skip] {original_path.name} 无可用拆分结果")
                    continue

                # 按序号排序
                parts.sort(key=lambda x: x[0])
                part_paths = [p for _, p in parts]

                rel = original_path.relative_to(args.src)
                final_target = args.dst / rel.with_suffix(".md")
                merge_markdown_parts(part_paths, final_target)

            # 清理临时目录
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"[cleanup] 临时目录已清理")

    return 0


if __name__ == "__main__":
    sys.exit(main())
