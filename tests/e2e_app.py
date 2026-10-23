"""Test E2E giao diện Streamlit bằng Playwright (chạy thật, có screenshot).

Chạy::

    python tests/e2e_app.py

Script tự khởi động Streamlit ở cổng riêng, chạy 9 kịch bản, lưu:
- ảnh chụp UI: outputs/figures/11_ui_*.png
- kết quả:   outputs/results/e2e_ui.json
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
RES = ROOT / "outputs" / "results"
TMP = ROOT / "outputs" / "results" / "e2e_tmp"
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

checks: list[dict] = []
latencies: list[float] = []


def check(name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})
    print(("PASS " if passed else "FAIL ") + name + " — " + detail)


def wait_port(port: int, timeout: float = 90.0) -> None:
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Streamlit không mở cổng {port} sau {timeout:.0f}s")


def wait_health(timeout: float = 90.0) -> None:
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        try:
            with urllib.request.urlopen(f"{BASE}/_stcore/health", timeout=2) as response:
                if response.read().decode().strip() == "ok":
                    return
        except Exception:  # noqa: BLE001 - server chưa sẵn sàng
            time.sleep(0.5)
    raise TimeoutError("Streamlit health check thất bại")


def start_server(env_extra: dict[str, str] | None = None) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(env_extra or {})
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    log_path = TMP / "streamlit.log"
    TMP.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless=true",
            f"--server.port={PORT}",
            "--server.address=127.0.0.1",
            "--server.fileWatcherType=none",
        ],
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_port(PORT)
        wait_health()
    except Exception:
        stop_server(process)
        raise
    return process


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
    time.sleep(1.0)


def open_app(page: Page) -> None:
    page.goto(BASE, wait_until="domcontentloaded")
    expect(page.get_by_text("Phân loại biển báo giao thông Đức", exact=False).first).to_be_visible(timeout=60000)
    expect(page.get_by_text("không dùng cho quyết định giao thông thực tế", exact=False).first).to_be_visible(
        timeout=30000
    )


def upload(page: Page, path: Path) -> float:
    started = time.perf_counter()
    page.locator('input[type="file"]').first.set_input_files(str(path))
    return started


def wait_result(page: Page, timeout: float = 30000) -> None:
    page.get_by_text("Dự đoán:", exact=False).or_(
        page.get_by_text("Không xác định", exact=False)
    ).first.wait_for(timeout=timeout)


def screenshot(page: Page, name: str) -> str:
    path = FIG / f"11_ui_{name}.png"
    page.screenshot(path=str(path), full_page=True)
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    fake = TMP / "fake.jpg"
    fake.write_bytes(b"day khong phai anh")
    big = TMP / "big.jpg"
    with big.open("wb") as fh:
        fh.write(b"\xff" * (21 * 1024 * 1024))

    server = start_server()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1400, "height": 1000}, locale="vi-VN")
            page = context.new_page()
            page.set_default_timeout(30000)

            # 1) Trang chủ + footer + sidebar
            open_app(page)
            screenshot(page, "01_home")
            check(
                "home_footer_sidebar",
                page.get_by_text("Test accuracy", exact=False).count() > 0
                and page.get_by_text("không dùng cho quyết định giao thông thực tế", exact=False).count() > 0,
                "tiêu đề, sidebar (test accuracy) và footer hiển thị",
            )

            # 2) Ảnh test GTSRB → lớp 16
            started = upload(page, ROOT / "external_test/ext_01_gtsrb_test_00000.png")
            wait_result(page)
            latencies.append(time.perf_counter() - started)
            expect(page.get_by_text("Cấm xe trên 3,5 tấn", exact=False).first).to_be_visible()
            page.wait_for_timeout(400)
            bars = page.locator('[role="progressbar"]')
            top1 = page.get_by_text("1. Cấm xe trên 3,5 tấn", exact=False).count() > 0
            check("upload_anh_test", bars.count() >= 3 and top1,
                  f"top-3 hiển thị ({bars.count()} thanh), dòng '1. Cấm xe trên 3,5 tấn': {top1}")
            screenshot(page, "02_result_test")

            # 3) Biển Hàn Quốc 30 (crop) → lớp 1
            open_app(page)
            started = upload(page, ROOT / "external_test/ext_03_kr_speed30_crop.jpg")
            wait_result(page)
            latencies.append(time.perf_counter() - started)
            expect(page.get_by_text("Giới hạn tốc độ 30 km/h", exact=False).first).to_be_visible()
            check("upload_bien_han_30", True, "top-1 = Giới hạn tốc độ 30 km/h")
            screenshot(page, "03_result_kr30")

            # 4) Ảnh không có biển → "không xác định"
            open_app(page)
            upload(page, ROOT / "external_test/ext_09_road_hills.jpg")
            expect(page.get_by_text("Không xác định", exact=False).first).to_be_visible(timeout=30000)
            check("tu_choi_anh_khong_bien", True, "hiển thị cảnh báo 'Không xác định'")
            screenshot(page, "04_unknown")

            # 5) File .txt đổi đuôi .jpg → lỗi thân thiện
            open_app(page)
            upload(page, fake)
            expect(page.get_by_text("Không đọc được ảnh", exact=False).first).to_be_visible(timeout=30000)
            check("loi_file_hong", True, "báo 'Không đọc được ảnh: …'")
            screenshot(page, "05_error_fake")

            # 6) File > 20 MB → lỗi dung lượng
            open_app(page)
            upload(page, big)
            expect(page.get_by_text("quá lớn", exact=False).first).to_be_visible(timeout=30000)
            check("loi_qua_20mb", True, "báo 'Ảnh quá lớn … Giới hạn 20 MB'")
            screenshot(page, "06_error_size")

            # 7) Tab "Về mô hình" có bảng số liệu test
            open_app(page)
            tab = page.get_by_role("tab", name="Về mô hình").first
            tab.click()
            expect(tab).to_have_attribute("aria-selected", "true", timeout=30000)
            expect(page.get_by_text("Mô hình và cách đánh giá", exact=False).first).to_be_visible(timeout=30000)
            page.wait_for_timeout(400)
            body = page.inner_text("body")
            check(
                "tab_ve_mo_hinh",
                "98.56 %" in body and "0.9799" in body,
                "bảng test hiển thị 98.56 % / 0.9799",
            )
            screenshot(page, "07_about")

            # 8) Ảnh mẫu chọn từ selectbox
            open_app(page)
            page.locator('[data-testid="stSelectbox"]').first.click()
            page.get_by_role("option", name="Biển Hàn Quốc 30 km/h (đã crop)").click()
            page.wait_for_timeout(400)
            expect(page.get_by_text("Giới hạn tốc độ 30 km/h", exact=False).first).to_be_visible(timeout=30000)
            check("chon_anh_mau", True, "chọn ảnh mẫu → hiện kết quả lớp 1")
            screenshot(page, "08_sample")

            browser.close()
    finally:
        stop_server(server)

    # 9) Thiếu checkpoint → app báo lỗi rõ ràng
    missing = start_server({"GTSRB_RUN_DIR": "outputs/runs/khong_ton_tai"})
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1400, "height": 900})
            page.goto(BASE, wait_until="domcontentloaded")
            expect(page.get_by_text("Không tìm thấy checkpoint", exact=False).first).to_be_visible(timeout=60000)
            check("thieu_checkpoint", True, "báo 'Không tìm thấy checkpoint …'")
            screenshot(page, "09_missing_checkpoint")
            browser.close()
    finally:
        stop_server(missing)

    latencies.sort()
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_url": BASE,
        "browser": "chromium (playwright)",
        "latency_ui_seconds": {
            "median": round(latencies[len(latencies) // 2], 3) if latencies else None,
            "max": round(latencies[-1], 3) if latencies else None,
            "n": len(latencies),
            "scope": "từ lúc set_input_files đến khi kết quả hiển thị (gồm upload + Streamlit rerun + render)",
        },
        "screenshots": sorted(p.name for p in FIG.glob("11_ui_*.png")),
        "passed": sum(1 for c in checks if c["passed"]),
        "total": len(checks),
        "checks": checks,
    }
    (RES / "e2e_ui.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{payload['passed']}/{payload['total']} PASS | UI latency {payload['latency_ui_seconds']}")
    return 0 if payload["passed"] == payload["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
