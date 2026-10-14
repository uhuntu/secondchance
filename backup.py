"""把题库和照片打包成一个 zip 快照。

用法：
    python backup.py                # 输出到 backups/
    python backup.py "D:/备份"      # 指定输出目录

数据库用 sqlite3 的在线备份 API 导出，服务在运行时也能拿到一致的快照；
照片直接从 uploads/ 归档。zip 里的结构就是项目结构，解压即可还原。
"""
import sqlite3
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "questions.db"
UPLOAD_DIR = BASE_DIR / "uploads"
DEFAULT_OUTPUT_DIR = BASE_DIR / "backups"


def _snapshot_db(destination: Path) -> None:
    """以只读方式打开在线库并导出一致快照，避免直接复制得到半个事务。"""
    source = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    try:
        target = sqlite3.connect(destination)
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()


def create_backup(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    has_db = DB_PATH.is_file()
    has_uploads = UPLOAD_DIR.is_dir()
    if not has_db and not has_uploads:
        raise FileNotFoundError(f"未找到 {DB_PATH} 或 {UPLOAD_DIR}，请在项目目录下运行")

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"one-more-paper-{datetime.now():%Y%m%d-%H%M%S}.zip"
    with tempfile.TemporaryDirectory() as temp:
        snapshot = Path(temp) / "questions.db"
        if has_db:
            _snapshot_db(snapshot)
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            if has_db:
                archive.write(snapshot, "questions.db")
            if has_uploads:
                for path in sorted(UPLOAD_DIR.rglob("*")):
                    if path.is_file():
                        arcname = f"uploads/{path.relative_to(UPLOAD_DIR).as_posix()}"
                        archive.write(path, arcname)
    return target


def main() -> None:
    output_dir = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR
    try:
        target = create_backup(output_dir)
    except (FileNotFoundError, sqlite3.Error) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    size_mb = target.stat().st_size / (1024 * 1024)
    print(f"已备份：{target}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
