from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text

from database import Base


class PageSplit(Base):
    """一页的切法（编辑器里见到的线 + 首段是否页眉）。

    切线不存库的时代，编辑器用切片高度反推边界；页眉可以丢弃之后，
    第一条线（页眉和第 1 题的界）随页眉一起消失，反推会退化。存下来
    编辑器才能原样还原上次保存的切法。
    """

    __tablename__ = "page_splits"

    source_image_path = Column(String(300), primary_key=True)
    lines = Column(Text, nullable=False, default="[]")  # JSON 数组
    skip_first = Column(Boolean, nullable=False, default=False)  # 首段是页眉，保存时丢弃


class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)
    subject = Column(String(50), nullable=False, index=True)
    knowledge_point = Column(String(200), default="")
    note = Column(Text, default="")
    image_path = Column(String(300), nullable=False)
    source_image_path = Column(String(300), default="")  # 自动切题前的整页原图
    answer_masks = Column(Text, default="")       # 练习卷答案遮挡区域（JSON）
    crop_reviewed = Column(Boolean, nullable=False, default=False)  # 已在"确认错题切割"里人工确认
    suspected_wrong = Column(Boolean, nullable=False, default=False)  # AI 判定的疑似错题（仅建议）
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )
