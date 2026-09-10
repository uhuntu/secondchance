from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, Text

from database import Base


class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)
    subject = Column(String(50), nullable=False, index=True)
    knowledge_point = Column(String(200), default="")
    note = Column(Text, default="")
    image_path = Column(String(300), nullable=False)
    source_image_path = Column(String(300), default="")  # 自动切题前的整页原图
    answer_masks = Column(Text, default="")       # 练习卷答案遮挡区域（JSON）
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )
