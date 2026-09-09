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
    # AI 分析结果
    question_text = Column(Text, default="")      # 识别出的题目文字
    child_answer = Column(Text, default="")       # 孩子的作答
    error_analysis = Column(Text, default="")     # 错因分析
    correct_answer = Column(Text, default="")     # 正确答案
    explanation = Column(Text, default="")        # 讲解
    similar_questions = Column(Text, default="")  # 举一反三（JSON 数组）
    original_redo = Column(Text, default="")      # 原题重练（JSON {"题目","图"}）
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )
