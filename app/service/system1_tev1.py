"""
System1 实现：使用火山引擎 System One API 进行情绪打分
"""
import logging
from typing import Dict, Any

from app.client.doubao_client import doubao_client

logger = logging.getLogger(__name__)


# System1解析失败时的中性情绪
NEUTRAL_EMOTION = {
    "emotion": "平静",
    "intensity": 0.0,
    "impulse": "无",
    "engagement": 0.5,
    "requires_knowledge": False,
    "user_intent": "其他",
    "quick_action": "随口接话",
    "action_confidence": 0.0,
    "think_prob": 1.0,  # 失败时不走快速路径，交给System2兜底
    "needs_memory": 0.5,
    "needs_reasoning": 1.0,
    "expectation_violated": False,
}

# 直觉行为里交给System2的出口
THINK_ACTION = "得想一想"

# 情境判断：对方这句话的场景性质
SITUATION_TYPES = {
    "正常交流": "平和的陈述、询问、分享、正常讨论",
    "表达关心": "关心我的状态、需求、安危",
    "提出请求": "希望我帮忙、配合、做某事",
    "开玩笑调侃": "调侃、打趣、不正经、互损",
    "质疑批评": "对我的观点、行为、作品提出异议或不满",
    "冒犯挑衅": "贬低、否定、攻击、羞辱、为难",
    "提及共同经历": "说起我们之间发生过的事、之前的约定或承诺",
}

# 对方意图：只描述对方在干什么，不含"我"的态度
USER_INTENTS = {
    "寒暄问候": "打招呼、问在干嘛、问最近怎样",
    "闲聊": "随便聊聊天气、日常、见闻",
    "开玩笑": "调侃、损人、逗乐",
    "分享近况": "说自己的事、好消息或坏消息",
    "提问": "问一个事实、知识或看法",
    "求助": "请我帮忙做事、借东西、要技术指导",
    "征求意见": "问我该怎么办、怎么选",
    "倾诉": "诉苦、难过、需要安慰",
    "邀约": "约吃饭、约见面、约活动",
    "批评质疑": "否定我、我的作品或我的观点",
    "挑衅冒犯": "带恶意地攻击、羞辱、为难我",
    "提起往事": "说起过去发生的事、之前的约定",
    "正常讨论观点": "表达不同观点但不是在找茬，可以正常讨论",
    "其他": "以上都不是",
}

# 直觉行为：不经思考的第一反应；"得想一想"是交给System2的出口
QUICK_ACTIONS = {
    "随口接话": "顺嘴回一句，不往深了聊",
    "调侃回去": "跟着开玩笑、损回去",
    "反问对方": "把问题抛回去，问问对方",
    "敷衍应付": "不太想搭理，随便应一声",
    "关心对方": "问候、体贴、嘘寒问暖",
    "爽快答应": "直接答应下来",
    "直接拒绝": "不行就是不行",
    "顶回去": "不服，直接反驳",
    "简单回答": "知道就直接说，不知道就说不知道",
    "岔开话题": "不想接这个茬，换个话题",
    "愣住或追问": "没反应过来，或者问对方啥意思、为什么这么说",  # 🔥 新增
    "得想一想": "这事不能凭直觉，得先想清楚怎么回",  # THINK_ACTION
}


class System1Tev1:
    """使用火山引擎 System One API 的 System1 情绪反应模块"""

    def __init__(self):
        self.client = doubao_client

    async def analyze(
        self,
        brief: str,
        signal: str,
        persona: str = "",
        speaker_info: str = "",  # 🔥 说话人信息
        situation_context: str = "",  # 🔥 主体性上下文（overall_state + situation）
        recent_dialogue: str = "",  # 最近几条对话原文（不含本句）
    ) -> Dict[str, Any]:
        """
        分析外部信号的情绪反应

        Args:
            brief: 信条与事实（格式化后的字符串）
            signal: 外部信号
            persona: 人格参数（可选）
            speaker_info: 说话人信息（可选）
            situation_context: 主体性上下文（overall_state + situation，可选）
            recent_dialogue: 最近几条对话（不含本句）。"需要""好的""不用了"这类回答的意思全在上一句，
                没有它 System1 只能对着孤立的两个字猜

        Returns:
            {
                "emotion": str,         # 主情绪
                "intensity": float,     # 强度 0-1
                "impulse": str,         # 第一冲动
                "valence": float,       # 效价 -1~1
                "arousal": float,       # 唤醒度 0-1
                "dominance": float,     # 掌控感 -1~1
                "requires_knowledge": bool, # 是否需要技能召回
                "expectation_violated": bool  # 预期违背（突兀）
            }
        """
        try:
            # 构建上下文：说话人放最前、并在信号前再点一次（小模型对靠前和紧邻信号的内容权重更高）
            speaker_line = speaker_info or "未知说话人（陌生人）"
            context = f"【对方是谁】\n{speaker_line}\n\n"

            # 主体性上下文（overall_state + situation）
            if situation_context:
                context += f"{situation_context}\n\n"

            context += f"【我的人格画像】\n{persona}\n\n" if persona else ""
            context += f"【我的信条与事实】\n{brief}\n\n"
            # 刚才的对话紧挨在信号前：省略回答要结合上一句才能理解（小模型对紧邻信号的内容权重更高）
            if recent_dialogue:
                context += f"【刚才的对话】（按时间先后，越靠后越新）\n{recent_dialogue}\n\n"
            context += f"【外部信号】（说话人：{speaker_line.split('（')[0]}）\n{signal}"

            # 批量决策（使用 System One API）
            results = await self.client.decide(
                state=context,
                questions={
                    "situation": {
                        "type": "choice",
                        "instructions": "这句话的场景性质是什么？先判断场景，再判断情绪。",
                        "criteria": SITUATION_TYPES,
                    },
                    "emotion": {
                        "type": "choice",
                        "instructions": "基于刚才判断的场景（situation），结合对方是谁、我和对方的关系，此刻的主要情绪是什么？\n"
                                        "关键原则：\n"
                                        "- 先理解这句话对我的意义，再结合人格、关系和当前状态产生情绪。\n"
                                        "- '正常交流'或'表达关心'场景：通常是平静、困惑，但根据话题意义也可能担忧、喜悦、焦虑等。\n"
                                        "- '质疑批评'或'冒犯挑衅'场景：才可能是厌恶、愤怒。\n"
                                        "- 场景类型只是参考，不直接决定情绪。",
                        "criteria": {
                            # === 正面高能量 (Positive High-Arousal) ===
                            "兴奋": "充满活力、跃跃欲试",
                            "喜悦": "开心、愉快",
                            "惊喜": "意外的好消息",
                            "自豪": "对成就的满足感",
                            # === 正面低能量 (Positive Low-Arousal) ===
                            "平静": "平和、轻松",
                            "满足": "知足、舒适",
                            "信任": "信赖和安全感",
                            "感激": "感谢和珍惜",
                            # === 负面高能量 (Negative High-Arousal) ===
                            "愤怒": "生气、不满",
                            "焦虑": "担忧、不安",
                            "恐惧": "害怕、恐慌",
                            "沮丧": "受挫、失望",
                            # === 负面低能量 (Negative Low-Arousal) ===
                            "悲伤": "失落、难过",
                            "无聊": "乏味、提不起兴趣",
                            "厌恶": "反感、排斥",
                            "困惑": "迷茫、不理解"
                        }
                    },
                    "impulse": {
                        "type": "choice",
                        "instructions": "此刻的第一直觉冲动是什么？（不是理性应该做什么，而是情绪驱动的第一反应）\n"
                                        "关键原则：\n"
                                        "- 冲动由人格、刺激、情绪和关系共同决定。\n"
                                        "- 陌生人/不熟的人突然求助、聊人生、要求帮忙：第一反应是困惑、回避、追问来意，而不是热心帮助。\n"
                                        "- 熟人、亲密关系：才会有主动帮助、分享、亲近的冲动。\n"
                                        "- 有分歧不等于被冒犯。\n"
                                        "- 情绪驱动冲动：愤怒→对抗，恐惧/困惑→回避，好奇→探索，喜悦/信任→帮助/分享。",
                        "criteria": {
                            # === 积极主动 ===
                            "探索": "想要了解更多、深入研究、弄清楚（好奇驱动）",
                            "创造": "想要动手做、实现想法、改变现状（创造驱动）",
                            "分享": "想要告诉别人、传播信息、获得认同（表达驱动）",
                            "帮助": "想要支持对方、提供帮助、解决他人问题（利他驱动）",
                            "庆祝": "想要庆贺成功、享受成就、表达喜悦（仅限已完成的成功）",
                            # === 防御反应 ===
                            "回避": "想要逃避、退缩、忽略、拖延（恐惧/焦虑/困惑驱动）",
                            "对抗": "想要反击、争论、辩护、证明自己（愤怒/不满驱动）",
                            "掩饰": "想要隐藏真实感受、伪装、粉饰太平（羞愧/尴尬驱动）",
                            "僵住": "不知所措、呆住、无法反应、大脑空白（震惊/恐慌驱动）",
                            # === 社交调节 ===
                            "亲近": "想要建立联系、拉近距离、寻求支持（依附驱动）",
                            "退让": "想要妥协、让步、顺从、避免冲突（和平驱动）",
                            # === 中性 ===
                            "无": "无特别冲动、平静理性、情绪平和"
                        }
                    },
                    "intensity": {
                        "type": "score",
                        "instructions": "这个情绪有多强烈？（结合我和对方的关系：熟人玩笑话通常不会激起强烈情绪）",
                        "criteria": ["几乎没感觉", "轻微", "明显", "强烈", "极其强烈"]
                    },
                    "engagement": {
                        "type": "score",
                        "instructions": "此刻有多想跟对方多聊？（亲密关系话匣子容易打开，陌生人自然不想多说）",
                        "criteria": ["完全不想聊", "只想应付一两句", "正常交流", "愿意多聊几句", "话匣子打开了"]
                    },
                    "conflict": {
                        "type": "score",
                        "instructions": "对方的话与我的价值观、专业、在意的事有多大冲突？\n"
                                        "关键原则：\n"
                                        "- 否定我的核心价值观（如'对作品要有敬畏心''诚信为本'）：conflict=0.7+\n"
                                        "- 否定我的专业能力、技术选择、身份认同：conflict=0.6+\n"
                                        "- 观点不同但不涉及核心：conflict=0.3-0.5\n"
                                        "- 正常交流、寒暄、无关话题：conflict=0.0-0.2",
                        "criteria": ["完全无冲突", "轻微分歧", "有点不同意", "明显冲突", "触犯底线"],
                    },
                    "requires_knowledge": {
                        "type": "noul",
                        "instructions": "这个问题是否涉及专业技能、技术知识或需要专业能力才能回答？"
                                        "（包括编程、技术、古诗词鉴赏、书法、养生保健、教育方法等）"
                    },
                    "user_intent": {
                        "type": "choice",
                        "instructions": "对方说这句话，想干什么？",
                        "criteria": USER_INTENTS,
                    },
                    "quick_action": {
                        "type": "choice",
                        "instructions": "不经过思考，凭我的性格和我跟对方的关系，第一反应会怎么接这句话？"
                                        "拿不准、需要先想想的，选'得想一想'。",
                        "criteria": QUICK_ACTIONS,
                    },
                    "needs_memory": {
                        "type": "noul",
                        "instructions": "要接这句话，是否必须想起某件具体的往事、承诺或经历？"
                                        "（寒暄、玩笑、泛泛而谈的话题不需要）"
                    },
                    "needs_reasoning": {
                        "type": "noul",
                        "instructions": "这句话是否需要我停下来认真想一想才能回应？"
                                        "（涉及价值观冲突、重要决定、给建议、敏感的关系问题、被为难，才需要；"
                                        "寒暄、闲聊、玩笑、简单问答凭直觉就能接）"
                    },
                }
            )

            # 解析 System One 响应
            answers = results.get("answers", {})

            # 映射到标准格式
            # quick_action 取最可能的具体行为；"得想一想"的概率单独作为路由信号
            action_probs = dict(answers["quick_action"].get("probabilities", {}))
            think_prob = action_probs.pop(THINK_ACTION, 0.0)
            quick_action = max(action_probs, key=action_probs.get) if action_probs else answers["quick_action"]["choice"]

            emotion = {
                "situation": answers["situation"]["choice"],
                "emotion": answers["emotion"]["choice"],
                "intensity": self._map_score_to_range(answers["intensity"]["score"], 0.0, 1.0),
                "impulse": answers["impulse"]["choice"],
                "engagement": self._map_score_to_range(answers["engagement"]["score"], 0.0, 1.0),
                "conflict": self._map_score_to_range(answers["conflict"]["score"], 0.0, 1.0),
                "requires_knowledge": answers["requires_knowledge"].get("noul", 0.5) > 0.5,
                # 认知门控：供路由使用，概率保留原值，阈值在 settings 里调
                "user_intent": answers["user_intent"]["choice"],
                "quick_action": quick_action,
                "action_confidence": round(action_probs.get(quick_action, 0.0), 3),
                "think_prob": round(think_prob, 3),
                "needs_memory": round(answers["needs_memory"].get("noul", 0.5), 3),
                "needs_reasoning": round(answers["needs_reasoning"].get("noul", 0.5), 3),
            }

            logger.info(
                f"System1(SystemOne): emotion={emotion['emotion']}, "
                f"intensity={emotion['intensity']:.2f}, "
                f"engagement={emotion['engagement']:.2f}, "
                f"requires_knowledge={emotion['requires_knowledge']}, "
                f"intent={emotion['user_intent']}, action={emotion['quick_action']}"
                f"({emotion['action_confidence']:.2f}), think={emotion['think_prob']:.2f}, "
                f"mem={emotion['needs_memory']:.2f}, reason={emotion['needs_reasoning']:.2f}"
            )

            return emotion

        except Exception as e:
            logger.error(f"System1(SystemOne) failed, fallback to neutral: {e}", exc_info=True)
            return dict(NEUTRAL_EMOTION)

    @staticmethod
    def _map_score_to_range(score: float, min_val: float, max_val: float) -> float:
        """
        将 0-4 的连续分数映射到指定范围

        Args:
            score: 0-4（浮点）
            min_val: 最小值
            max_val: 最大值

        Returns:
            映射后的值
        """
        # 0 -> min_val, 2 -> (min_val+max_val)/2, 4 -> max_val
        step = (max_val - min_val) / 4.0
        return round(min_val + float(score) * step, 3)


# 全局实例
system1_tev1 = System1Tev1()
