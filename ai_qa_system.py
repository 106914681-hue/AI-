# ====================== AI基础问答系统 - 实验三完整代码 ======================
# 功能：知识库构建、关键词提取、模糊匹配问答、用户提问日志记录、循环交互
# 标准库仅使用：string、collections，无第三方依赖
import string

# ---------------------- 1. 知识库构建（18条AI基础问答） ----------------------
# 问答知识库：key=完整问题，value=标准答案
qa_knowledge = {
    "什么是人工智能？": "人工智能是使机器模拟、延伸和扩展人类智能的技术科学，包含感知、推理、学习、决策等能力。",
    "Python在人工智能中的作用？": "Python拥有丰富AI库（TensorFlow、PyTorch、Scikit-learn），语法简洁，是机器学习、深度学习主流开发语言。",
    "机器学习和深度学习的区别？": "机器学习依靠人工提取特征；深度学习通过多层神经网络自动提取特征，适合图像、文本、语音等高维数据。",
    "什么是监督学习？": "监督学习使用带标注标签的数据集训练模型，目标是学习输入到标签的映射关系，如分类、回归任务。",
    "什么是无监督学习？": "无监督学习无人工标签，依靠数据自身分布规律聚类、降维，典型算法K-Means、PCA。",
    "什么是强化学习？": "强化学习通过智能体与环境交互，依靠奖励值迭代优化策略，多用于游戏AI、机器人控制。",
    "神经网络的基本组成单元是什么？": "基础单元为神经元（感知器），包含输入、权重、激活函数、输出四部分。",
    "什么是过拟合？": "模型在训练集表现极好，测试集效果差，过度记住训练数据噪声，泛化能力弱。",
    "如何解决过拟合？": "常用方法：增加数据集、正则化、Dropout、早停、降低模型复杂度。",
    "梯度下降是什么？": "梯度下降是优化算法，沿损失函数梯度反向更新参数，最小化模型预测误差。",
    "什么是大语言模型？": "基于Transformer架构、海量文本数据训练的深度学习模型，具备文本理解、生成、问答、翻译能力。",
    "训练AI模型需要哪些数据？": "根据任务区分，分类任务需要标注样本，大模型需要海量无标注文本，图像任务需要图片标签数据集。",
    "什么是卷积神经网络CNN？": "专门处理网格结构数据（图像），依靠卷积核提取局部空间特征，减少参数量。",
    "RNN循环神经网络适用场景？": "处理时序序列数据，如文本、语音、时间序列预测，存在梯度消失缺陷。",
    "Transformer模型核心机制？": "自注意力机制，可并行计算全局序列依赖，是大语言模型、视觉大模型基础架构。",
    "什么是特征工程？": "原始数据清洗、转换、提取有效特征，提升机器学习模型训练效果。",
    "准确率Accuracy含义？": "模型预测正确样本占全部样本的比例，适合样本均衡分类任务。",
    "人工智能的应用领域有哪些？": "计算机视觉、自然语言处理、自动驾驶、智能推荐、医疗诊断、工业质检、语音识别等。"
}

# 全局存储：所有问题关键词集合、关键词-问题索引映射、用户提问日志
all_keywords_set = set()
keyword_index = dict()  # key:关键词, value:对应问题列表
user_question_log = []  # 存储用户所有提问记录

# 停用词（过滤无意义词汇，不参与匹配）
stop_words = {"什么", "是", "的", "和", "有哪些", "如何", "怎么", "在", "中", "作用", "区别", "需要", "哪些"}

# 文本预处理：去除标点、分词、过滤停用词，返回关键词集合
def extract_keywords(text: str) -> set:
    # 去除标点符号
    punc = set(string.punctuation + "，。、；：？！‘’“”（）")
    clean_text = "".join([c for c in text if c not in punc])
    # 简单中文分词（按中文短句拆分）
    words = clean_text.split()
    temp_keywords = []
    for word in words:
        # 简单单字/词组拆分适配中文场景
        seg_words = [w for w in word if len(w) > 1 or w not in stop_words]
        temp_keywords.extend(seg_words)
    # 过滤停用词，去重返回集合
    keywords = set()
    for w in temp_keywords:
        if w not in stop_words and len(w.strip()) > 0:
            keywords.add(w)
    return keywords

# 初始化知识库关键词、关键词索引
def init_knowledge_index():
    global all_keywords_set
    for question in qa_knowledge:
        q_keys = extract_keywords(question)
        all_keywords_set.update(q_keys)
        # 构建关键词-问题反向索引
        for kw in q_keys:
            if kw not in keyword_index:
                keyword_index[kw] = []
            if question not in keyword_index[kw]:
                keyword_index[kw].append(question)

# ---------------------- 2. 核心匹配算法：关键词交集相似度匹配 ----------------------
def match_question(user_input: str) -> str:
    user_keys = extract_keywords(user_input)
    if len(user_keys) == 0:
        return "输入内容无有效关键词，请重新提问！"
    max_intersect = -1
    best_match_q = None
    # 遍历全部知识库问题，计算交集关键词数量
    for q in qa_knowledge:
        q_keys = extract_keywords(q)
        intersect = len(user_keys & q_keys)
        if intersect > max_intersect:
            max_intersect = intersect
            best_match_q = q
    # 交集为0代表无匹配
    if max_intersect <= 0:
        return "抱歉，未找到相关答案，请尝试其他问题"
    return qa_knowledge[best_match_q]

# ---------------------- 3. 人机交互主循环模块 ----------------------
def run_qa_system():
    print("===== AI基础智能问答系统 V1.0 =====")
    print("输入你的AI相关问题进行提问，输入【退出】结束程序\n")
    init_knowledge_index()  # 初始化知识库索引
    while True:
        user_input = input("请输入问题：").strip()
        # 退出逻辑
        if user_input == "退出":
            print("\n===== 问答会话结束 =====")
            print(f"本次会话共提问 {len(user_question_log)} 次，所有提问记录：")
            for idx, q in enumerate(user_question_log, 1):
                print(f"{idx}. {q}")
            print("系统已退出")
            break
        # 记录用户提问日志
        user_question_log.append(user_input)
        # 匹配答案并输出
        ans = match_question(user_input)
        print(f"系统回答：{ans}\n")

# 程序入口
if __name__ == "__main__":
    run_qa_system()
