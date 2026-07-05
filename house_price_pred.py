import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import unittest
from tqdm import tqdm
import random

# 解决matplotlib中文乱码
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ====================== 1. 数据读取类 DataReader（离线模拟数据，无网络请求） ======================
class DataReader:
    def __init__(self, file_path=None):
        self.file_path = file_path

    def load(self):
        # 优先读取本地csv
        if self.file_path is not None:
            try:
                df = pd.read_csv(self.file_path, encoding="utf-8")
                print(f"成功读取本地文件：{self.file_path}")
                return df
            except FileNotFoundError:
                print("本地housing.csv不存在，自动生成模拟房价数据集（离线无网络依赖）")
        
        # 生成离线模拟房价数据
        np.random.seed(42)
        sample_num = 2000
        MedInc = np.random.normal(3.5, 1.2, sample_num)
        HouseAge = np.random.randint(5, 50, sample_num)
        AveRooms = np.random.normal(6, 2, sample_num)
        AveBedrms = np.random.normal(1.2, 0.4, sample_num)
        Population = np.random.normal(1400, 400, sample_num)
        AveOccup = np.random.normal(3, 1, sample_num)
        Latitude = np.random.uniform(32, 42, sample_num)
        Longitude = np.random.uniform(-124, -114, sample_num)
        # 构造线性目标房价，带噪声
        median_house_value = (
            MedInc * 0.8 + HouseAge * 0.12 + AveRooms * 0.3 
            - AveBedrms * 0.2 - AveOccup * 0.15 
            + np.random.normal(0, 0.4, sample_num)
        )
        df = pd.DataFrame({
            "MedInc": MedInc,
            "HouseAge": HouseAge,
            "AveRooms": AveRooms,
            "AveBedrms": AveBedrms,
            "Population": Population,
            "AveOccup": AveOccup,
            "Latitude": Latitude,
            "Longitude": Longitude,
            "median_house_value": median_house_value
        })
        return df

# ====================== 2. 数据预处理类 DataPreprocessor ======================
class DataPreprocessor:
    @staticmethod
    def preprocess(df, target_col):
        # 缺失值填充均值
        df = df.fillna(df.mean())
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values.reshape(-1, 1)

        # 特征标准化
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        # 目标变量标准化
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y).flatten()

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test, scaler_X, scaler_y

# ====================== 3. 自定义梯度下降线性回归（学习率衰减防震荡） ======================
class LinearRegression:
    def __init__(self, learning_rate=0.1, n_iterations=1000, decay_rate=0.001):
        self.init_lr = learning_rate
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.decay_rate = decay_rate
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # tqdm训练进度条
        for epoch in tqdm(range(self.n_iter), desc="模型训练迭代"):
            # 学习率衰减
            self.lr = self.init_lr / (1 + self.decay_rate * epoch)
            y_pred = np.dot(X, self.weights) + self.bias

            # 梯度计算
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # 参数更新
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # 记录损失
            mse_loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(mse_loss)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# ====================== 4. 模型评估与可视化类 ModelEvaluator ======================
class ModelEvaluator:
    @staticmethod
    def evaluate(y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return mse, r2

    @staticmethod
    def plot_residuals(y_true, y_pred):
        residuals = y_true - y_pred
        plt.figure(figsize=(8, 5))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('预测房价（标准化）')
        plt.ylabel('残差')
        plt.title('残差分布图')
        plt.grid(True, alpha=0.3)
        plt.show(block=True)

    @staticmethod
    def plot_loss(loss_history):
        plt.figure(figsize=(8, 5))
        plt.plot(loss_history, color="#2E86AB")
        plt.xlabel('迭代次数')
        plt.ylabel('MSE损失值')
        plt.title('训练损失下降曲线')
        plt.grid(True, alpha=0.3)
        plt.show(block=True)

# ====================== 5. 结果导出类 ResultExporter ======================
class ResultExporter:
    @staticmethod
    def save_predictions(y_true, y_pred, save_path="predictions.csv"):
        df_out = pd.DataFrame({
            "真实标准化房价": y_true,
            "预测标准化房价": y_pred,
            "残差": y_true - y_pred
        })
        df_out.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"预测结果已导出至：{save_path}")

    @staticmethod
    def save_params(weights, bias, save_path="model_params.txt"):
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("=== 线性回归模型参数 ===\n")
            f.write(f"偏置项 bias = {bias:.6f}\n")
            f.write("特征权重 weights:\n")
            for idx, w in enumerate(weights):
                f.write(f"特征{idx+1}权重 = {w:.6f}\n")
        print(f"模型参数已导出至：{save_path}")

# ====================== 单元测试用例（修复参数名lr→learning_rate、n_iter→n_iterations） ======================
class TestLinearRegression(unittest.TestCase):
    def test_fit_predict(self):
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])
        # 修复：匹配构造函数参数名
        model = LinearRegression(learning_rate=0.1, n_iterations=1500)
        model.fit(X, y)
        pred = model.predict(np.array([[4]]))
        self.assertAlmostEqual(pred[0], 8, places=1)

# ====================== 主程序入口 ======================
def main():
    print("========== 基于梯度下降线性回归房价预测系统 ==========")
    # 1. 加载数据（本地文件不存在则生成模拟数据，无网络请求）
    reader = DataReader(file_path="housing.csv")
    df = reader.load()

    # 2. 数据预处理
    target_name = "median_house_value"
    X_train, X_test, y_train, y_test, scalerX, scalerY = DataPreprocessor.preprocess(df, target_name)

    # 3. 初始化并训练模型
    model = LinearRegression(learning_rate=0.1, n_iterations=1000, decay_rate=0.001)
    model.fit(X_train, y_train)

    # 4. 测试集预测
    y_pred = model.predict(X_test)

    # 5. 模型评估
    mse, r2 = ModelEvaluator.evaluate(y_test, y_pred)
    print(f"\n【模型评估指标】")
    print(f"均方误差 MSE = {mse:.4f}")
    print(f"决定系数 R² = {r2:.4f}")

    # 6. 可视化输出
    ModelEvaluator.plot_loss(model.loss_history)
    ModelEvaluator.plot_residuals(y_test, y_pred)

    # 7. 导出文件结果
    exporter = ResultExporter()
    exporter.save_predictions(y_test, y_pred, "predictions.csv")
    exporter.save_params(model.weights, model.bias, "model_params.txt")

    print("\n系统执行完毕！已生成预测文件与模型参数文件")

if __name__ == "__main__":
    main()
    # 运行单元测试
    print("\n========== 执行单元测试 ==========")
    unittest.main(argv=[''], verbosity=2, exit=False)
