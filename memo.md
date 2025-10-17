Kaggleコンペの具体的な流れ
Kaggleコンペティションに参加する際の全体の流れを、初心者向けにステップバイステップで説明します。
📋 コンペ参加の全体フロー
Phase 1: コンペ選択・参加登録
Phase 2: データ理解・探索
Phase 3: ベースライン構築
Phase 4: 特徴量エンジニアリング
Phase 5: モデル改善
Phase 6: アンサンブル・最終調整
Phase 7: 提出・リーダーボード確認
Phase 1: コンペ選択・参加登録 🎯
1-1. コンペを探す
Kaggle Competitions にアクセス
初心者向けなら「Getting Started」カテゴリーがおすすめ
例: Titanic, House Prices, Digit Recognizer
1-2. コンペに参加
コンペページで「Join Competition」ボタンをクリック
ルール（Rules）を確認・同意
1-3. 重要情報の確認
Evaluation: 評価指標（RMSE、AUC、Accuracyなど）
Timeline: 締め切り
Prize: 賞金（あれば）
Data: データの説明
Phase 2: データ理解・探索 🔍
2-1. データのダウンロード
方法1: Web UIから
Kaggleサイト → Data タブ → Download All
方法2: Kaggle API（推奨）
# コンペ名を確認（URLから）
# 例: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
# → コンペ名は "house-prices-advanced-regression-techniques"

uv run kaggle competitions download -c house-prices-advanced-regression-techniques
unzip house-prices-advanced-regression-techniques.zip -d data/raw/
2-2. データの確認
ls data/raw/
# 通常の構成例:
# train.csv        - 訓練データ（正解ラベル付き）
# test.csv         - テストデータ（予測対象）
# sample_submission.csv - 提出フォーマットの例
# data_description.txt  - データの説明
2-3. EDA (Exploratory Data Analysis)
notebooks/01_eda.ipynb で実施：
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# データ読み込み
train = pd.read_csv('../data/raw/train.csv')
test = pd.read_csv('../data/raw/test.csv')

# 基本情報
print(train.shape)  # (行数, 列数)
train.head()
train.info()        # データ型、欠損値
train.describe()    # 統計量

# 目的変数の分布
plt.hist(train['target'])
plt.title('Target Distribution')

# 欠損値の確認
train.isnull().sum()

# 相関分析
correlation = train.corr()
sns.heatmap(correlation, annot=True)

# 特徴量と目的変数の関係
sns.scatterplot(x='feature1', y='target', data=train)
👀 EDAで確認すること
✅ データサイズ（行数・列数）
✅ 欠損値の有無・パターン
✅ 目的変数の分布（偏りはないか）
✅ 特徴量の型（数値・カテゴリ）
✅ 外れ値の有無
✅ 特徴量間の相関
✅ train/testのデータ分布の違い
Phase 3: ベースライン構築 🚀
3-1. シンプルなモデルで最初の提出
notebooks/02_baseline.ipynb:
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

# データ読み込み
train = pd.read_csv('../data/raw/train.csv')
test = pd.read_csv('../data/raw/test.csv')

# シンプルな前処理
# 数値列のみ使用、欠損値は中央値で埋める
numeric_features = train.select_dtypes(include=[np.number]).columns
numeric_features = numeric_features.drop('target')  # 目的変数を除外

X = train[numeric_features].fillna(train[numeric_features].median())
y = train['target']
X_test = test[numeric_features].fillna(train[numeric_features].median())

# 訓練・検証分割
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# モデル訓練
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 検証スコア
y_pred = model.predict(X_valid)
score = mean_squared_error(y_valid, y_pred, squared=False)  # RMSE
print(f'Validation RMSE: {score}')

# テストデータで予測
predictions = model.predict(X_test)

# 提出ファイル作成
submission = pd.DataFrame({
    'id': test['id'],
    'target': predictions
})
submission.to_csv('../submissions/baseline_submission.csv', index=False)
3-2. 初回提出
# Web UIから submissions/baseline_submission.csv をアップロード
# または
uv run kaggle competitions submit \
    -c house-prices-advanced-regression-techniques \
    -f submissions/baseline_submission.csv \
    -m "Baseline: Random Forest with numeric features only"
3-3. リーダーボードの確認
Public Leaderboard: テストデータの一部で評価（コンペ期間中に見える）
Private Leaderboard: テストデータの残り部分で評価（終了後に公開）
Phase 4: 特徴量エンジニアリング 🛠️
4-1. 特徴量の作成・改善
notebooks/03_feature_engineering.ipynb で実験：
# 欠損値の扱いを改善
train['feature1_is_missing'] = train['feature1'].isnull().astype(int)

# カテゴリ変数のエンコーディング
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['category_encoded'] = le.fit_transform(train['category'])

# 新しい特徴量の作成
train['feature_interaction'] = train['feature1'] * train['feature2']
train['feature_ratio'] = train['feature1'] / (train['feature2'] + 1)

# 対数変換（分布が偏っている場合）
train['feature_log'] = np.log1p(train['feature1'])

# 日付から特徴量抽出
train['year'] = pd.to_datetime(train['date']).dt.year
train['month'] = pd.to_datetime(train['date']).dt.month
train['day_of_week'] = pd.to_datetime(train['date']).dt.dayofweek

# 集約特徴量
train['category_mean_target'] = train.groupby('category')['target'].transform('mean')
4-2. 関数化して再利用可能に
良い特徴量が見つかったら src/features/feature_engineering.py に移す：
def create_features(df):
    """特徴量を作成"""
    df = df.copy()
    
    # 欠損フラグ
    df['feature1_is_missing'] = df['feature1'].isnull().astype(int)
    
    # 相互作用
    df['feature_interaction'] = df['feature1'] * df['feature2']
    
    # 対数変換
    df['feature_log'] = np.log1p(df['feature1'])
    
    return df
Phase 5: モデル改善 📈
5-1. より強力なモデルを試す
# LightGBM（勾配ブースティング）
import lightgbm as lgb

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
}

train_data = lgb.Dataset(X_train, y_train)
valid_data = lgb.Dataset(X_valid, y_valid, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(50)]
)
5-2. クロスバリデーション
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_valid_fold = X.iloc[valid_idx]
    y_valid_fold = y.iloc[valid_idx]
    
    model.fit(X_train_fold, y_train_fold)
    pred = model.predict(X_valid_fold)
    score = mean_squared_error(y_valid_fold, pred, squared=False)
    scores.append(score)
    print(f'Fold {fold+1}: {score}')

print(f'Average CV Score: {np.mean(scores)}')
5-3. ハイパーパラメータチューニング
# Optuna（自動チューニング）
import optuna

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
    }
    
    model = lgb.train(params, train_data, num_boost_round=100)
    pred = model.predict(X_valid)
    return mean_squared_error(y_valid, pred, squared=False)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
print(f'Best params: {study.best_params}')
Phase 6: アンサンブル・最終調整 🎯
6-1. 複数モデルのアンサンブル
# 複数モデルの予測を平均
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb

# モデル1: LightGBM
model1 = lgb.train(params1, train_data)
pred1 = model1.predict(X_test)

# モデル2: Random Forest
model2 = RandomForestRegressor()
model2.fit(X_train, y_train)
pred2 = model2.predict(X_test)

# モデル3: XGBoost
import xgboost as xgb
model3 = xgb.XGBRegressor()
model3.fit(X_train, y_train)
pred3 = model3.predict(X_test)

# 加重平均アンサンブル
final_pred = 0.5 * pred1 + 0.3 * pred2 + 0.2 * pred3
6-2. スタッキング
from sklearn.linear_model import Ridge

# Level 1: 複数モデルの予測
# Level 2: それらを入力として新しいモデルで学習
meta_features = np.column_stack([pred1, pred2, pred3])
meta_model = Ridge()
meta_model.fit(meta_features_train, y_train)
final_pred = meta_model.predict(meta_features_test)
Phase 7: 提出・リーダーボード確認 📊
7-1. 複数回提出
Kaggleでは1日の提出回数に制限があります（コンペによる）：
例: 1日5回まで
戦略：
初期: 色々試して頻繁に提出
中期: CV（クロスバリデーション）スコアを信頼して提出を絞る
終盤: 最も信頼できるモデル2つを選んで提出
7-2. Public vs Private LB
Public LB (30%):  コンペ期間中に見えるスコア
Private LB (70%): コンペ終了後に判明する最終順位
注意: Public LBに過学習しない！
CVスコアとPublic LBスコアの相関を確認
CVスコアを信頼する
7-3. 最終提出の選択
コンペ終了前に2つの提出を選択：
最もCVスコアが良いもの
最もPublic LBスコアが良いもの（異なる場合）
🗓️ タイムライン例（3ヶ月コンペの場合）
期間	フェーズ	やること
Week 1	データ理解	EDA、ベースライン提出
Week 2-4	特徴量開発	様々な特徴量を試す
Week 5-8	モデル改善	複数モデル、パラメータ調整
Week 9-11	アンサンブル	モデルの組み合わせ
Week 12	最終調整	コード整理、最終提出選択
💡 初心者向けTips
✅ やるべきこと
まず提出する: 完璧を目指さず、ベースラインをすぐ提出
CVを信頼: クロスバリデーションスコアを重視
Discussion/Notebookを見る: 他の参加者のアイデアを学ぶ
小さな改善を積み重ねる: 一気に完璧を目指さない
❌ 避けるべきこと
過度な複雑化: 初期から複雑なモデルを使わない
Public LB過学習: Public LBだけを追いかけない
1つのモデルに固執: 複数のアプローチを試す
締め切り直前の提出: 余裕を持って提出
📚 おすすめの初心者向けコンペ
Titanic - 分類問題の入門
House Prices - 回帰問題の入門
Digit Recognizer - 画像分類の入門
これらは「Getting Started」カテゴリーで、チュートリアルやサンプルコードが豊富です。
このような流れでKaggleコンペに取り組みます。質問があれば教えてください！