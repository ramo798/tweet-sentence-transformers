import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import japanize_matplotlib
import random
from datetime import datetime, timedelta
import math

import chromadb
chroma_client = chromadb.PersistentClient(path="./chromadb")
collection = chroma_client.get_or_create_collection(name="tweet_embeddings")


def generate_fake_tweets_with_emb(num=500):
    all_tweet = collection.get(limit=300000, include=[
                               "metadatas", "embeddings"])
    tweets = []
    for ids, metadata, emb in zip(all_tweet['ids'], all_tweet['metadatas'], all_tweet['embeddings']):
        tweets.append({
            "id": int(ids),
            "full_text": metadata['full_text'],
            "created_at": metadata['created_at'],
            "genre": metadata['genre'],
            "emb": emb
        })
    return pd.DataFrame(tweets)


# データロード
tweets_df = generate_fake_tweets_with_emb()

# ページネーション用の関数


def paginate(data, page, items_per_page=100):
    start_idx = page * items_per_page
    end_idx = start_idx + items_per_page
    return data.iloc[start_idx:end_idx]


# サイドバーでページを切り替える
st.sidebar.title("メニュー")
page = st.sidebar.radio("ページを選択", ["ツイート検索", "ジャンル集計", "埋め込み可視化"])

# ツイート検索ページ
if page == "ツイート検索":
    st.title("ツイート検索アプリ")

    # 初期設定
    ITEMS_PER_PAGE = 100
    current_page = st.number_input("ページ番号を選択してください", min_value=0, max_value=math.ceil(
        len(tweets_df) / ITEMS_PER_PAGE) - 1, value=0, step=1)

    # 検索ボックスとジャンル選択
    search_query = st.text_input("検索ワードを入力してください", "")
    selected_genre = st.selectbox(
        "ジャンルを選択してください", ["すべて"] + tweets_df['genre'].unique().tolist())

    # 検索ボタン
    if st.button("検索"):
        # 初期状態で全てを選択
        filtered_df = tweets_df

        # 検索クエリに基づくフィルタリング
        if search_query:
            filtered_df = filtered_df[
                filtered_df['full_text'].str.contains(search_query, case=False)
            ]

        # ジャンルが選択されている場合のフィルタリング
        if selected_genre != "すべて":
            filtered_df = filtered_df[filtered_df['genre'] == selected_genre]
    else:
        filtered_df = tweets_df

    # ページネーションされたデータ取得
    paginated_df = paginate(filtered_df, current_page, ITEMS_PER_PAGE)

    # 結果の表示
    st.subheader("検索結果")
    st.markdown(
        f"**ページ: {current_page + 1} / {math.ceil(len(filtered_df) / ITEMS_PER_PAGE)}**")

    for index, row in paginated_df.iterrows():
        with st.container():
            # ツイートIDをリンクに
            tweet_link = f"https://twitter.com/urakutenism/status/{row['id']}"
            st.markdown(f"**ツイートID:** [{row['id']}]({tweet_link})")

            # 他の情報をそのまま表示
            st.markdown(f"**投稿日:** {row['created_at']}")
            st.markdown(f"**ジャンル:** {row['genre']}")
            st.markdown(f"**内容:** {row['full_text']}")
            st.markdown("---")

# ジャンル集計ページ
elif page == "ジャンル集計":
    st.title("ジャンルごとの集計")
    # 各ジャンルのツイート数を集計
    genre_counts = tweets_df['genre'].value_counts().reset_index()
    genre_counts.columns = ['ジャンル', 'ツイート数']

    # ツイート数が20を超えるジャンルをフィルタリング
    filtered_genre_counts = genre_counts[genre_counts['ツイート数'] > 20]

    st.subheader("ジャンル別ツイート数（20個以上）")
    st.dataframe(filtered_genre_counts)

    # 棒グラフを表示
    st.bar_chart(data=filtered_genre_counts.set_index('ジャンル'))

# 埋め込み可視化ページ
elif page == "埋め込み可視化":
    st.title("埋め込み可視化とクラスタリング")

    # 次元削減の方法
    method = st.sidebar.selectbox("次元削減の方法を選択", ["t-SNE", "PCA"])

    # クラスタ数の設定
    k_clusters = st.sidebar.slider(
        "クラスタ数 (k-means)", min_value=2, max_value=10, value=3, step=1)

    # 埋め込みを取得
    embeddings = np.array(tweets_df['emb'].to_list())

    # 次元削減
    if method == "t-SNE":
        reducer = TSNE(n_components=2, random_state=42)
    elif method == "PCA":
        reducer = PCA(n_components=2)
    reduced_embeddings = reducer.fit_transform(embeddings)

    # クラスタリング
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    # プロットの準備
    reduced_df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
    reduced_df["cluster"] = clusters
    reduced_df["genre"] = tweets_df["genre"]
    reduced_df["full_text"] = tweets_df["full_text"]

    # プロット
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        reduced_df["x"], reduced_df["y"], c=reduced_df["cluster"], cmap="tab10", s=50, alpha=0.8
    )
    legend1 = ax.legend(
        *scatter.legend_elements(), loc="upper right", title="Cluster"
    )
    ax.add_artist(legend1)
    plt.title(f"{method}による埋め込み可視化 (k={k_clusters})")
    plt.xlabel("次元1")
    plt.ylabel("次元2")

    # Streamlitで表示
    st.pyplot(fig)

    # データ表示
    st.subheader("可視化データ")
    st.dataframe(reduced_df)
