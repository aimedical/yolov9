# ベースイメージに NVIDIA PyTorch コンテナを使用
FROM nvcr.io/nvidia/pytorch:21.11-py3

# 必要なパッケージをインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        zip \
        htop \
        screen \
        libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# pipで追加パッケージをインストール
RUN pip install seaborn thop backports.tarfile

# 作業ディレクトリを /yolov9 に変更
WORKDIR /yolov9

# コンテナ起動時のデフォルトコマンド（必要に応じて変更）
CMD ["/bin/bash"]