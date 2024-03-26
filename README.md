# akari_chatgpt_indian_poker

AKARIとインディアン・ポーカーで勝負できるアプリ

## submoduleの更新
`git submodule update --init --recursive`  

## ライブラリのインストール  
`sudo apt install pportaudio19-dev`  

## 仮想環境の作成
`python -m venv venv`  
`source venv/bin/activate`  
`pip install -r requirements.txt`  

## セットアップ方法
[akari_chatgpt_botのREADME](https://github.com/AkariGroup/akari_chatgpt_bot/blob/main/README.md)のセットアップ手順に沿って実行する。

## 起動方法
1. マイクとスピーカーを接続する。  

2. [akari_chatgpt_botのREADME](https://github.com/AkariGroup/akari_chatgpt_bot/blob/main/README.md)内 **VOICEVOXをOSS版で使いたい.場合** の手順を元に、別PCでVoicevoxを起動しておく。  

3. akari_motion_serverを起動する。  
   起動方法は https://github.com/AkariGroup/akari_motion_server を参照。  

4. gpt_indian_poker.pyを起動する。
   `python3 gpt_indian_poker.py --voicevox_local --voicevox_host {VOICEVOXを起動しているPCのIPアドレス}`  
