import argparse

import os
import sys
import openai
import grpc
import threading
import json
import time
import random
from akari_client import AkariClient
from akari_client.color import Colors
from akari_client.position import Positions
from akari_chatgpt_bot.lib.chat_akari import ChatStreamAkari
from akari_chatgpt_bot.lib.google_speech import (
    MicrophoneStream,
    get_db_thresh,
    listen_print_loop,
)
import cv2
from distutils.util import strtobool
from lib.akari_yolo_lib.oakd_yolo import OakdYolo
from lib.akari_yolo_lib.util import download_file

sys.path.append(os.path.join(os.path.dirname(__file__), "lib/grpc"))
import motion_server_pb2
import motion_server_pb2_grpc

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
POWER_THRESH_DIFF = 25  # 周辺音量にこの値を足したものをpower_threshouldとする

host: str = ""
port: str = ""

akari = AkariClient()
m5 = akari.m5stack


class ChatStreamAkariPoker(ChatStreamAkari):
    def judge_card_change_gpt(
        self,
        messages: list,
        model: str = "gpt-4-turbo",
        temperature: float = 0.7,
    ) -> bool:
        result = openai.chat.completions.create(
            model=model,
            messages=messages,
            n=1,
            temperature=temperature,
            functions=[
                {
                    "name": "judge_card_change",
                    "description": "今までの会話から、インディアン・ポーカーに勝つために自分のカードを変更すべきか判断する。あかりの数値が弱いと思ったら積極的に変更する。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "change": {
                                "type": "boolean",
                                "description": "カードを変更するか",
                            },
                        },
                        "required": ["change"],
                    },
                }
            ],
            function_call={"name": "judge_card_change"},
            stream=False,
            stop=None,
        )
        message = result.choices[0].message
        arguments = json.loads(message.function_call.arguments)
        return bool(arguments["change"])

    def judge_card_change_anthropic(
        self,
        messages: list,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
    ) -> bool:
        system_message = ""
        user_messages = []
        for message in messages:
            if message["role"] == "system":
                system_message = message["content"]
            else:
                user_messages.append(message)
        user_messages.append(
            self.create_message(
                text='今までの会話から、インディアン・ポーカーに勝つために自分のカードを変更すべきか判断してください。そのまま勝てそうなら変更しないで、負けそうなら変更してください。返答は下記のJSON形式で出力してください。{"change": "カードを変更するか。"true" or "false"で返す。"}'
            )
        )
        # 最後の1文を動作と文章のJSON形式出力指定に修正
        response = self.anthropic_client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=temperature,
            messages=user_messages,
            system=system_message,
        )
        print(response.content[0].text)
        arguments = json.loads(response.content[0].text)
        return strtobool(arguments["change"])

    def judge_card_change(
        self,
        messages: list,
        model: str = "gpt-4-turbo",
        temperature: float = 0.7,
    ) -> bool:
        res = False
        if model in self.openai_model_name:
            res = self.judge_card_change_gpt(
                messages=messages, model=model, temperature=temperature
            )
        elif model in self.anthropic_model_name:
            res = self.judge_card_change_anthropic(
                messages=messages, model=model, temperature=temperature
            )
        else:
            print(f"Model name {model} can't use for this function")
        return res


class CardDetector(OakdYolo):
    def __init__(self, config_path: str, model_path: str, fps: int = 10) -> None:
        super().__init__(config_path, model_path, fps)
        self.detected_count = [0] * 52
        self.card_result: str = ""
        self.DETECTION_THRESHOULD = 5
        self.is_judging = False

    def get_card_result(self) -> str:
        while True:
            time.sleep(0.1)
            if len(self.card_result) == 0:
                continue
            # 最後の1文字を除去
            string_without_last = self.card_result[:-1]
            # 整数に変換
            try:
                result = int(string_without_last)
                break
            except ValueError:
                continue
        return result

    def start_judging(self) -> None:
        self.detected_count = [0] * 52
        self.card_result: str = ""
        self.is_judging = True

    def loop(self) -> None:
        end = False
        while not end:
            while True:
                frame = None
                detections = []
                try:
                    frame, detections = self.get_frame()
                except BaseException:
                    print("===================")
                    print("get_frame() error! Reboot OAK-D.")
                    print("If reboot occur frequently, Bandwidth may be too much.")
                    print("Please lower FPS.")
                    print("==================")
                if detections is not None:
                    if len(detections) >= 1 and self.is_judging:
                        self.detected_count[detections[0].label] += 1
                        if (
                            self.detected_count[detections[0].label]
                            >= self.DETECTION_THRESHOULD
                        ):
                            self.card_result = self.labels[detections[0].label]
                            self.is_judging = False
                if frame is not None:
                    self.display_frame("nn", frame, detections)
                if cv2.waitKey(1) == ord("q"):
                    end = True
                    break
            self.close()


def countdown(seconds):
    for i in range(seconds, 0, -1):
        m5.set_display_text(
            text=str(i),
            pos_x=Positions.CENTER,
            pos_y=Positions.CENTER,
            size=9,
            text_color=Colors.BLACK,
            back_color=Colors.WHITE,
            refresh=False,
            sync=False,
        )
        time.sleep(1)  # 1秒待つ


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", help="Ip address", default="127.0.0.1", type=str)
    parser.add_argument("--robot_port", help="Port number", default="50055", type=str)
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=0.5,
        help="Microphone input power timeout",
    )
    parser.add_argument(
        "-p",
        "--power_threshold",
        type=float,
        default=0,
        help="Microphone input power threshold",
    )
    parser.add_argument("--voicevox_local", action="store_true")
    parser.add_argument(
        "--voice_host",
        type=str,
        default="127.0.0.1",
        help="voice server host",
    )
    parser.add_argument(
        "--voice_port",
        type=str,
        default="50021",
        help="voice server port",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Provide model name or model path for inference",
        default="model/card.blob",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Provide config path for inference",
        default="json/card.json",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--fps",
        help="Camera frame fps. This should be smaller than nn inference fps",
        default=7,
        type=int,
    )
    args = parser.parse_args()
    model_path = "model/card.blob"
    config_path = "config/card.json"
    download_file(
        model_path,
        "https://github.com/AkariGroup/akari_yolo_models/raw/main/card/card.blob",
    )
    download_file(
        config_path,
        "https://github.com/AkariGroup/akari_yolo_models/raw/main/card/card.json",
    )
    timeout: float = args.timeout
    power_threshold: float = args.power_threshold
    if power_threshold == 0:
        power_threshold = get_db_thresh() + POWER_THRESH_DIFF
    print(f"power_threshold set to {power_threshold:.3f}db")
    if args.voicevox_local:
        from akari_chatgpt_bot.lib.voicevox import TextToVoiceVox

        host = args.voice_host
        port = args.voice_port
        text_to_voice = TextToVoiceVox(host, port)
    else:
        from akari_chatgpt_bot.lib.conf import VOICEVOX_APIKEY
        from akari_chatgpt_bot.lib.voicevox import TextToVoiceVoxWeb

        text_to_voice = TextToVoiceVoxWeb(apikey=VOICEVOX_APIKEY)
    channel = grpc.insecure_channel(args.robot_ip + ":" + str(args.robot_port))
    stub = motion_server_pb2_grpc.MotionServerServiceStub(channel)
    chat_stream_akari_poker = ChatStreamAkariPoker(args.robot_ip, args.robot_port)
    card_detector = CardDetector(
        config_path=config_path,
        model_path=model_path,
        fps=args.fps,
    )
    detection_thread = threading.Thread(target=card_detector.loop)
    detection_thread.start()
    m5.set_display_text(
        "AKARI", Positions.CENTER, 30, 5, Colors.RED, Colors.WHITE, True
    )
    m5.set_display_text(
        "インディアン",
        Positions.CENTER,
        100,
        3,
        Colors.BLACK,
        Colors.WHITE,
        False,
    )
    m5.set_display_text(
        "ポーカー",
        Positions.CENTER,
        140,
        3,
        Colors.BLACK,
        Colors.WHITE,
        False,
    )
    m5.set_display_text(
        "スタート",
        Positions.CENTER,
        Positions.BOTTOM,
        3,
        Colors.RED,
        Colors.WHITE,
        False,
    )
    while True:
        data = m5.get()
        if data["button_b"]:
            break
        time.sleep(0.01)
    end = False
    while not end:
        card_detector.start_judging()
        messages = [
            {
                "role": "system",
                "content": "チャットボットとしてロールプレイをします。あかりという名前の、カメラロボットとして振る舞ってください。これからインディアン・ポーカーをします。1から13のカードがあり、数字が大きいほど強いです。敵のカードの数字は見えますが、自分のカードの数字は分かりません。敵より強いカードを持っていたらあなたの勝ちで、1回だけカードを交換できます。",
            }
        ]

        # カードの判定
        m5.set_display_text(
            text="AKARIのカード",
            pos_x=Positions.CENTER,
            pos_y=Positions.TOP,
            size=4,
            text_color=Colors.BLACK,
            back_color=Colors.WHITE,
            refresh=True,
        )
        akari_card_num = random.randint(1, 13)
        m5.set_display_text(
            text=str(akari_card_num),
            pos_x=Positions.CENTER,
            pos_y=Positions.CENTER,
            size=11,
            text_color=Colors.RED,
            back_color=Colors.WHITE,
            refresh=False,
            sync=False,
        )
        player_card_num = card_detector.get_card_result()
        with MicrophoneStream(RATE, CHUNK, timeout, power_threshold) as stream:
            # 対話タイム
            m5.set_display_text(
                text="対話タイムまで",
                pos_x=Positions.CENTER,
                pos_y=Positions.TOP,
                size=4,
                text_color=Colors.BLACK,
                back_color=Colors.WHITE,
                refresh=True,
            )
            countdown(3)
            text = ""
            m5.set_display_text(
                text="発言してください",
                pos_x=Positions.CENTER,
                pos_y=Positions.TOP,
                size=3,
                text_color=Colors.BLACK,
                back_color=Colors.WHITE,
                refresh=True,
            )
            m5.set_display_text(
                text=str(akari_card_num),
                pos_x=Positions.CENTER,
                pos_y=Positions.CENTER,
                size=11,
                text_color=Colors.RED,
                back_color=Colors.WHITE,
                refresh=False,
                sync=False,
            )
            # うなずきモーション再生
            try:
                stub.SetMotion(
                    motion_server_pb2.SetMotionRequest(
                        name="nod", priority=3, repeat=True
                    )
                )
            except BaseException:
                print("akari_motion_server is not working.")
            responses = stream.transcribe()
            if responses is not None:
                text = listen_print_loop(responses)
        messages.append(
            {
                "role": "user",
                "content": f"「{text}」これはあかりのカードの数値に対する相手の発言で、嘘かもしれません。敵のカードは{player_card_num}です。これは事実です。以上を元に、今度はあなたが敵のカードについて、敵を騙すか、たまに本当のことを伝えるコメントを簡潔にしてください。敵の数値は発言してはいけません。コメントだけを返してください。",
            }
        )
        # うなずきリピート停止
        try:
            stub.StopRepeat(motion_server_pb2.StopRepeatRequest(priority=3))
        except BaseException:
            print("akari_motion_server is not working.")
        response = ""
        for sentence in chat_stream_akari_poker.chat(
            messages, model="gpt-4-turbo"
        ):
            text_to_voice.put_text(sentence)
            response += sentence
            print(sentence, end="", flush=True)
        messages.append({"role": "assistant", "content": response})
        text_to_voice.wait_finish()
        print("")
        time.sleep(1)

        # カード交換タイム
        change = chat_stream_akari_poker.judge_card_change(messages)
        if change:
            try:
                stub.SetMotion(
                    motion_server_pb2.SetMotionRequest(
                        name="agree", priority=3, repeat=False
                    )
                )
            except BaseException:
                print("akari_motion_server is not working.")
            text_to_voice.put_text(
                "カードを交換するよ。あなたも交換するなら、カウントダウン中にね。"
            )
            akari_card_num = random.randint(1, 13)
        else:
            try:
                stub.SetMotion(
                    motion_server_pb2.SetMotionRequest(
                        name="swing", priority=3, repeat=False
                    )
                )
            except BaseException:
                print("akari_motion_server is not working.")
            text_to_voice.put_text(
                "カードを交換しないよ。あなたが交換するなら、カウントダウン中にね。"
            )
        time.sleep(4)
        # 判定タイム
        m5.set_display_text(
            text="カード交換終了まで",
            pos_x=Positions.CENTER,
            pos_y=Positions.TOP,
            size=2,
            text_color=Colors.BLACK,
            back_color=Colors.WHITE,
            refresh=True,
        )
        countdown(5)
        m5.set_display_text(
            text="AKARIのカード",
            pos_x=Positions.CENTER,
            pos_y=Positions.TOP,
            size=4,
            text_color=Colors.BLACK,
            back_color=Colors.WHITE,
            refresh=True,
        )
        m5.set_display_text(
            text=str(akari_card_num),
            pos_x=Positions.CENTER,
            pos_y=Positions.CENTER,
            size=11,
            text_color=Colors.RED,
            back_color=Colors.WHITE,
            refresh=False,
            sync=False,
        )
        card_detector.start_judging()
        player_card_num = card_detector.get_card_result()
        if akari_card_num > player_card_num:
            messages.append(
                {"role": "user", "content": "勝負の結果あかりが勝ちました。"}
            )
            m5.set_display_text(
                text="あなたの負け",
                pos_x=Positions.CENTER,
                pos_y=Positions.BOTTOM,
                size=4,
                text_color=Colors.BLUE,
                back_color=Colors.WHITE,
                refresh=False,
                sync=False,
            )
        elif akari_card_num < player_card_num:
            messages.append(
                {"role": "user", "content": "勝負の結果あかりが負けました。"}
            )
            m5.set_display_text(
                text="あなたの勝ち",
                pos_x=Positions.CENTER,
                pos_y=Positions.BOTTOM,
                size=4,
                text_color=Colors.RED,
                back_color=Colors.WHITE,
                refresh=False,
                sync=False,
            )
        else:
            messages.append({"role": "user", "content": "勝負の結果引き分けでした。"})
            m5.set_display_text(
                text="引き分け",
                pos_x=Positions.CENTER,
                pos_y=Positions.BOTTOM,
                size=4,
                text_color=Colors.GREEN,
                back_color=Colors.WHITE,
                refresh=False,
                sync=False,
            )
        response = ""
        for sentence in chat_stream_akari_poker.chat(
            messages, model="gpt-3.5-turbo"
        ):
            text_to_voice.put_text(sentence)
            response += sentence
            print(sentence, end="", flush=True)
        print("")
        time.sleep(7)

        # 再プレイ判定
        m5.set_display_text(
            text="もう一回",
            pos_x=20,
            pos_y=Positions.BOTTOM,
            size=3,
            text_color=Colors.RED,
            back_color=Colors.WHITE,
            refresh=True,
            sync=True,
        )
        m5.set_display_text(
            text="終わり",
            pos_x=200,
            pos_y=Positions.BOTTOM,
            size=3,
            text_color=Colors.BLUE,
            back_color=Colors.WHITE,
            refresh=False,
            sync=True,
        )
        while True:
            data = m5.get()
            if data["button_a"]:
                break
            elif data["button_c"]:
                end = True
                break
            time.sleep(0.01)
    # 終了処理
    m5.set_display_image("/jpg/logo320.jpg")
    detection_thread.join()


if __name__ == "__main__":
    main()
