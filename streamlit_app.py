import streamlit as st
import cv2
import av
import time
import numpy as np
from ultralytics import YOLO

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    VideoProcessorBase,
)

# Custom CSS
hide_github_style = """
<style>
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_github_style, unsafe_allow_html=True)


# 경고음을 대신할 경고메시지 표시 함수
def play_alert():
    st.warning("⚠️ 사람 감지됨!")
    print('\a')  # 간단한 콘솔 비프음

# 사람 감지용 비디오 처리 클래스
class YOLOPersonDetector(VideoProcessorBase):
    def __init__(self):
        # YOLO 모델 로드 (원하는 모델 가중치 사용 가능)
        self.model = YOLO("yolov8n.pt")
        self.last_alert_time = 0

    def recv(self, frame):
        # webrtc로부터 받은 프레임(AV)을 OpenCV용 BGR 배열로 변환
        img = frame.to_ndarray(format="bgr24")

        # YOLO 추론
        results = self.model(img)[0]

        person_count = 0
        for box in results.boxes:
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())

            # class_id == 0 이 'person' 클래스임 (COCO 데이터셋 기준)
            if class_id == 0 and confidence > 0.5:
                person_count += 1
                # 바운딩 박스 그리기
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"Person {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # 1초 간격으로 경고 표시
        current_time = time.time()
        if person_count > 0 and (current_time - self.last_alert_time) >= 1.0:
            play_alert()
            self.last_alert_time = current_time

        # 다시 webrtc로 반환할 수 있는 av.VideoFrame 형태로 변환
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.title("실시간 인원 감지 시스템 (streamlit-webrtc)")

    st.markdown("""
    - 브라우저에서 '카메라 권한'을 허용하면, 실시간 웹캠 영상이 스트리밍됩니다.
    - 사람이 감지되면 경고 메시지(⚠️)가 표시됩니다.
    """)

    # WebRTC 연결을 위한 STUN 서버 설정 (구글 공개 STUN)
    rtc_configuration = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    # webrtc_streamer: 브라우저와 오디오/비디오 양방향 송수신
    webrtc_ctx = webrtc_streamer(
        key="person-detect",
        mode=WebRtcMode.SENDRECV,   # 영상 업/다운 모두
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=YOLOPersonDetector,
        async_processing=True
    )

    # 감지 중단 버튼
    if st.button("감지 정지"):
        if webrtc_ctx.state.playing:
            webrtc_ctx.stop()
            st.info("감지를 정지했습니다.")


if __name__ == "__main__":
    main()
