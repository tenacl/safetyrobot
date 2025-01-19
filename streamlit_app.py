import streamlit as st
import cv2
import av
import time
from datetime import datetime
import numpy as np
from ultralytics import YOLO

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    VideoProcessorBase,
)

# 감지 로그 시작 시간 저장
if "detection_start_time" not in st.session_state:
    st.session_state.detection_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Custom CSS for dark theme and UI styling
dark_theme = """
<style>
body {
    background-color: #2c2f33;  /* 어두운 배경색 */
    color: white;  /* 기본 텍스트 색상 */
}
.stButton>button {
    background-color: #5865f2;
    color: white;
    border-radius: 5px;
}
</style>
"""
st.markdown(dark_theme, unsafe_allow_html=True)

# 사람 감지용 비디오 처리 클래스
class YOLOPersonDetector(VideoProcessorBase):
    def __init__(self):
        self.model = YOLO("yolov8n.pt")  # YOLO 모델 로드
        self.last_alert_time = 0
        self.person_detected = False  # 사람 감지 여부

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # OpenCV BGR 배열로 변환

        # YOLO 추론
        results = self.model(img)[0]

        person_count = 0
        for box in results.boxes:
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())

            # 'person' 클래스 확인
            if class_id == 0 and confidence > 0.5:
                person_count += 1
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

        # 감지 여부 업데이트
        self.person_detected = person_count > 0

        # 경고 메시지 표시
        current_time = time.time()
        if self.person_detected and (current_time - self.last_alert_time) >= 1.0:
            self.last_alert_time = current_time

        # 다시 AV VideoFrame으로 반환
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    # 로그인 후 앱 실행
    st.title("실시간 인원 감지 시스템")
    st.subheader(f"감지 로그 시작 시간: {st.session_state.detection_start_time}")

    # WebRTC 설정
    rtc_configuration = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    # YOLO 감지기 초기화
    webrtc_ctx = webrtc_streamer(
        key="person-detect",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=YOLOPersonDetector,
        async_processing=True
    )

    # 카메라 위에 사람 감지 메시지 표시
    if webrtc_ctx.video_processor:
        if webrtc_ctx.video_processor.person_detected:
            st.markdown(
                """
                <div style="color: red; font-size: 20px; text-align: center; margin-top: 10px;">
                사람 감지됨!
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 로그아웃 버튼
    if st.button("로그아웃"):
        st.session_state.logged_in = False
        st.experimental_rerun()


if __name__ == "__main__":
    # 세션 상태 관리
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = True  # 기본값 True로 설정

    if st.session_state.logged_in:
        main()
    else:
        st.write("로그인 필요")
