# Streamlit, YOLO, OpenCV 사용, 소리 알림 포함
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import platform  # 운영체제 확인용

# 경고음을 재생하는 함수
def play_alert():
    # 소리 대신 로그로 대체 (Streamlit Cloud에서는 소리 재생이 제한됨)
    st.warning("⚠️ 사람 감지됨!")
    print('\a')  # 시스템 비프음

# 메인 함수
def main():
    st.title("실시간 인원 감지 시스템")
    
    # 디버깅 메시지
    st.write("프로그램 시작됨")
    
    # YOLO 모델 로드
    @st.cache_resource
    def load_model():
        try:
            model = YOLO('yolov8n.pt')  # YOLO 모델 로드
            st.write("YOLO 모델 로드 성공")
            return model
        except Exception as e:
            st.error(f"모델 로드 실패: {str(e)}")
            return None

    model = load_model()
    
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("웹캠을 열 수 없습니다. 웹캠이 연결되어 있는지 확인하세요.")
        return
    
    st.write("웹캠 초기화 성공")
    
    # 프레임을 표시할 플레이스홀더
    frame_placeholder = st.empty()
    
    # 제어 버튼
    stop_button = st.button("감지 정지")
    
    # 마지막 경고음 재생 시간
    last_alert_time = 0

    # 감지 루프
    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("프레임을 읽을 수 없습니다.")
            continue
            
        try:
            # YOLO로 객체 감지
            if model is not None:
                results = model(frame)[0]
                
                # 현재 프레임의 사람 수 계산
                person_count = 0
                
                # 감지된 객체에 박스 그리기
                for box in results.boxes:
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    
                    if class_id == 0 and confidence > 0.5:  # person class
                        person_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Person {confidence:.2f}", 
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, (0, 255, 0), 2)
                
                # 사람이 감지되고 마지막 경고음으로부터 1초가 지났으면 경고 알림 표시
                current_time = time.time()
                if person_count > 0 and (current_time - last_alert_time) >= 1.0:
                    play_alert()
                    last_alert_time = current_time

            # BGR을 RGB로 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 프레임 표시
            frame_placeholder.image(frame, channels="RGB")
            
        except Exception as e:
            st.error(f"에러 발생: {str(e)}")
            break
            
        # CPU 사용량 조절
        time.sleep(0.1)
    
    # 리소스 해제
    cap.release()
    st.write("프로그램 종료됨")

if __name__ == '__main__':
    main()
