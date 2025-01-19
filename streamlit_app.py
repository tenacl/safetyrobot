#클로드로작성, 스트림릿, 소리까지 나옴
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import winsound  # Windows 시스템용 소리 재생
import platform  # 운영체제 확인용

def play_alert():
    if platform.system() == 'Windows':
        winsound.Beep(1000, 500)  # 1000Hz로 0.5초 동안 소리
    else:
        # macOS나 Linux의 경우 print로 대체
        print('\a')  # 시스템 비프음

def main():
    st.title("실시간 인원 감지 시스템")
    
    # 디버깅 메시지
    st.write("프로그램 시작됨")
    
    # YOLO 모델 로드
    @st.cache_resource
    def load_model():
        try:
            model = YOLO('yolov8n.pt')
            st.write("YOLO 모델 로드 성공")
            return model
        except Exception as e:
            st.error(f"모델 로드 실패: {str(e)}")
            return None

    model = load_model()
    
    # 웹캠 초기화 - 수정된 부분
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("웹캠을 열 수 없습니다.")
        return
    
    st.write("웹캠 초기화 성공")
    
    # 프레임을 표시할 플레이스홀더
    frame_placeholder = st.empty()
    
    # 제어 버튼
    stop_button = st.button("감지 정지")
    
    # 이전 프레임의 사람 수를 저장할 변수
    prev_person_count = 0
    
    # 마지막 경고음 재생 시간
    last_alert_time = 0
    
    while not stop_button:
        # 웹캠에서 프레임 읽기 - 수정된 부분
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
                
                # 사람이 감지되고 마지막 경고음으로부터 1초가 지났으면 경고음 재생
                current_time = time.time()
                if person_count > 0 and (current_time - last_alert_time) >= 1.0:
                    play_alert()
                    last_alert_time = current_time
                
                # 현재 사람 수 저장
                prev_person_count = person_count
            
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
