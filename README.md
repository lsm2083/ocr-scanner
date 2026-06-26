# VisionLab — 실시간 비전 처리 · OCR · 번역 · 스마트 액션 안드로이드 앱

> **NDK(C++)·OpenCV 실시간 영상처리**부터 **온디바이스/클라우드 OCR**, **번역**, **스마트 액션**까지 한 앱에 통합한 카메라 비전 애플리케이션

카메라로 비추는 영상을 실시간으로 처리(엣지·도형 검출 등)하고, 문서를 스캔해 글자를 인식·번역하며, 인식된 전화번호·날짜 등을 탭 한 번으로 실행(전화·캘린더 등)할 수 있습니다.

---

## ✨ 주요 기능

### 1. 실시간 영상처리 (NDK + OpenCV)
- 8개 모드: 원본 / 흑백 / 블러 / 엣지(Canny) / 엣지 오버레이 / 스케치 / **도형 검출** / **문서 스캔**
- **도형 검출**: 윤곽선(contour) → 꼭짓점·원형도 분석으로 삼각형·사각형·원 분류 + 실시간 개수 카운트
- 임계값 슬라이더 실시간 조절, **네이티브 처리시간(ms)·FPS** 표시, 해상도 선택

### 2. 문서 스캐너 + OCR
- 최대 사각형 검출 → **원근 보정(perspective warp)** → 대비 보정(CLAHE)·샤프닝
- **풀해상도 정지촬영**(ImageCapture)으로 인식 정확도 확보
- OCR 바운딩박스 좌표로 **문서의 행/열 레이아웃 복원**
- **전체 스캔 → 문단 터치** 시 해당 문단만 추출 (좌표 역변환 오버레이)

### 3. 하이브리드 OCR
- **온디바이스**(ML Kit, 빠름·무료·오프라인) ↔ **클라우드**(Google Cloud Vision, 세로쓰기·복잡 문서 정확)
- 상황에 따라 선택하는 하이브리드 전략

### 4. 번역
- **셀 단위 번역**으로 표 레이아웃 유지, 한↔영 방향 선택, 고유명사 보존
- **실시간 AR 번역**: 카메라에 비친 글자를 그 자리에 덮어 번역
  — **프레임 간 차이(차분) 기반 움직임 감지**로 정지 시 고정, 깜빡임 최소화

### 5. 스마트 액션
- 전화·주소·날짜·이메일·링크·택배번호 자동 인식(Entity Extraction)
- 칩 탭 한 번으로 **전화 · 지도 · 캘린더 · 메일 · 브라우저 · 배송조회** 즉시 실행 (Android 표준 Intent)

---

## 🛠 기술 스택

| 구분 | 사용 기술 |
|---|---|
| **언어** | Kotlin (앱), **C++ (네이티브 영상처리)** |
| **플랫폼** | Android (minSdk 29), AGP 9.2 |
| **네이티브** | **NDK · JNI · CMake · prefab** |
| **영상처리** | **OpenCV 4.11** — Canny, findContours, warpPerspective, CLAHE, GaussianBlur 등 |
| **카메라** | **CameraX** (ImageAnalysis / ImageCapture) |
| **온디바이스 ML** | **ML Kit** — Text Recognition(한국어), Translation, Language ID, Entity Extraction |
| **클라우드 ML** | **Google Cloud Vision API** (DOCUMENT_TEXT_DETECTION, REST) |
| **기타** | WorkManager(ML 모델 다운로드), MediaStore(저장), Material Components, ViewBinding, BuildConfig(키 주입) |

---

## 🧩 아키텍처 / 데이터 흐름

```
CameraX (ImageAnalysis, RGBA)
        │
        ▼
OpenCV Mat  ──(네이티브 포인터 전달)──►  C++ / JNI  (Canny·contour·warp 등)
        │                                      │
        ▼                                      ▼
     Bitmap  ◄──── matToBitmap ──────────  처리 결과 Mat
        │
        ├─► ImageView 실시간 렌더 (필터·도형·AR 번역)
        │
        └─► (캡처 시) ML Kit OCR / Cloud Vision
                   │
                   ├─► 레이아웃 복원 → 번역(셀 단위)
                   └─► Entity Extraction → 스마트 액션(Intent)
```

- **JNI 브리지**: Kotlin↔C++ 간 `Mat`의 네이티브 주소를 주고받아 데이터 복사 최소화
- **성능 최적화**: Mat/Bitmap 재사용, 분석 스레드 분리, 실시간 OCR 스로틀링·움직임 감지

---

## 🚀 빌드 & 실행

### 요구 사항
- Android Studio (AGP 9.2+), Android SDK / **NDK**, CMake 3.22+
- 실기기 또는 에뮬레이터 (카메라 필요)

### OpenCV
별도 SDK 다운로드 불필요 — **Maven의 `org.opencv:opencv:4.11.0`**(prefab)으로 자동 연동됩니다.

### (선택) Cloud Vision API 키
클라우드 정밀 인식을 쓰려면 `local.properties`에 키를 추가하세요 *(소스/깃에 노출되지 않음)*:
```properties
CLOUD_VISION_API_KEY=AIza...
```
> 키가 없어도 온디바이스 기능은 모두 동작합니다.

### 실행
```bash
./gradlew :app:installDebug
```

---

## 🧠 기술적 도전 / 트러블슈팅

- **OpenCV prefab 링크**: AAR이 shared STL을 요구 → `-DANDROID_STL=c++_shared`, `find_package(OpenCV)` 후 `OpenCV::opencv_java4` 타깃 직접 링크
- **ML Kit 모델 다운로드 멈춤**: 다운로더가 WorkManager 작업으로 동작 → `androidx.work:work-runtime` 누락 시 무한 대기. 명시적 추가로 해결
- **edge-to-edge(targetSdk 35+)**: 시스템 바에 UI 가려짐 → WindowInsets로 오버레이/패널 패딩 처리
- **세로쓰기 한계**: 온디바이스 OCR이 세로 조판 한글을 오인식 → **클라우드 Vision 하이브리드**로 보완
- **AR 번역 안정화**: 프레임 차분으로 움직임을 감지해 "정지 시 번역 고정" UX 구현

---

## 📸 스크린샷
*(여기에 모드별 화면 / 문서 스캔 / 번역 / 스마트 액션 스크린샷을 추가하세요)*

---

## 📄 라이선스
개인 포트폴리오 프로젝트입니다.
