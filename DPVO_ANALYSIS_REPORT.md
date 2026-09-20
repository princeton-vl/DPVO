# DPVO (Deep Patch Visual Odometry) 상세 분석 보고서

## 1. 개요

### 1.1 프로젝트 정보
- **논문**: "Deep Patch Visual Odometry" (NeurIPS 2023)
- **저자**: Zachary Teed*, Lahav Lipson*, Jia Deng (Princeton University)
- **GitHub**: https://github.com/princeton-vl/DPVO
- **확장 프로젝트**: DPV-SLAM (ECCV 2024)

### 1.2 핵심 기여
DPVO는 **희소 패치 기반(Sparse Patch-based)** 딥러닝 Visual Odometry 시스템으로, 기존의 Dense Flow 기반 방법(DROID-SLAM)보다 **3배 빠르면서 1/3의 메모리만 사용**하고도 **더 높은 정확도**를 달성합니다.

### 1.3 주요 성능 지표
| 메트릭 | DROID-VO | DPVO (Default) | DPVO (Fast) |
|--------|----------|----------------|-------------|
| 평균 FPS | 40 | 60 | 120 |
| 최소 FPS | 11 | 48 | 98 |
| GPU 메모리 | 8.7GB | 4.9GB | 2.5GB |

---

## 2. 프로젝트 구조

```
DPVO/
├── dpvo/                      # 핵심 라이브러리
│   ├── __init__.py
│   ├── dpvo.py               # 메인 DPVO 클래스
│   ├── net.py                # VONet 신경망 정의
│   ├── extractor.py          # Feature Extractor
│   ├── blocks.py             # 신경망 블록 (GatedResidual, SoftAgg 등)
│   ├── patchgraph.py         # Patch Graph 데이터 구조
│   ├── projective_ops.py     # 투영 연산 (iproj, proj, transform)
│   ├── ba.py                 # Bundle Adjustment (PyTorch)
│   ├── config.py             # 설정 파일 (YACS)
│   ├── utils.py              # 유틸리티 함수
│   ├── stream.py             # 이미지/비디오 스트림 처리
│   ├── plot_utils.py         # 시각화 유틸리티
│   ├── logger.py             # 학습 로거
│   │
│   ├── altcorr/              # CUDA Correlation 연산
│   │   ├── __init__.py
│   │   ├── correlation.py    # Python wrapper
│   │   ├── correlation.cpp   # C++ binding
│   │   └── correlation_kernel.cu  # CUDA 커널
│   │
│   ├── fastba/               # CUDA Bundle Adjustment
│   │   ├── __init__.py
│   │   ├── ba.py             # Python wrapper
│   │   ├── ba.cpp            # C++ binding
│   │   ├── ba_cuda.cu        # CUDA 커널
│   │   └── block_e.cu        # Block elimination
│   │
│   ├── lietorch/             # Lie Group 연산 라이브러리
│   │   ├── __init__.py
│   │   ├── groups.py         # SO3, SE3, Sim3 클래스
│   │   ├── group_ops.py      # Exp, Log, Inv, Mul 등
│   │   ├── broadcasting.py   # 브로드캐스팅 유틸리티
│   │   └── src/              # C++/CUDA 소스
│   │
│   ├── loop_closure/         # Loop Closure (DPV-SLAM)
│   │   ├── long_term.py      # Long-term Loop Closure
│   │   ├── optim_utils.py    # 최적화 유틸리티
│   │   └── retrieval/        # 이미지 검색 (DBoW2)
│   │
│   └── data_readers/         # 데이터 로더
│       ├── factory.py
│       ├── base.py
│       ├── tartan.py         # TartanAir 데이터셋
│       └── augmentation.py
│
├── DPViewer/                 # Pangolin 기반 시각화 모듈
├── DPRetrieval/              # DBoW2 기반 이미지 검색 (C++)
├── Pangolin/                 # 3D 시각화 라이브러리 (submodule)
├── DBoW2/                    # Bag of Words (submodule)
│
├── demo.py                   # 데모 실행 스크립트
├── train.py                  # 학습 스크립트
├── evaluate_*.py             # 벤치마크 평가 스크립트
├── setup.py                  # 패키지 설치
└── environment.yml           # Conda 환경
```

---

## 3. 핵심 알고리즘 분석

### 3.1 Patch 표현 (Patch Representation)

DPVO의 핵심은 **Deep Patch Representation**입니다. 각 패치는 4×p² 동차 좌표 배열로 표현됩니다:

```
P_k = [x, y, 1, d]^T    (x, y, d ∈ R^(1×p²))
```

여기서:
- `(x, y)`: 픽셀 좌표
- `d`: 역깊이 (inverse depth)
- `p`: 패치 크기 (기본값 3×3)

**패치는 frontoparallel plane을 가정**하여 전체 패치에 동일한 깊이를 적용합니다.

### 3.2 Patch Graph

패치와 프레임 간의 관계를 **이분 그래프(Bipartite Graph)**로 표현합니다:

```python
# patchgraph.py에서 정의
class PatchGraph:
    - ii: 소스 프레임 인덱스
    - jj: 타겟 프레임 인덱스
    - kk: 패치 인덱스
    - net: Edge별 hidden state
    - target: 2D flow revision targets
    - weight: Confidence weights
```

기본적으로 각 패치는 소스 프레임으로부터 거리 `r` 이내의 모든 프레임과 연결됩니다.

### 3.3 Feature Extraction

두 개의 ResNet 기반 인코더를 사용합니다:

```python
# extractor.py - BasicEncoder4
class Patchifier:
    fnet: BasicEncoder4(output_dim=128, norm_fn='instance')  # Matching features
    inet: BasicEncoder4(output_dim=384, norm_fn='none')      # Context features
```

**구조**:
1. 7×7 Conv (stride 2) → 1/2 resolution
2. 2× ResidualBlock (32 channels)
3. 2× ResidualBlock (64 channels, stride 2) → 1/4 resolution
4. 1×1 Conv → output dimension

**출력**:
- **Matching features**: 128-dim, 1/4 & 1/8 해상도 피라미드
- **Context features**: 384-dim, 패치별 context 정보

### 3.4 Update Operator

Update Operator는 DPVO의 핵심 반복 모듈입니다:

```
Input: Hidden State (k, 384), Context (k, 384), Correlation (k, 2×49×p²)
       ↓
[Correlation Encoder] → (k, 384)
       ↓
[1D Temporal Convolution] → 시간축 정보 전파
       ↓
[SoftMax Aggregation] → 패치/프레임간 메시지 전달
       ↓
[Transition Block (GRU)] → Hidden state 업데이트
       ↓
[Factor Head] → δ (2D flow revision), Σ (confidence)
       ↓
[Bundle Adjustment] → 카메라 포즈 & 깊이 업데이트
```

#### 3.4.1 Correlation 연산

```python
# altcorr/correlation.py
def corr(fmap1, fmap2, coords, ii, jj, radius=1, dropout=1):
    # 패치 feature g와 프레임 feature f 간의 내적
    # 7×7 neighborhood에서 correlation 계산
    C_uvαβ = <g_uv, f(P'_kj(u,v) + Δ_αβ)>
```

2-level 피라미드(1/4, 1/8)에서 correlation을 계산하여 concatenate합니다.

#### 3.4.2 1D Temporal Convolution

```python
# net.py - Update.forward()
ix, jx = fastba.neighbors(kk, jj)  # 시간적 이웃 인덱싱
net = net + self.c1(mask_ix * net[:,ix])  # (k, j-1) 이웃
net = net + self.c2(mask_jx * net[:,jx])  # (k, j+1) 이웃
```

각 패치 trajectory를 따라 시간 정보를 전파합니다.

#### 3.4.3 SoftMax Aggregation

```python
# blocks.py
class SoftAgg:
    def forward(self, x, ix):
        w = scatter_softmax(self.g(x), jx, dim=1)  # Attention weight
        y = scatter_sum(self.f(x) * w, jx, dim=1)  # Weighted sum
        return self.h(y)[:,jx]  # Expand back
```

두 종류의 aggregation 수행:
1. **Patch Aggregation**: 같은 패치에 연결된 모든 edge
2. **Frame Aggregation**: 같은 소스/타겟 프레임에 연결된 edge

#### 3.4.4 Factor Head

```python
# net.py
self.d = nn.Sequential(nn.ReLU(), nn.Linear(DIM, 2), GradientClip())  # Flow revision
self.w = nn.Sequential(nn.ReLU(), nn.Linear(DIM, 2), GradientClip(), nn.Sigmoid())  # Confidence
```

### 3.5 Differentiable Bundle Adjustment

Factor Head에서 예측한 `(δ, Σ)`를 사용하여 최적화 목표 정의:

```
minimize Σ_{(k,j)∈E} ||ω_ij(T, P_k) - [P'_kj + δ_kj]||²_Σkj
```

**Gauss-Newton 2회 반복**으로 카메라 포즈와 역깊이를 업데이트합니다.

```python
# fastba/ba.py
def BA(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, M, iterations):
    return cuda_ba.forward(...)  # CUDA 최적화 구현
```

**Schur Complement**를 사용하여 효율적인 분해 수행:
- 포즈 변수를 먼저 제거하여 깊이만의 선형 시스템으로 변환

---

## 4. 모듈별 상세 분석

### 4.1 dpvo/dpvo.py - 메인 클래스

```python
class DPVO:
    def __init__(self, cfg, network, ht=480, wd=640, viz=False):
        # 상태 초기화
        self.M = cfg.PATCHES_PER_FRAME  # 프레임당 패치 수 (기본 80)
        self.N = cfg.BUFFER_SIZE        # 버퍼 크기 (기본 4096)

        # 메모리 할당
        self.imap_ = torch.zeros(pmem, M, DIM)      # Context features
        self.gmap_ = torch.zeros(pmem, M, 128, P, P) # Matching features
        self.fmap1_, self.fmap2_ = ...              # Feature pyramid

        self.pg = PatchGraph(...)  # Patch graph 초기화
```

**핵심 메서드**:

| 메서드 | 설명 |
|--------|------|
| `__call__(t, image, intrinsics)` | 새 프레임 처리 |
| `update()` | Update operator + BA 실행 |
| `keyframe()` | 키프레임 제거 로직 |
| `terminate()` | 최종 포즈 trajectory 반환 |
| `motion_probe()` | 초기화용 모션 체크 |

**처리 흐름**:
1. Feature extraction (`network.patchify`)
2. Patch 초기화 (랜덤 깊이 또는 이전 프레임 median)
3. Edge 추가 (forward/backward)
4. Update operator 실행
5. Keyframe 관리

### 4.2 dpvo/net.py - 신경망 정의

```python
class VONet(nn.Module):
    P = 3          # Patch size
    DIM = 384      # Hidden dimension
    RES = 4        # Feature resolution (1/4)

    def __init__(self):
        self.patchify = Patchifier(self.P)
        self.update = Update(self.P)
```

**Patchifier**:
- 이미지에서 특징 추출 및 패치 샘플링
- 랜덤 또는 그래디언트 기반 패치 위치 선택

```python
def forward(self, images, patches_per_image=80, centroid_sel_strat='RANDOM'):
    fmap = self.fnet(images) / 4.0  # Matching features
    imap = self.inet(images) / 4.0  # Context features

    # 랜덤 패치 샘플링
    x = torch.randint(1, w-1, size=[n, patches_per_image])
    y = torch.randint(1, h-1, size=[n, patches_per_image])

    # Bilinear sampling으로 패치 추출
    imap = altcorr.patchify(imap, coords, 0)
    gmap = altcorr.patchify(fmap, coords, P//2)
```

### 4.3 dpvo/blocks.py - 신경망 블록

#### GatedResidual
```python
class GatedResidual(nn.Module):
    # x + gate(x) * res(x)
    gate = nn.Sequential(nn.Linear(dim), nn.Sigmoid())
    res = nn.Sequential(nn.Linear(dim), nn.ReLU(), nn.Linear(dim))
```

#### SoftAgg (Softmax Aggregation)
```python
class SoftAgg(nn.Module):
    # Weighted aggregation with learned attention
    # w = softmax(g(x))
    # y = sum(f(x) * w)
    # return h(y)
```

#### GradientClip
```python
class GradientClip(nn.Module):
    # 역전파 시 그래디언트를 [-0.01, 0.01]로 클리핑
    # NaN 그래디언트를 0으로 대체
```

### 4.4 dpvo/projective_ops.py - 투영 연산

```python
def iproj(patches, intrinsics):
    """역투영: 2D → 3D"""
    xn = (x - cx) / fx
    yn = (y - cy) / fy
    X = torch.stack([xn, yn, 1, d], dim=-1)
    return X

def proj(X, intrinsics):
    """투영: 3D → 2D"""
    d = 1.0 / Z.clamp(min=0.1)
    x = fx * (d * X) + cx
    y = fy * (d * Y) + cy
    return torch.stack([x, y], dim=-1)

def transform(poses, patches, intrinsics, ii, jj, kk):
    """패치 k를 프레임 i에서 j로 재투영"""
    X0 = iproj(patches[:,kk], intrinsics[:,ii])  # 역투영
    Gij = poses[:, jj] * poses[:, ii].inv()       # 상대 포즈
    X1 = Gij[:,:,None,None] * X0                  # 변환
    x1 = proj(X1, intrinsics[:,jj])               # 투영
    return x1
```

### 4.5 dpvo/lietorch/ - Lie Group 라이브러리

SE3, SO3, Sim3 등 Lie Group 연산을 위한 커스텀 라이브러리:

```python
class SE3(LieGroup):
    group_name = 'SE3'
    manifold_dim = 6        # 6-DOF
    embedded_dim = 7        # [tx, ty, tz, qx, qy, qz, qw]
    id_elem = [0,0,0, 0,0,0,1]  # Identity

    def exp(cls, x):   # 지수 맵: tangent → group
    def log(self):     # 로그 맵: group → tangent
    def inv(self):     # 역원
    def mul(self, other):  # 곱셈
    def act(self, p):  # 점에 대한 action
```

**CUDA 백엔드**: `lietorch_backends` 확장으로 GPU 가속

### 4.6 dpvo/altcorr/ - Correlation 연산

```python
# correlation.py
class CorrLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2, coords, ii, jj, radius, dropout):
        corr = cuda_corr.forward(fmap1, fmap2, coords, ii, jj, radius)
        return corr

    @staticmethod
    def backward(ctx, grad):
        # Dropout으로 gradient 효율화
        fmap1_grad, fmap2_grad = cuda_corr.backward(...)
        return fmap1_grad, fmap2_grad, None, None, None, None, None
```

**Patchify 연산**: Bilinear interpolation으로 패치 추출
```python
def patchify(net, coords, radius, mode='bilinear'):
    patches = PatchLayer.apply(net, coords, radius)
    # Bilinear interpolation
    x00 = (1-dy) * (1-dx) * patches[...,:d,:d]
    x01 = (1-dy) * (  dx) * patches[...,:d,1:]
    x10 = (  dy) * (1-dx) * patches[...,1:,:d]
    x11 = (  dy) * (  dx) * patches[...,1:,1:]
    return x00 + x01 + x10 + x11
```

### 4.7 dpvo/fastba/ - CUDA Bundle Adjustment

```python
# ba.py
def BA(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, M, iterations):
    return cuda_ba.forward(poses.data, patches, intrinsics, target, weight,
                          lmbda, ii, jj, kk, M, t0, t1, iterations, eff_impl)
```

**CUDA 커널 파일**:
- `ba_cuda.cu`: Jacobian 계산, 선형 시스템 구성
- `block_e.cu`: Block elimination (Schur complement)

### 4.8 dpvo/loop_closure/ - Loop Closure (DPV-SLAM)

```python
class LongTermLoopClosure:
    def __init__(self, cfg, patchgraph):
        self.retrieval = RetrievalDBOW()  # DBoW2 기반 검색
        self.imcache = ImageCache()       # 이미지 캐시
        self.detector = KF.DISK.from_pretrained("depth")  # 키포인트 검출
        self.matcher = KF.LightGlue("disk")  # 특징 매칭

    def attempt_loop_closure(self, n):
        # 1. 루프 후보 검색
        cands = self.retrieval.detect_loop(thresh=cfg.LOOP_RETR_THRESH)
        # 2. 3D 키포인트 추정 및 매칭
        # 3. RANSAC으로 Sim(3) 추정
        # 4. Pose Graph Optimization
```

---

## 5. Submodule 분석

### 5.1 Pangolin (시각화)
- **경로**: `Pangolin/`
- **출처**: https://github.com/zachteed/Pangolin.git
- **용도**: 실시간 3D 재구성 시각화
- **특징**: DSO viewer에서 적응, PyTorch 텐서 직접 접근으로 메모리 복사 최소화

### 5.2 DBoW2 (이미지 검색)
- **경로**: `DBoW2/`
- **출처**: https://github.com/lahavlipson/DBoW2.git
- **용도**: Bag of Words 기반 이미지 유사도 검색
- **적용**: Classical loop closure에서 사용

### 5.3 DPViewer
- **경로**: `DPViewer/`
- **구성**: Pangolin + pybind11 wrapper
- **기능**: 실시간 포즈 및 포인트 클라우드 시각화

### 5.4 DPRetrieval
- **경로**: `DPRetrieval/`
- **구성**: DBoW2 + pybind11 wrapper
- **기능**: 이미지 검색을 통한 루프 감지

---

## 6. 시스템 아키텍처 및 Flow

### 6.1 Inference Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Video Stream                        │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  [1] Feature Extraction                                          │
│  ┌──────────────┐   ┌──────────────┐                            │
│  │  fnet (128)  │   │  inet (384)  │                            │
│  │ Instance Norm│   │    No Norm   │                            │
│  └──────┬───────┘   └──────┬───────┘                            │
│         │                   │                                    │
│         ▼                   ▼                                    │
│  Matching Features    Context Features                          │
│  (2-level pyramid)    (per-patch)                               │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  [2] Patch Extraction (Random Sampling)                          │
│  - 64-96 patches per frame                                       │
│  - 3×3 patch size                                                │
│  - Bilinear interpolation                                        │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  [3] Patch Graph Construction                                    │
│  - Add forward edges: old patches → new frame                    │
│  - Add backward edges: new patches → old frames                  │
│  - Edge lifetime: r frames (default 12)                          │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  [4] Update Operator (Recurrent)                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                                                           │   │
│  │   Correlation → 1D Conv → SoftAgg → GRU → Factor Head    │   │
│  │                                                           │   │
│  │   Output: δ (flow revision), Σ (confidence)              │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  [5] Differentiable Bundle Adjustment                            │
│  - 2 Gauss-Newton iterations                                     │
│  - Schur complement for efficiency                               │
│  - Update: camera poses (SE3) + patch depths                     │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  [6] Keyframe Management                                         │
│  - Remove redundant frames (motion < threshold)                  │
│  - Store relative pose for interpolation                         │
│  - Remove old edges outside optimization window                  │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Output: Poses + Point Cloud                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Training Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     TartanAir Dataset                            │
│  - Synthetic data with GT poses & depth                          │
│  - 15-frame sequences                                            │
│  - Flow magnitude: 16-72 pixels                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  VONet.forward()                                                 │
│  - 8 frames for initialization                                   │
│  - 18 update iterations                                          │
│  - Poses fixed for first 1000 steps                              │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Loss Computation                                                │
│                                                                  │
│  L = 10 * L_pose + 0.1 * L_flow                                  │
│                                                                  │
│  L_pose: Relative pose error (SE3 log)                          │
│  L_flow: Reprojection error (min over p×p patch)                │
│                                                                  │
│  Scale alignment: Kabsch-Umeyama algorithm                       │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Optimization                                                    │
│  - AdamW, lr=8e-5                                                │
│  - Linear decay                                                  │
│  - 240k iterations                                               │
│  - Gradient clipping: 10.0                                       │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Loop Closure Flow (DPV-SLAM)

```
┌─────────────────────────────────────────────────────────────────┐
│  [1] Image Retrieval (DBoW2)                                     │
│  - Add frames to vocabulary                                      │
│  - Query for similar frames                                      │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  [2] Keypoint Detection & Matching (DISK + LightGlue)            │
│  - Detect 2048 keypoints per frame                               │
│  - Match across 3-frame window                                   │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  [3] 3D Point Triangulation                                      │
│  - Structure-only BA                                             │
│  - Filter by reprojection error                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  [4] Sim(3) Estimation (RANSAC + Umeyama)                        │
│  - Estimate scale, rotation, translation                         │
│  - Minimum 30 inlier matches                                     │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  [5] Pose Graph Optimization                                     │
│  - Add loop edge with Sim(3) constraint                          │
│  - Optimize all poses jointly                                    │
│  - Rescale depth maps                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 설정 파라미터 (config.py)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `BUFFER_SIZE` | 4096 | 최대 키프레임 수 |
| `PATCHES_PER_FRAME` | 80 | 프레임당 패치 수 |
| `OPTIMIZATION_WINDOW` | 12 | BA 최적화 윈도우 |
| `REMOVAL_WINDOW` | 20 | Edge 제거 윈도우 |
| `PATCH_LIFETIME` | 12 | 패치 연결 범위 |
| `KEYFRAME_INDEX` | 4 | 키프레임 제거 체크 위치 |
| `KEYFRAME_THRESH` | 12.5 | 키프레임 제거 flow 임계값 |
| `MOTION_MODEL` | DAMPED_LINEAR | 모션 모델 |
| `MOTION_DAMPING` | 0.5 | 모션 감쇠 계수 |
| `MIXED_PRECISION` | True | FP16 사용 |
| `LOOP_CLOSURE` | False | 루프 클로저 활성화 |
| `BACKEND_THRESH` | 64.0 | 백엔드 flow 임계값 |
| `GLOBAL_OPT_FREQ` | 15 | 글로벌 최적화 빈도 |

---

## 8. 주요 혁신점 및 기술적 특징

### 8.1 Sparse vs Dense
- Dense flow (DROID) 대신 **희소 패치 매칭** 사용
- 프레임당 64-96개 패치로 충분한 정보 획득
- **랜덤 샘플링**이 SIFT/ORB/Superpoint보다 우수

### 8.2 Recurrent Architecture
- Edge별 hidden state 유지
- 시간적 정보 전파 (1D Convolution)
- 패치/프레임간 정보 공유 (SoftMax Aggregation)

### 8.3 End-to-End Learning
- Differentiable BA로 전체 파이프라인 학습
- Pose + Flow 동시 supervision
- Confidence weight 자동 학습 (outlier rejection)

### 8.4 Constant Runtime
- 카메라 모션과 무관하게 일정한 FPS
- DROID-SLAM: 11-40 FPS (모션에 따라 변동)
- DPVO: 48-60 FPS (거의 일정)

---

## 9. 벤치마크 결과 요약

### TartanAir (ATE)
| Method | ME Avg | MH Avg | Total Avg |
|--------|--------|--------|-----------|
| DROID-VO | 0.58 | 0.58 | 0.58 |
| **DPVO** | **0.21** | **0.21** | **0.21** |

### EuRoC (ATE)
| Method | Average |
|--------|---------|
| DROID-VO | 0.186 |
| **DPVO** | **0.105** |

### TUM-RGBD (ATE)
| Method | Average |
|--------|---------|
| DROID-VO | 0.098 |
| **DPVO** | **0.089** |

---

## 10. 사용 방법

### 10.1 설치
```bash
git clone https://github.com/princeton-vl/DPVO.git --recursive
cd DPVO
conda env create -f environment.yml
conda activate dpvo

# Eigen 설치
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty

# DPVO 설치
pip install .
```

### 10.2 추론
```bash
python demo.py \
    --imagedir=<path_to_images> \
    --calib=<calibration_file> \
    --viz \
    --save_trajectory
```

### 10.3 학습
```bash
python train.py --steps=240000 --lr=0.00008 --name=my_model
```

---

## 11. 결론

DPVO는 **희소 패치 기반 접근법**을 통해 기존 Dense Flow 방식의 한계를 극복했습니다. 핵심 혁신은:

1. **효율성**: 3배 빠른 속도, 1/3 메모리
2. **정확성**: SOTA를 40-64% 능가
3. **안정성**: 일정한 FPS, 카메라 모션 무관
4. **일반화**: 합성 데이터 학습 → 실제 영상 적용

이 시스템은 자율주행, 로봇 네비게이션, AR/VR 등 실시간 Visual Odometry가 필요한 다양한 응용 분야에서 활용될 수 있습니다.

---

## References

1. Teed, Z., Lipson, L., & Deng, J. (2023). Deep Patch Visual Odometry. NeurIPS.
2. Lipson, L., Teed, Z., & Deng, J. (2024). Deep Patch Visual SLAM. ECCV.
3. Teed, Z., & Deng, J. (2021). DROID-SLAM. NeurIPS.
4. Engel, J., Koltun, V., & Cremers, D. (2017). Direct Sparse Odometry. TPAMI.

---

**문서 작성일**: 2026-01-05
**분석 대상 버전**: princeton-vl/DPVO (commit: 859bbbf)
