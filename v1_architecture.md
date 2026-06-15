# vLLM V1 架构详解

## 概述

vLLM V1 采用**多进程架构**，将系统拆分为独立进程，通过 ZMQ socket 和共享内存高效通信。核心设计理念：

1. **简单、模块化、易 hack**：每个组件职责清晰，代码可读性大幅提升
2. **近零 CPU 开销**：GPU 采样、ZMQ IPC、异步执行、busy loop 调度
3. **零配置优化**：chunked prefill、prefix caching 等默认开启
4. **统一架构**：不再区分 prefill/decode 阶段，调度器用统一的 token 预算分配

---

## 一、进程架构

V1 有 **4 种进程类型**：

### 1. API Server 进程

| 属性 | 说明 |
|---|---|
| **数量** | `A`（默认等于 `data_parallel_size`，可通过 `--api-server-count` 覆盖） |
| **源文件** | `vllm/entrypoints/openai/api_server.py` |
| **职责** | 处理 HTTP 请求（OpenAI 兼容 API）、tokenization、多模态数据加载、流式返回结果 |

核心组件：
- **AsyncLLM**（`vllm/v1/engine/async_llm.py`）：异步引擎前端，通过 `EngineCoreClient` 经 ZMQ 与 EngineCore 通信
- **InputProcessor**（`vllm/v1/engine/input_processor.py`）：将用户 prompt（文本/多模态）转换为 `EngineCoreRequest`
- **OutputProcessor**（`vllm/v1/engine/output_processor.py`）：将 `EngineCoreOutput` 转换为用户可见的 `RequestOutput`

### 2. Engine Core 进程

| 属性 | 说明 |
|---|---|
| **数量** | `DP`（默认 1，每个数据并行 rank 一个） |
| **源文件** | `vllm/v1/engine/core.py` |
| **职责** | 运行调度器、管理 KV cache、协调 GPU worker 执行模型 |

核心类：
- **EngineCore**（`core.py:85`）：引擎内循环，包含 Scheduler + Executor
- **EngineCoreProc**（`core.py:776`）：ZMQ 包装器，将 EngineCore 运行在后台进程中，包含 I/O 线程处理序列化/反序列化
- **DPEngineCoreProc**（`core.py:1571`）：数据并行扩展版，增加 wave 同步和弹性 EP 缩放

**Busy Loop**（`core.py:1127`）：

```
while not shutdown:
    1) 轮询输入队列，直到有工作要做
    2) 执行引擎核心步骤 (schedule → execute → update_from_output)
```

**step() 方法**（`core.py:378`）核心迭代周期：

1. **Schedule** — `scheduler.schedule()` 产生 `SchedulerOutput`
2. **Execute** — `model_executor.execute_model(scheduler_output)` 分发到 GPU workers
3. **Grammar Bitmask** — `scheduler.get_grammar_bitmask()` 处理结构化输出
4. **Sample** — `model_executor.sample_tokens()` 生成 token IDs
5. **Update** — `scheduler.update_from_output()` 处理模型输出，产生 `EngineCoreOutputs`

### 3. GPU Worker 进程

| 属性 | 说明 |
|---|---|
| **数量** | `N` = DP × PP × TP（每个 GPU 一个） |
| **源文件** | `vllm/v1/worker/gpu_worker.py` |
| **职责** | 加载模型权重、执行 GPU 前向传播、管理 GPU 内存 |

核心组件：
- **Worker**（`gpu_worker.py:105`）：初始化 CUDA 设备、分布式环境、模型
- **GPUModelRunner**（`gpu_model_runner.py:102`）：实际运行模型前向传播，管理 CUDA graph、采样等

### 4. DP Coordinator 进程（条件性）

| 属性 | 说明 |
|---|---|
| **数量** | 仅当 `data_parallel_size > 1` 时存在 1 个 |
| **源文件** | `vllm/v1/engine/coordinator.py` |
| **职责** | 跨 DP rank 的负载均衡、wave 协调、负载统计发布 |

核心类：
- **DPCoordinator**（`coordinator.py:22`）：启动协调器进程
- **DPCoordinatorProc**（`coordinator.py:118`）：协调器进程实现，使用三个 ZMQ socket：
  - `publish_front`（XPUB）— 向 API Server 发布统计和 wave 状态
  - `output_back`（PULL）— 接收 Engine Core 的统计和 wave 通知
  - `publish_back`（XPUB）— 向 Engine Core 广播 `START_DP_WAVE`

### 进程数量示例

| 部署场景 | API Server | Engine Core | GPU Worker | DP Coordinator | **总计** |
|---|---|---|---|---|---|
| 单 GPU | 1 | 1 | 1 | 0 | **3** |
| TP=4（4 GPU） | 1 | 1 | 4 | 0 | **6** |
| TP=2, DP=4（8 GPU） | 4 | 4 | 8 | 1 | **17** |

---

## 二、核心组件详解

### 前端组件（API Server 进程内）

#### 1. InputProcessor — 输入处理器

**文件**：`vllm/v1/engine/input_processor.py`

**职责**：将原始用户输入转换为 EngineCore 可处理的 `EngineCoreRequest`

```
PromptType / ProcessorInputs  →  InputProcessor.process_inputs()  →  EngineCoreRequest
```

主要处理：
- Tokenization（文本 → token IDs）
- 多模态数据加载与特征提取
- LoRA 验证
- 请求 ID 随机化（内部唯一性）
- 数据并行 rank 分配

#### 2. OutputProcessor — 输出处理器

**文件**：`vllm/v1/engine/output_processor.py`

**职责**：将 `EngineCoreOutput` 转换为用户可见的 `RequestOutput`

```
EngineCoreOutput  →  OutputProcessor.process_outputs()  →  RequestOutput
```

主要处理：
- 增量 detokenization（token IDs → 文本）
- Logprobs 处理
- 流式输出聚合
- 统计信息收集

内部组件：
- **IncrementalDetokenizer**（`vllm/v1/engine/detokenizer.py`）：增量解码
- **LogprobsProcessor**（`vllm/v1/engine/logprobs.py`）：logprobs 处理
- **RequestOutputCollector**：每请求异步队列

#### 3. EngineCoreClient — 引擎核心客户端

**文件**：`vllm/v1/engine/core_client.py`

**职责**：前端与 EngineCore 之间的通信抽象层

| 客户端类型 | 用途 |
|---|---|
| `InprocClient` | 进程内直接调用（同步 LLMEngine，无多进程） |
| `SyncMPClient` | ZMQ 同步多进程客户端 |
| `AsyncMPClient` | ZMQ 异步多进程客户端（AsyncLLM 使用） |
| `DPAsyncMPClient` | 数据并行 + 外部负载均衡 |
| `DPLBAsyncMPClient` | 数据并行 + 内部负载均衡 |

工厂方法：`EngineCoreClient.make_client()` 根据 `multiprocess_mode` 和 `asyncio_mode` 选择合适的客户端子类。

---

### 后端组件（Engine Core 进程内）

#### 4. Scheduler — 调度器

**文件**：`vllm/v1/core/sched/scheduler.py`

**接口**：`vllm/v1/core/sched/interface.py`（`SchedulerInterface` 抽象基类）

**职责**：决定每一步处理哪些请求、分配多少 token 预算

核心方法：
- `schedule()` → 产生 `SchedulerOutput`（哪些请求、多少 tokens）
- `update_from_output()` → 处理模型输出，更新请求状态
- `_preempt_request()` → 抢占运行中的请求，释放 KV cache 块

内部维护：
- **RequestQueue**（`request_queue.py`）：FCFS 或优先级排序的等待队列
- **running 列表**：当前运行的请求
- **KVCacheManager**：KV cache 块分配
- **EncoderCacheManager**：多模态编码器缓存
- **StructuredOutputManager**：结构化输出（grammar）

子类：**AsyncScheduler**（`async_scheduler.py`）— 支持投机解码

**统一调度设计**：V1 调度器用简单的 `{request_id: num_tokens}` 字典动态分配每个请求的 token 预算，`num_tokens` 可以是：
- 新请求的全部 prompt tokens（完整 prefill）
- 1（decode 请求）
- 中间值（chunked prefill、prefix caching、speculative decoding 等）

#### 5. KVCacheManager — KV Cache 管理器

**文件**：`vllm/v1/core/kv_cache_manager.py`

**职责**：管理 KV cache 块的分配、释放和 prefix caching

核心方法：
- `get_computed_blocks()` — 查找 prefix cache 命中
- `allocate_slots()` — 为新 tokens 分配 KV cache 块
- `free()` — 请求完成或抢占时释放块

内部结构：
- **KVCacheCoordinator**（`kv_cache_coordinator.py`）— 协调多种 KV cache 类型
- **BlockPool**（`block_pool.py`）— 空闲块池和块哈希缓存
- **SingleTypeKVCacheManager** 子类（`single_type_kv_cache_manager.py`）：
  - `FullAttentionManager` — 全注意力
  - `SlidingWindowManager` — 滑动窗口
  - `MambaManager` — Mamba SSM
  - `ChunkedLocalAttentionManager` — 分块局部注意力

---

### 执行组件

#### 6. Executor — 执行器

**文件**：`vllm/v1/executor/abstract.py`（抽象基类）

**职责**：编排跨多个设备/进程的模型执行

| 执行器 | 文件 | 用途 |
|---|---|---|
| `UniProcExecutor` | `uniproc_executor.py` | 单进程，单 GPU / 调试 |
| `MultiprocExecutor` | `multiproc_executor.py` | 多进程，**GPU 部署默认** |
| `RayDistributedExecutor` | `ray_executor.py` | Ray 分布式执行 |
| `ExecutorWithExternalLauncher` | `uniproc_executor.py` | torchrun 兼容启动器 |

关键方法：
- `execute_model(scheduler_output)` — 分发调度输出到 workers
- `collective_rpc()` — 统一 RPC 调用（支持非阻塞模式返回 `Future`）
- `determine_available_memory()` — GPU 内存分析
- `initialize_from_config()` — 初始化 KV cache
- `sleep()` / `wake_up()` — 模型权重卸载到 CPU 和恢复（RL 训练场景）
- `check_health()` — 健康检查
- `shutdown()` — 关闭

选择逻辑：`Executor.get_class()` 根据 `distributed_executor_backend` 配置选择执行器类。

#### 7. Worker — 工作进程

**文件**：`vllm/v1/worker/gpu_worker.py`

**职责**：代表一个 GPU 进程，初始化设备、加载模型、执行推理

核心方法：
- `init_device()` — 设置 CUDA 设备、分布式环境，创建 `GPUModelRunner`
- `load_model()` — 加载模型权重
- `determine_available_memory()` — GPU 内存分析
- `execute_model()` — 委托给 `model_runner.execute_model()`

包装器：**WorkerWrapperBase**（`worker_base.py`）— 延迟初始化 Worker，处理生命周期

其他变体：`CPUWorker`（`cpu_worker.py`）、`XPUWorker`（`xpu_worker.py`）

#### 8. GPUModelRunner — 模型运行器

**文件**：`vllm/v1/worker/gpu_model_runner.py`

**职责**：最底层组件，实际运行模型前向传播

主要功能：
- 准备输入张量（从 `SchedulerOutput` 构建 `InputBatch`）
- 管理 KV cache 张量（GPU 端）
- CUDA graph 捕获与回放
- 注意力后端选择
- Token 采样
- 投机解码支持

内部组件：
- **InputBatch**（`gpu/input_batch.py`）— GPU 常驻批状态（token IDs、采样参数、block tables 等）
- **BlockTables** — 每请求 KV cache 块表
- **ModelCudaGraphManager** — CUDA graph 管理
- **Sampler** — Token 采样器
- **KVConnector** — 分布式推理的 KV 传输

其他变体：`CPUModelRunner`（`cpu_model_runner.py`）、`XPUModelRunner`（`xpu_model_runner.py`）

---

### 辅助子系统

| 子系统 | 目录 | 说明 |
|---|---|---|
| **Attention** | `vllm/v1/attention/` | FlashInfer、FlashAttention、Triton、ROCm、MLA、Mamba 等后端 |
| **Sampling** | `vllm/v1/sample/` | GPU 采样、logits 处理器（temperature、top-k/p 等） |
| **Speculative Decoding** | `vllm/v1/spec_decode/` | EAGLE、Medusa、N-gram、Draft Model、Suffix Decoding |
| **Structured Output** | `vllm/v1/structured_output/` | xgrammar、guidance、outlines 语法后端 |
| **KV Offloading** | `vllm/v1/kv_offload/` | LRU、ARC、Reuse 缓存淘汰策略 |

---

## 三、组件所有权层级

```
AsyncLLM / LLMEngine（前端）
  ├── InputProcessor
  ├── OutputProcessor
  │     ├── IncrementalDetokenizer
  │     └── LogprobsProcessor
  └── EngineCoreClient（InprocClient / AsyncMPClient / ...）
        │
        ▼  ZMQ socket
EngineCore（核心进程）
  ├── Scheduler
  │     ├── KVCacheManager
  │     │     ├── KVCacheCoordinator
  │     │     └── BlockPool
  │     ├── EncoderCacheManager
  │     └── StructuredOutputManager
  └── Executor（UniProc / Multiproc / Ray）
        └── Worker(s)（每 GPU 一个进程）
              └── GPUModelRunner
                    ├── nn.Module（模型）
                    ├── InputBatch
                    ├── BlockTables
                    ├── Sampler
                    └── ModelCudaGraphManager
```

---

## 四、进程间通信拓扑

```
API Server ←──ZMQ──→ Engine Core ←──Shared Memory──→ GPU Worker(s)
    ↑                     ↑
    │                     │
    └── ZMQ (XPUB) ──→ DP Coordinator ←── ZMQ (PULL) ──┘
                    (仅 DP>1 时存在)
```

| 通信路径 | 协议 | 数据 |
|---|---|---|
| API Server → Engine Core | ZMQ PUSH/PULL + msgpack | `EngineCoreRequest` |
| Engine Core → API Server | ZMQ PUSH/PULL + msgpack | `EngineCoreOutputs` |
| Engine Core → GPU Workers | 共享内存 `MessageQueue` | `SchedulerOutput` |
| GPU Workers → Engine Core | 响应消息队列 | `ModelRunnerOutput` |
| Engine Core → DP Coordinator | ZMQ PULL | 负载统计、wave 通知 |
| DP Coordinator → Engine Cores | ZMQ XPUB | `START_DP_WAVE` |
| DP Coordinator → API Servers | ZMQ XPUB | 负载统计、wave 状态 |

---

## 五、端到端数据流

```
用户请求
    │
    ▼
[AsyncLLM / LLMEngine]  ← 前端进程
    ├─ InputProcessor: Prompt → EngineCoreRequest
    ├─ OutputProcessor: EngineCoreOutput → RequestOutput
    │
    ▼  (ZMQ socket)
[EngineCore]  ← 核心进程
    ├─ Scheduler.schedule() → SchedulerOutput
    ├─ Executor.execute_model() → ModelRunnerOutput
    │       │
    │       ▼
    │   Worker (每 GPU 一个进程)
    │       └─ GPUModelRunner.execute_model()
    │
    ├─ Scheduler.update_from_output() → EngineCoreOutputs
    │
    ▼  (ZMQ socket)
[AsyncLLM / LLMEngine]  ← 前端进程
    └─ OutputProcessor → RequestOutput
    │
    ▼
用户响应（流式）
```

**逐步流程**：

1. **前端**（`LLMEngine` 或 `AsyncLLM`）接收用户请求（prompt + 采样参数）
2. **InputProcessor** 对 prompt 进行 tokenization、验证参数、分配唯一内部请求 ID，产生 `EngineCoreRequest`
3. **EngineCoreClient** 将 `EngineCoreRequest` 发送到 **EngineCore**（进程内或经 ZMQ 到后台进程）
4. **EngineCore** 运行 busy loop：
   - **Scheduler.schedule()** 检查等待/运行中的请求，通过 **KVCacheManager** 分配 KV cache 块，产生 `SchedulerOutput`
   - **Executor.execute_model(scheduler_output)** 分发到 **Worker(s)**，委托给 **GPUModelRunner.execute_model()** 运行前向传播，产生 `ModelRunnerOutput`
   - **Scheduler.update_from_output()** 处理模型输出，更新请求状态，产生 `EngineCoreOutputs`
5. **EngineCoreClient** 将 `EngineCoreOutputs` 返回给前端
6. **OutputProcessor** 将 `EngineCoreOutput` 转换为用户可见的 `RequestOutput`（detokenize、logprobs、流式聚合）

---

## 六、数据并行部署模式

| 模式 | 标志 | 说明 |
|---|---|---|
| **内部负载均衡** | 默认（DP>1 时） | API Server 在所有 DP 引擎间内部均衡请求 |
| **外部负载均衡** | `--data-parallel-external-lb` | 每个 DP rank 独立运行，外部负载均衡器路由请求 |
| **混合负载均衡** | `--data-parallel-hybrid-lb` | 每节点 API Server 处理本地 DP rank，上游 LB 跨节点分发 |

内部负载均衡模式下，DP Coordinator 收集各引擎的队列统计并发布给 API Server，API Server 据此将请求路由到负载最低的引擎。

---

## 七、V1 目录结构

```
vllm/v1/
├── __init__.py
├── utils.py
├── serial_utils.py
├── request.py
├── outputs.py
├── cudagraph_dispatcher.py
├── kv_cache_interface.py
│
├── engine/                    # 前端引擎层（面向 API）
│   ├── __init__.py            # EngineCoreRequest, EngineCoreOutput, FinishReason 等
│   ├── llm_engine.py          # LLMEngine（同步）
│   ├── async_llm.py           # AsyncLLM（异步）
│   ├── core.py                # EngineCore, EngineCoreProc, DPEngineCoreProc
│   ├── core_client.py         # EngineCoreClient（Inproc/SyncMP/AsyncMP）
│   ├── input_processor.py     # InputProcessor
│   ├── output_processor.py    # OutputProcessor, RequestOutputCollector
│   ├── detokenizer.py         # IncrementalDetokenizer
│   ├── logprobs.py            # LogprobsProcessor
│   ├── parallel_sampling.py   # ParentRequest（n>1 fan-out）
│   ├── coordinator.py         # DPCoordinator
│   ├── utils.py               # ZMQ 辅助工具
│   └── exceptions.py          # EngineDeadError, EngineGenerateError
│
├── core/                      # 核心调度 & KV cache 管理
│   ├── sched/
│   │   ├── interface.py       # SchedulerInterface ABC
│   │   ├── scheduler.py       # Scheduler（主实现）
│   │   ├── async_scheduler.py # AsyncScheduler
│   │   ├── output.py          # SchedulerOutput
│   │   ├── request_queue.py   # RequestQueue（优先级支持）
│   │   └── utils.py           # 调度工具函数
│   ├── kv_cache_manager.py    # KVCacheManager
│   ├── kv_cache_utils.py      # KV cache 配置生成
│   ├── kv_cache_coordinator.py # KV cache 协调器
│   ├── kv_cache_metrics.py    # KV cache 指标
│   ├── block_pool.py          # 空闲块池
│   ├── encoder_cache_manager.py # 编码器缓存
│   └── single_type_kv_cache_manager.py # 单类型 KV cache 管理器
│
├── executor/                  # 模型执行
│   ├── __init__.py
│   ├── abstract.py            # Executor ABC
│   ├── uniproc_executor.py    # UniProcExecutor
│   ├── multiproc_executor.py  # MultiprocExecutor
│   ├── ray_executor.py        # RayDistributedExecutor
│   └── ray_utils.py           # Ray 工具
│
├── worker/                    # Worker 进程
│   ├── worker_base.py         # WorkerBase, WorkerWrapperBase
│   ├── gpu_worker.py          # GPU Worker
│   ├── cpu_worker.py          # CPU Worker
│   ├── xpu_worker.py          # XPU Worker
│   └── gpu/
│       ├── model_runner.py    # GPUModelRunner
│       ├── input_batch.py     # InputBatch
│       └── sample/            # GPU 采样子组件
│
├── attention/                 # 注意力后端 & 算子
│   ├── backend.py             # 注意力后端注册与选择
│   └── backends/              # FlashInfer, FlashAttention, Triton, ROCm, MLA, Mamba 等
│
├── sample/                    # 采样逻辑
│   ├── sampler.py             # 主采样入口
│   ├── logits_processor/      # Logits 处理器
│   └── ops/                   # 采样核实现
│
├── spec_decode/               # 投机解码
│   ├── eagle.py               # EAGLE
│   ├── medusa.py              # Medusa
│   ├── ngram_proposer.py      # N-gram
│   ├── draft_model.py         # Draft Model
│   └── suffix_decoding.py     # Suffix Decoding
│
├── structured_output/         # 结构化输出 / Grammar
│   ├── backend_xgrammar.py
│   ├── backend_guidance.py
│   └── backend_outlines.py
│
└── kv_offload/                # KV cache 卸载
    ├── lru_manager.py
    ├── arc_manager.py
    └── reuse_manager.py
```

---

## 八、V0 vs V1 架构对比

| 特性 | V0 | V1 |
|---|---|---|
| **进程架构** | 单体式单进程 | 多进程（API Server + Engine Core + Workers） |
| **引擎** | 单体 LLMEngine | 前后端分离（AsyncLLM + EngineCore） |
| **调度器** | Prefill/Decode 分阶段 | 统一调度（无阶段分离） |
| **调度策略** | FCFS | FCFS + 优先级 |
| **IPC** | Python 队列 / Ray | ZMQ + msgpack |
| **抢占方式** | SWAP（GPU↔CPU 交换） | RECOMPUTE（重计算） |
| **Chunked Prefill** | 条件性启用 | 默认开启 |
| **Prefix Caching** | 可选 | 默认开启 |
| **KV Cache Swap** | 支持 | 已移除 |
| **采样** | CPU | GPU |
| **best_of 采样** | 支持 | 已移除 |
| **Per-Request Logits Processors** | 支持 | 已移除（改为全局） |
| **Sleep/Wake** | 不支持 | 支持（权重卸载到 CPU） |
| **数据并行** | 有限 | 完整支持 + DP Coordinator |
| **CUDA Graph 内存** | 较低 | 较高（更多内存用于捕获） |
