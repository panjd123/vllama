# vllama

**像 Ollama 一样简单地使用 vLLM，并且具有极快的模型切换速度** - 无需手动管理模型，自动启动，开箱即用，高效切换（在预热后，模型切换时间仅需几秒钟！）

```bash
vllama serve

vllama ps/list/pull/start/stop

# 你可能需要在第一次启动时配置针对某个模型的启动参数，否则每个模型会用默认参数启动
vllama assign Qwen/Qwen3-30B-A3B-Instruct-2507 --devices 1 --gpu-memory-utilization 0.93 --max-model-len 32768 --trust-remote-code --extra enable-prefix-caching=true --restart
```

```python
import openai

client = openai.OpenAI(base_url="http://localhost:33258/v1", api_key="not-needed")

# 直接使用，无需预先加载模型
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Hello!"}]
)

# 无感切换到另一个模型
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## 💡 为什么选择 vllama？

| 传统 vLLM | ✨ vllama |
|----------|----------|
| 每个模型需要手动启动独立进程 | **自动启动** - API 请求时自动加载模型 |
| 需要记住每个模型的端口 | **统一入口** - 所有模型共用一个端点 |
| 显存不足时需要手动停止其他模型 | **智能切换** - 自动淘汰最久未用的模型 |
| 切换模型需要等待漫长的 vLLM 初始化 | **无感切换** - 借助 vLLM 的 Sleep Mode 在几秒内自动完成切换 |

## ✨ 核心特性

### 🚀 自动启动
无需预先启动模型，API 请求时自动加载。首次请求会等待模型加载，后续请求直接使用。

### 🔄 无感切换
显存不足时，自动休眠最久未使用的模型（休眠和唤醒的代价仅为几秒），为新模型腾出空间。整个过程完全自动，无需人工干预。

> 该功能是借助 vLLM 的 [Sleep Mode](https://docs.vllm.ai/en/latest/features/sleep_mode/) 实现的，你可以查看 vLLM 的官方 blog 了解更多细节：[Zero-Reload Model Switching with vLLM Sleep Mode](https://blog.vllm.ai/2025/10/26/sleep-mode.html)

### 🧠 智能管理
- **LRU 淘汰策略** - 优先淘汰最少使用的模型
- **三级休眠** - 从秒级恢复到完全停止的灵活策略
- **自动优化** - 智能计算显存利用率和参数
- **多 GPU 支持** - 自动选择最大显存的 GPU

### 🔌 完全兼容
- **OpenAI API** - 直接替换 OpenAI 端点即可使用
- **流式输出** - 支持 SSE 流式响应
- **多种任务** - Chat、Completion、Embedding、Rerank 全支持

## 🚀 快速开始（Docker - 推荐）

```bash
git clone https://github.com/panjd123/vllama.git && cd vllama

docker compose up -d

curl http://localhost:33258/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 使用预构建镜像

```bash
# 直接拉取镜像
docker pull panjd123/vllama:latest

docker compose up -d
```

### Docker 配置

主机的模型缓存会自动挂载到容器，无需重复下载：

```yaml
volumes:
  - ${HOME}/.cache/huggingface:/root/.cache/huggingface  # 模型自动共享
  - ./vllama_config:/root/.vllama                        # 配置持久化
```

通过环境变量自定义：

```yaml
environment:
  - VLLAMA_PORT=33258                    # 服务端口
  - VLLAMA_DEFAULT_DEVICE=0              # 默认 GPU（可选，未指定则选择总显存最大的 GPU）
  - VLLAMA_UNLOAD_TIMEOUT=1800           # 空闲多久后自动卸载
  - HF_HOME=/root/.cache/huggingface     # 模型缓存位置
```

## 🎬 实战演示

### 场景：8GB 显存，自动切换三个模型

```python
import openai

client = openai.OpenAI(base_url="http://localhost:33258/v1", api_key="not-needed")

# 1️⃣ 请求 Model A（自动启动，使用 4GB）
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "你好"}]
)

# 2️⃣ 请求 Model B（自动启动使用 8GB，显存满）
response = client.chat.completions.create(
    model="BAAI/bge-m3",
    messages=[{"role": "user", "content": "Hello"}]
)

# 3️⃣ 请求 Model C
#    🔄 自动淘汰 Model A（最久未用）
#    ⏳ 等待几秒释放显存
#    🚀 启动 Model C
response = client.chat.completions.create(
    model="google/gemma-3-270m-it",
    messages=[{"role": "user", "content": "Hi"}]
)

# 4️⃣ 再次请求 Model A
#    🔄 自动唤醒 Model A（几秒内恢复）
#    ✨ 无需重新加载，快速恢复
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "再见"}]
)
```

**整个过程完全自动，无需任何手动操作！**

## 📦 本地安装

```bash
git clone https://github.com/panjd123/vllama.git && cd vllama

uv sync

# pip install -e .

vllama serve
```

## 📚 深入了解

### 环境变量配置

```bash
# 服务器配置
export VLLAMA_HOST=0.0.0.0                    # 监听地址
export VLLAMA_PORT=33258                      # 服务端口
export VLLAMA_DEFAULT_DEVICE=0                # 默认 GPU ID

# 自动卸载配置
export VLLAMA_UNLOAD_TIMEOUT=1800             # 空闲 30 分钟后自动卸载
export VLLAMA_UNLOAD_MODE=2                   # 卸载级别 (1/2/3)

# 模型缓存
export HF_HOME=/path/to/huggingface           # 模型存储位置
```

### 模型配置文件

编辑 `~/.vllama/models.yaml` 为特定模型配置参数：

```yaml
Qwen/Qwen3-0.6B:
  gpu_memory_utilization: 0.85    # GPU 显存使用率
  max_model_len: 32768            # 最大上下文长度
  devices: [0]                    # 使用的 GPU
  tensor_parallel_size: 1         # 张量并行大小
  dtype: auto                     # 数据类型
  trust_remote_code: false        # 是否信任远程代码
  auto_start: true                # 服务器启动时自动加载
```

修改后重启模型应用配置：
```bash
vllama restart Qwen/Qwen3-0.6B
```

## 🔧 CLI 命令参考

### 服务器管理

```bash
vllama serve              # 启动服务器
vllama info               # 查看服务器信息
```

### 模型管理

```bash
vllama list                   # 列出可用模型
vllama pull MODEL             # 下载模型
vllama ps                     # 查看运行状态
vllama start MODEL            # 启动模型
vllama stop MODEL             # 停止模型
vllama restart MODEL          # 重启模型
vllama sleep MODEL [-l 2]     # 休眠模型
vllama wake-up MODEL          # 唤醒模型（和 vllama start 相同）

# 预热模型 - 预先加载模型以加快首次访问速度
vllama warm-up MODEL1 MODEL2          # 立即预热指定模型
vllama warm-up MODEL --save           # 保存到配置，服务器启动时自动预热
vllama warm-up --show                 # 查看自动预热列表
vllama warm-up --remove MODEL         # 从自动预热列表中移除
vllama warm-up --clear                # 清空自动预热列表

# 交互式聊天
vllama run MODEL                      # 启动交互式聊天会话
vllama run MODEL --system "prompt"    # 使用自定义系统提示词
```

### 配置管理

```bash
vllama assign MODEL [OPTIONS]

选项：
  --gpu-memory, -m FLOAT          GPU 显存使用率 (0.1-1.0)
  --devices, -d TEXT              GPU 设备 ID (例如: "0,1")
  --max-model-len, -l INT         最大上下文长度
  --tensor-parallel-size, -t INT  张量并行大小
  --dtype TEXT                    数据类型 (auto/float16/bfloat16/float32)
  --trust-remote-code             启用信任远程代码
  --no-trust-remote-code          禁用信任远程代码
  --auto-start                    启用服务器启动时自动加载
  --no-auto-start                 禁用服务器启动时自动加载
  --extra-args, -e TEXT           额外参数 (key=value，可多次使用)
  --clear-extra-args              清空所有额外参数
  --restart, -r                   应用配置后重启模型
  --show, -s                      显示当前配置
```

示例：
```bash
# 基本配置
vllama assign MODEL --devices 1 --gpu-memory 0.85

# 启用 trust remote code 和 auto-start
vllama assign MODEL --trust-remote-code --auto-start

# 禁用 auto-start
vllama assign MODEL --no-auto-start

# 配置并重启
vllama assign MODEL --max-model-len 32768 --restart
```

### 模型预热 (Warm-up)

预热功能允许在服务器启动时自动加载常用模型，避免首次 API 请求时的等待时间。

**使用场景：**
- 避免首次请求的冷启动延迟
- 自动化部署流程

**配置示例：**

```bash
vllama warm-up Qwen/Qwen3-0.6B BAAI/bge-m3 --save
# 或
# vllama assign Qwen/Qwen3-0.6B --auto-start
# vllama assign BAAI/bge-m3 --auto-start
vllama serve
```

配置保存在 `~/.vllama/models.yaml` 中（作为模型配置的 `auto_start` 字段）：
```yaml
Qwen/Qwen3-0.6B:
  auto_start: true
  # ... 其他配置

BAAI/bge-m3:
  auto_start: true
  # ... 其他配置
```

### 三级休眠模式

vllama 使用 vLLM 的休眠功能实现快速切换：

| 级别 | 释放内存 | 恢复时间 | 默认 |
|-----|---------|---------|----------|
| **L1** | Weights + KV cache，权重会备份到 CPU 内存中 | 秒级 |  |
| **L2** | Weights + KV cache，权重不会备份到 CPU 内存中 | 秒级 | 默认配置 |
| **L3** | 完全停止进程 | 分钟级 | |

## 🌐 API 端点

vllama 提供完整的 OpenAI 兼容 API：

| 端点 | 说明 | 示例模型 |
|-----|------|---------|
| `POST /v1/chat/completions` | 聊天补全 | Qwen, Llama, Gemma |
| `POST /v1/completions` | 文本补全 | 任何语言模型 |
| `POST /v1/embeddings` | 生成嵌入 | bge-m3, e5 |
| `POST /v1/rerank` | 重排序 | reranker 模型 |
| `POST /v1/score` | 评分 | 评分模型 |
| `GET /v1/models` | 列出模型 | - |
| `GET /health` | 健康检查 | - |

### 使用示例

**Chat 补全（流式）**
```python
import openai

client = openai.OpenAI(base_url="http://localhost:33258/v1", api_key="not-needed")

stream = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "讲个笑话"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

**Embedding**
```python
response = client.embeddings.create(
    model="BAAI/bge-m3",
    input=["Hello world", "你好世界"]
)

embeddings = [data.embedding for data in response.data]
```

**交互式聊天（命令行）**
```bash
# 启动交互式聊天会话
vllama run Qwen/Qwen3-0.6B

# 使用自定义系统提示词
vllama run Qwen/Qwen3-0.6B --system "你是一个乐于助人的 AI 助手"

# 退出方式：输入 /exit 或按 Ctrl+D 或 Ctrl+C
```

## ❓ 常见问题

<details>
<summary><b>Q: 如何指定默认使用哪张 GPU？</b></summary>

通过环境变量设置：
```bash
export VLLAMA_DEFAULT_DEVICE=1  # 使用 GPU 1
```

或在 docker-compose.yml 中：
```yaml
environment:
  - VLLAMA_DEFAULT_DEVICE=1
```

未设置时，vllama 会自动选择总显存最大的 GPU。
</details>

<details>
<summary><b>Q: 如何在多 GPU 环境下运行不同模型？</b></summary>

```bash
# Model A 在 GPU 0
vllama assign ModelA --devices 0

# Model B 在 GPU 1
vllama assign ModelB --devices 1

# Model C 使用多卡并行（GPU 0,1）
vllama assign ModelC --devices 0,1 --tensor-parallel-size 2
```
</details>

<details>
<summary><b>Q: 显存不足怎么办？</b></summary>

vllama 会自动淘汰旧模型，如果仍然失败：

```bash
# 1. 降低显存使用率
vllama assign MODEL --gpu-memory 0.7 --restart

# 2. 限制上下文长度
vllama assign MODEL --max-model-len 16384 --restart

# 3. 手动释放某个模型
vllama stop MODEL
```
</details>

<details>
<summary><b>Q: 如何查看模型加载日志？</b></summary>

日志位于 `~/.vllama/logs/`：

```bash
# 查看特定模型的日志
tail -f ~/.vllama/logs/Qwen_Qwen3-0.6B_33300.log

# Docker 中
docker compose exec vllama tail -f /root/.vllama/logs/Qwen_Qwen3-0.6B_33300.log
```
</details>

<details>
<summary><b>Q: Docker 容器如何使用主机已下载的模型？</b></summary>

docker-compose.yml 已自动配置卷挂载：

```yaml
volumes:
  - ${HOME}/.cache/huggingface:/root/.cache/huggingface
```

容器会直接使用主机的模型，无需重复下载。
</details>

## 🤝 对比 Ollama

| 特性 | Ollama | vllama |
|-----|--------|--------|
| 推理后端 | llama.cpp (CPU/GPU) | vLLM (GPU only, 更快) |
| 使用方式 | ✅ 一键启动 | ✅ 一键启动 |
| 自动加载 | ✅ | ✅ |
| 模型切换 | ✅ 自动卸载旧模型 | ✅ LRU 智能淘汰 |
| API 兼容 | ✅ OpenAI 兼容 | ✅ OpenAI 兼容 |
| 快速切换 | ✅ 秒级重新加载 | ✅ 秒级快速唤醒 |
| 流式输出 | ✅ | ✅ |
| 适用场景 | 轻量部署 | GPU 推理、高性能批量请求需求 |

**vllama = Ollama 的易用性 + vLLM 的高性能**

## 🙏 致谢

vllama 基于以下优秀开源项目构建：

- [vLLM](https://github.com/vllm-project/vllm) - 高性能 LLM 推理引擎
- [FastAPI](https://fastapi.tiangolo.com/) - 现代 Python Web 框架
- [Typer](https://typer.tiangolo.com/) - 优雅的 CLI 框架

## TODO

- 测试多卡环境（目前仅测试了多卡下分别用单卡）

## 📄 许可证

MIT License - 自由使用、修改和分发
