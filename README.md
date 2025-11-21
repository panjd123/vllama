# vllama

类 Ollama 的 vLLM 管理工具，支持动态模型加载、卸载和自动淘汰。

## ✨ 核心特性

- **自动启动** - API 请求时自动启动/唤醒模型
- **LRU 自动淘汰** - 显存不足时自动休眠最久未使用的模型
- **三级休眠模式** - L1(轻度)/L2(深度)/L3(完全停止)
- **进程组管理** - 服务器关闭时自动清理所有子进程
- **内存存储** - 所有状态存于内存，重启后清空
- **OpenAI 兼容** - 标准 OpenAI API 接口
- **多 GPU 支持** - 灵活配置 GPU 设备和内存

## 📦 安装

```bash
# 克隆仓库
git clone <repository-url>
cd vllama

# 使用 uv 安装（推荐）
uv sync

# 或使用 pip
pip install -e .
```

## 🚀 快速开始

### 启动服务器

```bash
vllama serve --port 33258
```

### 使用 API（自动启动模型）

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:33258/v1",
    api_key="not-needed"
)

# 直接使用模型，vllama 会自动启动
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### 管理模型

```bash
# 查看服务器信息
vllama info

# 查看所有模型实例
vllama ps

# 手动启动模型
vllama start "Qwen/Qwen3-0.6B"

# 手动停止模型
vllama stop "Qwen/Qwen3-0.6B"

# 休眠模型
vllama sleep "Qwen/Qwen3-0.6B" --level 2

# 配置模型
vllama assign "Qwen/Qwen3-0.6B" --gpu-memory 0.85 --devices 0
```

## 🎯 自动淘汰机制

当显存不足时，vllama 会自动：
1. 按 LRU（最近最少使用）策略选择要淘汰的模型
2. 逐个将运行中的模型休眠（Level 2）
3. 释放显存后启动新请求的模型

**示例场景**：
```
8GB 显存，每个模型需要 4GB

请求 Model A → 启动 (使用 4GB)
请求 Model B → 启动 (使用 8GB，显存满)
请求 Model C → 自动淘汰 Model A → 启动 Model C

最终: Model A (sleeping), Model B (running), Model C (running)
```

## ⚙️ 配置

### 环境变量

```bash
export TRANSFORMERS_CACHE=/path/to/models  # 模型缓存目录
```

### 模型配置文件

`~/.vllama/models.yaml`:

```yaml
Qwen/Qwen3-0.6B:
  gpu_memory_utilization: 0.85
  max_model_len: 32768
  devices: [0]
  tensor_parallel_size: 1
```

### 三级休眠模式

| 级别 | 释放内存 | 恢复速度 | 用途 |
|-----|---------|---------|-----|
| **L1** | 部分 KV cache | 最快 (秒级) | 短期空闲 |
| **L2** | Weights + KV cache | 中等 (10-30秒) | 自动淘汰 |
| **L3** | 完全停止进程 | 最慢 (需重启) | 长期不用 |

## 📝 CLI 命令

### vllama serve
```bash
vllama serve [--host HOST] [--port PORT] [--log-level LEVEL]
```
启动 vllama 服务器（默认: `0.0.0.0:33258`）

### vllama info
```bash
vllama info
```
查看服务器信息（端口、配置、健康状态等）

### vllama ps
```bash
vllama ps
```
查看所有模型实例状态和最后访问时间

### vllama start / stop
```bash
vllama start MODEL    # 启动或唤醒模型
vllama stop MODEL     # 停止模型（清理 ERROR/STARTING 状态）
```

### vllama sleep
```bash
vllama sleep MODEL [--level {1,2,3}]
```
休眠模型（默认 Level 2）

### vllama assign
```bash
vllama assign MODEL \
  [--devices DEVICES] \
  [--gpu-memory RATIO] \
  [--restart]
```
配置模型参数并可选重启

## 🔌 API 端点

| 端点 | 说明 |
|-----|------|
| `GET /health` | 健康检查 |
| `GET /v1/models` | 列出所有模型 |
| `POST /v1/chat/completions` | 聊天补全 |
| `POST /v1/completions` | 文本补全 |
| `POST /v1/embeddings` | 生成嵌入 |
| `POST /v1/rerank` | 重排序 |
| `POST /v1/score` | 评分 |

## 🛑 停止服务器

### 方法 1: 使用 PID
```bash
# 查找进程
ps aux | grep "vllama.*serve"

# 停止
kill <PID>
```

### 方法 2: 快捷方式
```bash
kill $(pgrep -f "vllama.*serve")
```

**重要**: 停止 vllama 服务器会：
- ✅ 自动终止所有 vLLM 子进程
- ✅ 清空所有内存中的状态
- ✅ 优雅关闭（等待当前请求完成）

## 📚 常见问题

### Q: 显存不足导致模型启动失败？

**A**: vllama 会自动淘汰旧模型。如果仍失败：
```bash
# 降低内存使用率
vllama assign "MODEL" --gpu-memory 0.7

# 限制上下文长度
# 编辑 ~/.vllama/models.yaml
MODEL:
  max_model_len: 16384
```

### Q: 如何查看模型日志？

**A**: 日志文件位于 `~/.vllama/logs/`
```bash
# 启动模型时会显示日志路径
vllama start "MODEL"
# Logs: /home/user/.vllama/logs/MODEL_33300.log

# 实时查看
tail -f ~/.vllama/logs/MODEL_33300.log
```

### Q: 重启 vllama 后模型状态丢失？

**A**: 这是设计行为。状态存储在内存中，重启后自动清空。模型会在下次 API 请求时自动启动。

### Q: 如何使用多 GPU？

**A**: 使用 `assign` 命令配置：
```bash
# 模型 A 使用 GPU 0
vllama assign "ModelA" --devices 0

# 模型 B 使用 GPU 1
vllama assign "ModelB" --devices 1

# 模型 C 使用多卡并行
vllama assign "ModelC" --devices 0,1
```

## 🏗️ 项目结构

```
vllama/
├── vllama/
│   ├── cli.py           # CLI 命令
│   ├── config.py        # 配置管理
│   ├── gpu.py           # GPU 监控
│   ├── instance.py      # 实例管理（自动淘汰）
│   ├── models.py        # 模型信息
│   ├── scheduler.py     # 自动卸载调度
│   ├── server.py        # FastAPI 服务器
│   ├── state.py         # 内存状态管理
│   └── yaml_manager.py  # YAML 配置
├── tests/               # 单元测试
├── CLAUDE.md           # 项目指令
└── README.md           # 本文档
```

## 🧪 开发

```bash
# 运行测试
pytest

# 代码覆盖率
pytest --cov=vllama

# 特定测试
pytest tests/test_vllama.py::TestConfig
```

## 📄 许可证

MIT License

## 🙏 致谢

基于以下优秀开源项目：
- [vLLM](https://github.com/vllm-project/vllm) - 高性能 LLM 推理引擎
- [FastAPI](https://fastapi.tiangolo.com/) - 现代 Web 框架
- [Typer](https://typer.tiangolo.com/) - CLI 框架
