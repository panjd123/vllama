# vllama 项目指令

类 Ollama 的 vLLM 管理工具，支持动态模型加载、卸载和自动淘汰。

## 核心功能

### 1. 模型聚合与自动启动

- 根据请求的模型名称，自动转发到对应的 vLLM 实例端口
- **自动启动**：API 请求模型时，如果模型未运行，自动启动或唤醒
- 对外暴露 OpenAI 兼容的 API 接口
- `/v1/models` 端点列出所有可用模型（从 HF_HOME/hub 扫描）

### 2. LRU 自动淘汰机制

当 API 请求模型但显存不足时：
1. 按 LRU（最近最少使用）策略选择要淘汰的模型
2. 逐个将 RUNNING 状态的模型休眠（Level 2）
3. ���次休眠后检查显存是否充足
4. 重复直到有足够显存或无可淘汰的模型

### 3. 三级休眠模式

使用 vLLM 的 sleep/wake_up API：

**Level 1** - 轻度休眠：
```bash
curl -X POST 'http://localhost:8000/sleep?level=1'
curl -X POST 'http://localhost:8000/wake_up'
```

**Level 2** - 深度休眠（自动淘汰使用此级别）：
```bash
curl -X POST 'http://localhost:8000/sleep?level=2'

# 唤醒需要三步
curl -X POST 'http://localhost:8000/wake_up?tags=weights'
curl -X POST 'http://localhost:8000/collective_rpc' -H 'Content-Type: application/json' -d '{"method":"reload_weights"}'
curl -X POST 'http://localhost:8000/wake_up?tags=kv_cache'
```

**Level 3** - 完全停止：
- 直接终止 vLLM 进程

### 4. 内存状态管理

**重要**:
- 所有状态存储在**内存**中（不持久化到磁盘）
- vllama 服务器关闭时，所有状态自动清空
- 重启后需要重新启动模型（或等待 API 请求自动启动）

### 5. 进程组管理

- vLLM 子进程与 vllama 父进程在同一进程组
- vllama 关闭时，自动终止所有 vLLM 子进程
- 实现优雅关闭（等待当前请求完成后再退出）

### 6. CLI 命令

```bash
# 服务器管理
vllama serve [--host HOST] [--port PORT]  # 启动服务器
vllama info                                 # 查看服务器信息

# 模型管理
vllama ps                                   # 查看所有实例状态
vllama start MODEL                          # 启动/唤醒模型
vllama stop MODEL                           # 停止模型（清理 ERROR 状态）
vllama sleep MODEL [--level {1,2,3}]       # 休眠模型
vllama wake-up MODEL                        # 唤醒模型（别名：wakeup, wake_up）

# 配置管理
vllama assign MODEL [--devices D] [--gpu-memory M] [--restart]
```

### 7. 显存检测与记录

- 休眠时记录 `memory_delta`（释放的显存量）
- 唤醒前检查显存是否足够容纳 `memory_delta`
- 新启动模型时不预先检查显存，让 vLLM 自行处理

### 8. YAML 配置

`~/.vllama/models.yaml`:

```yaml
Qwen/Qwen3-0.6B:
  gpu_memory_utilization: 0.85    # 自动计算或配置的值
  max_model_len: 32768            # 限制上下文长度减少显存
  devices: [0]                    # 使用的 GPU
  tensor_parallel_size: 1         # 多卡并行
  trust_remote_code: false
  dtype: auto
  extra_args: {}
```

### 9. 日志系统

- vLLM 进程输出重定向到 `~/.vllama/logs/{model_name}_{port}.log`
- 启动模型时显示日志文件路径
- 方便调试和问题排查

## 技术细节

### 端口分配
- vllama 服务器默认端口：33258
- vLLM 实例端口范围：33300-34300

### 自动淘汰实现
- `instance.py:evict_models_for_memory()` - 淘汰逻辑
- `instance.py:ensure_instance_running()` - 集成淘汰和启动
- `server.py:_forward_request()` - API 自动调用

### 进程管理
- 子进程不使用 `start_new_session=True`
- shutdown 事件处理器调用 `cleanup_all_instances()`
- 信号处理器（SIGTERM/SIGINT）触发优雅关闭

### 状态管理
- `StateManager` 纯内存实现，无文件 I/O
- 构造函数：`StateManager()`（无参数）
- 关闭时调用 `clear_state()` 清空所有状态

## API 端点

- `GET /health` - 健康检查
- `GET /v1/models` - 列出所有模型
- `POST /v1/chat/completions` - 聊天补全
- `POST /v1/completions` - 文本补全
- `POST /v1/embeddings` - 生成嵌入
- `POST /v1/rerank` - 重排序
- `POST /v1/score` - 评分

## 测试

使用小模型进行测试：
- **Chat 模型**: Qwen/Qwen3-0.6B
- **Embedding 模型**: BAAI/bge-m3
- **Rerank 模型**: tomaarsen/Qwen3-Reranker-0.6B-seq-cls

确保不同 API 端点都能正确转发。

## 配置参数

```python
class VllamaConfig(BaseModel):
    port: int = 33258                      # vllama 监听端口
    vllm_port_start: int = 33300          # vLLM 实例起始端口
    vllm_port_end: int = 34300            # vLLM 实例结束端口
    unload_timeout: int = 1800            # 自动卸载超时（秒）
    unload_mode: int = 2                  # 卸载模式 (1/2/3)
    config_dir: Path                      # 配置目录 (~/.vllama)
    transformers_cache: Path              # 模型缓存目录
```

## 关键行为

1. **API 请求流程**：
   - 请求到达 → 查找模型 → `ensure_instance_running()`
   - 如果模型未运行 → 检查显存 → 淘汰旧模型（如需）→ 启动
   - 转发请求到 vLLM 实例 → 返回结果

2. **自动淘汰流程**：
   - 估算所需显存 → 检查当前可用显存
   - 如果不足 → 获取所有 RUNNING 实例 → 按 `last_request_time` 排序
   - 逐个休眠（Level 2）→ 检查显存 → 重复直到足够

3. **关闭行为**：
   - 收到 SIGTERM/SIGINT → 触发 shutdown 事件
   - 停止调度器 → 终止所有 vLLM 子进程 → 清空状态 → 退出
