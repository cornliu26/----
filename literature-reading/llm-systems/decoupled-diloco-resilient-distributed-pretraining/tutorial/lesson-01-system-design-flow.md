# 第 1 课：Decoupled DiLoCo 的系统设计流程

## 1. 本课定位

这节课不按论文顺序复述，而是回答一个工程问题：

> 如果我们要把一个巨大的同步预训练任务，改造成能容忍故障、慢节点、异构硬件和临时算力的训练系统，它的流程应该怎么设计？

Decoupled DiLoCo 的答案是：把“训练计算”和“全局同步”拆开。训练计算由多个 learner 独立推进，全局同步由一个轻量 syncer 异步协调。

## 2. 原文术语与中文对照

| Paper term | 中文理解 | 在系统里的作用 |
| --- | --- | --- |
| Decoupled DiLoCo | 解耦版 DiLoCo | 把 learner 训练循环和全局同步器拆开 |
| learner | 学习器 / 本地训练组 | 在一小组 TPU/GPU 上独立做本地训练 |
| syncer | 同步器 | 收集 learner 更新，维护全局参数和 outer optimizer |
| parameter fragments | 参数分片 | 把模型切小，逐片通信和合并 |
| minimum quorum | 最小法定数量 | 不等所有 learner，够数就合并 |
| adaptive grace window | 自适应等待窗口 | 吞吐允许时多等一点，收集更多 learner 更新 |
| outer optimizer | 外层优化器 | 在 learner 之间合并模型变化 |
| goodput | 有效训练吞吐 | 扣除故障、等待、重配置后的实际有效进度 |

这些术语背后的核心思想很一致：不要把一个训练任务设计成必须所有机器同时成功、同时可用、同时等速。

## 3. 传统同步训练为什么会卡住

先看普通 data-parallel / SPMD 预训练的流程：

```text
所有 worker 拿到同一版参数
  -> 各自算 forward / backward
  -> 全员 all-reduce 梯度
  -> 全员拿到同一版新参数
  -> 进入下一步
```

这个流程的优点是简单：每一步都是全局一致的。

但缺点也正来自这里：每一步都是全局同步点。只要有一个 worker 慢、失败、网络抖动，其他 worker 都会等。规模越大，最慢节点和故障概率越容易主导训练效率。

## 4. Decoupled DiLoCo 的整体形态

论文把训练系统拆成两个角色。

```text
Learner 0: 本地训练循环 + 后台 fragment 通信
Learner 1: 本地训练循环 + 后台 fragment 通信
Learner 2: 本地训练循环 + 后台 fragment 通信
...
        \        |        /
         \       |       /
              Syncer
  全局参数 fragment + outer optimizer 状态 + 合并逻辑
```

learner 是重计算单元，负责跑模型训练。syncer 是轻量协调单元，负责把不同 learner 的模型变化合起来。

关键变化是：learner 不再每一步都互相等待。一个 learner 可以继续本地训练，同时后台把某个 fragment 的更新发给 syncer。

## 5. learner 内部怎么跑

一个 learner 可以理解成一个小型 data-parallel 训练任务。它内部还是可以用正常的数据并行、张量并行或其他并行方式。

learner 的循环大致是：

```text
读取本地 batch
  -> forward / backward
  -> inner optimizer 更新本地参数
  -> 把某个参数 fragment 放到 host RAM
  -> 后台发送 fragment 给 syncer
  -> 继续下一步本地训练
```

这里有两个细节很重要。

第一，通信不必阻塞训练主循环。learner 把 fragment 放到 host RAM 后，就可以继续做计算。

第二，同步单位不是整个模型，而是 fragment。论文默认把模型切成多个 fragment，让每一步只处理一小块同步任务，从而降低峰值带宽，并让通信和计算更容易重叠。

## 6. syncer 内部怎么跑

syncer 不跑模型 forward/backward，也不保存激活。它维护的是：

- 每个 fragment 的全局参数。
- outer optimizer 的状态。
- 每个 learner 的训练进度。
- 每条通信通道上的状态和版本信息。

syncer 的循环大致是：

```text
等待 learner 发来 fragment 更新
  -> 检查这个 fragment 是否满足 quorum
  -> 必要时使用 grace window 多等一小会儿
  -> 对收到的更新做加权合并
  -> 更新全局 fragment 和 outer optimizer 状态
  -> 把新 fragment 发回 learner
```

这就是解耦的核心：syncer 合并的是 fragment 更新，而不是要求所有 learner 在同一个训练 step 上停住。

## 7. quorum 是怎么让系统不等慢节点的

假设有 8 个 learner。传统同步训练会等 8 个都完成。

Decoupled DiLoCo 可以设置 quorum，例如只要 1 个或几个 learner 的更新到达，就允许 syncer 合并。没有赶上的 learner 这轮不参与，或者权重很低。

这看上去牺牲了全局一致性，但换来两个好处：

1. 慢 learner 不会卡住快 learner。
2. 失败 learner 不会拖垮整轮训练。

为了避免“永远只用最快 learner”的问题，论文又引入 adaptive grace window。

## 8. grace window 在解决什么问题

如果 quorum 太小，系统虽然快，但每次合并可能只看到很少 learner 的更新，统计效率和训练稳定性会受影响。

grace window 的思路是：在不明显降低 goodput 的前提下，允许 syncer 多等一小段时间，看是否有更多 learner 的更新到达。

可以把它理解成一个很实用的工程折中：

```text
已经够 quorum 了
  -> 如果继续等会造成训练空转：马上合并
  -> 如果还有余量：稍等更多 learner
  -> grace window 结束后：不再等，直接合并
```

所以它不是为了恢复严格同步，而是为了在 availability 和模型质量之间找一个动态平衡点。

## 9. 参数 fragment 的一次完整生命周期

下面是一块参数 fragment 从 learner 到 syncer 再回来的流程。

```text
1. learner 本地训练若干步
2. learner 选中 fragment p
3. learner 把 p 的本地变化发送给 syncer
4. syncer 收集多个 learner 对 p 的更新
5. syncer 判断 quorum 和 grace window
6. syncer 做 token-weighted merging
7. syncer 用 outer optimizer 更新全局 p
8. syncer 把新的 p 发回 learner
9. learner 把收到的 p 合入本地模型
```

这条链路和 learner 的主训练循环并行，所以通信延迟不一定直接变成训练等待。

## 10. 为什么合并时不能只做简单平均

在多个 learner 异步训练时，不同 learner 的更新方向可能差异很大。简单平均有两个问题：

- 处理 token 数不同的 learner 被错误地等权看待。
- 不同方向的更新互相抵消，导致合并后的更新范数变小。

论文用两类机制缓解：

1. token-weighted merging：谁处理的 token 多，谁的贡献更大。
2. Radial-Directional Averaging：方向和模长分开处理，让更新规模更稳定。

这部分可以理解为算法层面的补偿：系统放松同步，优化器合并逻辑就要更小心。

## 11. learner 失败时发生什么

假设 learner 2 掉线。

传统同步训练里，其他 worker 很可能要等、重启或重配置。

Decoupled DiLoCo 的流程是：

```text
learner 2 掉线
  -> syncer 仍保留 learner 2 对应 shard 的状态
  -> 后续 fragment 合并不再等待 learner 2
  -> learner 2 的本轮权重可以视为 0
  -> learner 0/1/3/... 继续训练
  -> learner 2 恢复后，从 checkpoint 和 syncer 状态重新接入
```

系统的关键收益是 failure domain 变小了：一个 learner 的故障不会自然扩散成整个训练任务的故障。

## 12. 异构硬件为什么更容易接入

同步训练很怕异构硬件，因为快设备会等慢设备。为了避免浪费，通常要把 workload 切得非常均衡。

Decoupled DiLoCo 允许 learner 以不同速度前进。快 learner 贡献更多更新，慢 learner 少贡献一些。syncer 通过 quorum、grace window 和 token 权重吸收这种速度差。

这使得旧 TPU、新 TPU、不同机房里的临时资源，可以更自然地加入同一轮训练。

## 13. scavenging 临时算力的流程

scavenging 指利用临时可用、可能随时消失的算力。

在 Decoupled DiLoCo 里，加入临时 learner 的流程可以理解为：

```text
发现一批临时资源
  -> 创建新的 learner
  -> 从 syncer / checkpoint 拉取当前模型状态
  -> learner 开始本地训练并发送 fragment 更新
  -> 临时资源消失时 learner 退出
  -> syncer 后续不再等待它
```

它不像传统 DP 那样要求频繁重建全局同步组，因此更适合机会型资源。

## 14. 这篇论文的实验该怎么读

实验不是在说 Decoupled DiLoCo 总能比 DP 训练出更强模型。更准确的读法是：

```text
在模型质量基本接近 DP 的前提下，
它显著提升故障、异构、跨地域和临时算力场景下的训练可用性。
```

几个结果最值得记：

- 高故障率下，Decoupled DiLoCo goodput 明显高于 elastic DP。
- 下游 text / vision 指标大体可比。
- 通信带宽需求远低于 DP。
- scavenging 临时算力时，训练时间下降更明显。
- 异构硬件下，低 quorum 配合 grace window 可以避免被最慢 learner 卡住。

## 15. 和工程系统的映射

如果把这篇论文映射到一个真实训练平台，至少需要这些组件：

| 组件 | 对应职责 |
| --- | --- |
| learner manager | 创建、销毁、恢复 learner |
| syncer service | 维护 fragment 状态和 outer optimizer |
| fragment scheduler | 决定每一步同步哪块参数 |
| progress tracker | 记录 learner 进度和 token 数 |
| checkpoint service | 支持 learner 和 syncer 恢复 |
| event log / replay | 支持异步系统调试和复现 |
| admission controller | 决定临时资源能否加入训练 |

所以 Decoupled DiLoCo 不是一个单文件算法，而是一套训练系统协议。

## 16. 常见误区

**误区一：它就是异步 SGD。**

不是。它仍然保留 DiLoCo 式本地训练加外层合并，只是把 learner 与 syncer 解耦，并引入 fragment、quorum 和恢复机制。

**误区二：quorum 越小越好。**

quorum 小有利于 goodput，但可能降低每轮合并的信息量。实际系统要配合 grace window 和合并策略。

**误区三：syncer 是新的性能瓶颈。**

syncer 需要可靠设计，但它不做大模型前反向，状态和计算量远小于 learner，因此论文把它放在 CPU-only 资源上。

**误区四：它保证模型效果一定超过 DP。**

论文的主张更克制：在目标场景里，模型质量接近 DP，同时系统可用性和有效吞吐更好。

## 17. 本课检查点

读完后，你应该能回答：

1. 为什么大规模 SPMD 训练容易被故障和慢节点拖垮？
2. learner 和 syncer 分别保存什么状态？
3. parameter fragment 为什么能降低通信压力？
4. quorum 和 grace window 分别解决什么问题？
5. learner 掉线后，为什么其他 learner 可以继续跑？
6. 为什么异构硬件在这个框架下更容易利用？

## 18. 课后练习

1. 画一张时序图：4 个 learner、1 个 syncer，learner 2 在中途掉线，其他 learner 继续训练。
2. 设计一个简单指标面板：展示 active learners、fragment lag、quorum hit rate、grace wait time、goodput。
3. 思考一个反例：如果所有 learner 的数据分布严重不同，token-weighted merging 是否足够？
4. 比较 parameter-server、all-reduce、Decoupled DiLoCo 三种设计的故障域。

## 19. 拓展阅读

- 原始 DiLoCo / Streaming DiLoCo：理解本地训练和外层合并为什么能降低通信。
- Chandy-Lamport snapshot：理解异步系统里如何做一致性 checkpoint。
- 大规模训练平台的 checkpoint、replay、elastic recovery 机制。
