# SupportOrbit MuSiQue Pilot：Seed 17 结果冻结记录

> 状态：实验已结束；保留 **CONTROL / SupportOrbit-SFT** 作为成功主线，否决 **HopPAIR** 附加目标。  
> 冻结决定：`STOP`。不得据此启动 seed 29/43、读取 shadow、访问官方 dev/test 或继续 DPO。  
> 本文只报告 seed 17 的真实工件；不声称 MuSiQue 官方 dev/test 成绩、SOTA 或跨数据集泛化。

## 1. 一句话结论

在 Qwen3-4B-Instruct-2507 上，用充分证据 `C`、干扰改写 `D`、关键证据缺失 `M` 三状态组成的 evidence orbit 做 LoRA SFT，CONTROL 相对未训练 BASE 显著改善了回答、证据选择和拒答；但额外加入关系损失的 HopPAIR 使模型过度拒答，预注册门控 9 项中失败 6 项，因此严格停止该消融方向。

## 2. 实验对象与口径

- `C`：官方 answerable context，目标为 `S | evidence=[...] | answer=...`。
- `D`：从同一 `C` 出发，确定性替换恰好两个非支持段落，支持段落与答案保持不变；用于检验无关干扰下的稳健性。
- `M`：官方配对的 unanswerable context，目标为 `U | evidence=[] | answer=INSUFFICIENT_EVIDENCE`。
- `BASE`：未进行本项目后训练的 Qwen3-4B-Instruct-2507。
- `CONTROL` / `SupportOrbit-SFT`：对 `C/D/M` 的规范化输出做 completion CE。
- `HopPAIR`：在 CONTROL 目标上增加 `0.1` 的 stop-gradient `C→D` token KL，以及 `0.2`、margin `2` 的 action flip loss。

数据来自 **MuSiQue Full v1.0 的官方 train split**。评估集是从 train 内部构造且按 leakage component 隔离的 400 个 dev orbit，共 1,200 条状态记录；它不是 MuSiQue 官方 dev/test。

训练使用 BF16 LoRA（`r=16`、`alpha=32`、dropout `0.05`、all-transformer-linear），seed `17`。CONTROL 与 HopPAIR 使用相同的 1,920 个训练 orbit、相同顺序和初始化，均完成 240 个 optimizer step，skipped step 为 0。

## 3. 绝对结果：成功的 SupportOrbit-SFT 主线

下表是内部 dev 上的描述性绝对结果。`Δ` 为 CONTROL − BASE；拒答错误率越低越好，其余指标越高越好。

| 指标 | BASE | CONTROL | Δ（百分点） |
|---|---:|---:|---:|
| C Answer F1 | 33.75% | 61.05% | **+27.30** |
| D Answer F1 | 34.51% | 63.73% | **+29.21** |
| CD-Min Answer F1 | 30.71% | 59.44% | **+28.73** |
| M 拒答率 | 31.75% | 86.00% | **+54.25** |
| Orbit Answer + Sufficiency F1 | 7.48% | 53.75% | **+46.27** |
| Orbit Support + Sufficiency F1 | 9.19% | 69.53% | **+60.34** |
| Parse-valid rate | 88.17% | 99.92% | **+11.75** |
| C/D False-refusal rate ↓ | 16.25% | 10.50% | **−5.75** |

可支持的结论是：三状态数据构造与结构化 SFT 在这个单 seed、内部留出集 pilot 中产生了大幅、方向一致的增益。这里的成功主线是 **SupportOrbit-SFT 本身**，不是 HopPAIR。

## 4. 预注册比较：HopPAIR 消融失败

冻结协议比较的是 CONTROL → HopPAIR，而非 BASE → CONTROL。bootstrap 使用 400 个配对 orbit、10,000 次采样、seed `20260814`；以下置信区间均为预注册的 orbit bootstrap 结果。

| 预注册门控 | CONTROL | HopPAIR | Δ（百分点） | 95% CI（百分点） | 结果 |
|---|---:|---:|---:|---:|---|
| CD-Min Answer F1 gain | 59.44% | 52.63% | −6.81 | [−10.97, −2.72] | FAIL |
| Orbit Answer + Sufficiency gain | 53.75% | 46.75% | −7.00 | [−11.05, −2.99] | FAIL |
| D Answer F1 gain | 63.73% | 57.32% | −6.41 | [−10.46, −2.41] | FAIL |
| C Answer F1 non-inferiority | 61.05% | 55.67% | −5.38 | [−9.69, −1.20] | FAIL |
| False-refusal non-inferiority ↓ | 10.50% | 17.88% | +7.38 | [+4.25, +10.63] | FAIL |
| M refusal | 86.00% | 86.00% | 0.00 | [−2.00, +2.00] | PASS |
| Orbit Support + Sufficiency non-inferiority | 69.53% | 60.98% | −8.55 | [−12.33, −4.90] | FAIL |
| Parse rate | 99.92% | 99.67% | −0.25 | [−0.75, +0.08] | PASS |
| Run integrity | — | — | — | — | PASS |

正式判定是 9 项门控中 6 项失败，`decision = STOP`。这不是“训练没跑起来”：两臂训练、绑定和评估完整性均通过，且最终 checkpoint 确实不同；失败来自真实的处理效应。

## 5. Post-hoc component bootstrap：依赖修正后的证据

内部 dev 的 400 个 orbit 并非 400 个完全独立单元，而是 17 个 leakage component，大小为：

`[101, 69, 56, 50, 24, 20, 15, 14, 13, 9, 8, 8, 6, 2, 2, 2, 1]`

因此又做了**事后** component bootstrap：以 component 为采样单位，仍为 10,000 次、seed `20260814`，统计量保持 orbit-weighted。它是依赖结构修正，不是预注册分析，不能替代上节的冻结门控。

### BASE → CONTROL

| 指标 | Δ（百分点） | Post-hoc 95% CI（百分点） |
|---|---:|---:|
| CD-Min Answer F1 | +28.73 | [+7.82, +44.78] |
| Orbit Answer + Sufficiency F1 | +46.27 | [+24.22, +60.99] |
| D Answer F1 | +29.21 | [+10.40, +44.52] |
| C Answer F1 | +27.30 | [+6.66, +43.58] |
| False-refusal rate ↓ | −5.75 | [−10.89, +5.18] |
| M refusal | +54.25 | [+43.59, +65.82] |
| Orbit Support + Sufficiency F1 | +60.34 | [+42.97, +68.80] |
| Parse rate | +11.75 | [+8.97, +13.74] |

除 false-refusal 改善的区间跨 0 外，CONTROL 的主要正向增益在 component bootstrap 下仍远离 0。

### CONTROL → HopPAIR

component bootstrap 后，若干 answer 指标的区间跨 0，但点估计仍未达到预注册门槛；更关键的是两项伤害仍稳健：false-refusal `+7.38 pp`，95% CI `[+0.86, +11.27]`；Orbit Support + Sufficiency F1 `−8.55 pp`，95% CI `[−13.40, −0.59]`。因此 `STOP` 结论不变。

## 6. 失败诊断（探索性，不是新确认性结论）

HopPAIR 的主要问题是 action calibration，而不是在仍选择 `S` 时完全失去答案生成能力：

- C 状态的 `S/U/X` 从 CONTROL 的 `355/44/1` 变成 HopPAIR 的 `325/72/3`；D 状态从 `360/40/0` 变成 `328/71/1`。
- 共有 31 个 orbit 在 C 与 D 上同时新增拒答；理想 `SSU` 轨迹从 299 降至 264，`UUU` 从 35 增至 63。
- 对两臂仍然输出 `S` 的样本，HopPAIR 的 Answer F1 并未整体崩溃：C 约 `+1.12 pp`，D 约 `+0.43 pp`。主要损失来自决策边界向 `U` 偏移。
- action flip 辅助项相对单 token 的 SFT action 监督过强，并在 margin 已超过目标后继续推动 logit 间隔；这与新增 false refusal 的方向一致。

因此不应把 HopPAIR 包装成成功创新点。可讲的研究价值是：预注册门控及时否决了一个会过度锐化充分性决策边界的关系损失，并保住了更简单、效果更好的 SupportOrbit-SFT。

## 7. 工件身份与哈希证据

### 冻结输入

| 工件 | SHA-256 |
|---|---|
| `protocols/support_orbit_pilot_v2.json` | `1a888808d4a09983d0e98d31f1668b2f5faf844dde3b44f4d5e863e555e0c33f` |
| `protocols/pilot_v2_redteam_signoff.json` | `791088bc96491094ae08e7dea0e7612f69449e482d90b52ca3bf172760cdae4c` |
| `protocols/pilot_v2_launch_receipt.json` | `ee0b6c1533efa19668d0e349f96e84113dd3ba012de199fe65ca9d8a6ddbf2dd` |
| `prepared_data_v2/manifest.json` | `84e97c0eb189d664149822c1244436f0b97e69e3cbca7e16be8186df52a7dfc1` |
| `prepared_data_v2/train.jsonl` | `641bae95c4eb410229fb8c87a52b24e58628c376c636632d242115ca7e4d5c12` |
| `prepared_data_v2/dev.jsonl` | `84acfa99def02660aa74d836f07ff4219c0c838c29e19d7d335a126c7840b909` |
| schedule identity | `027c7d3eba760be6956060a487836088911ea5d7746e4fb5586f811c2c7c6ac4` |
| source lock | `4009ed46316e3f5994003500f2dfa2bf6f99fbf7957c331b69e1444641d3193f` |

### 训练身份

- 两臂初始 trainable-parameter SHA 相同：`c7afebf202928e3d5e2687167f239a2638ba33333e79e10eced993da49402c44`。
- CONTROL run manifest：`f75280c92e445af51ccdf40be15d8a1d8033de5a2a4d44b10802a405cc73a5e9`；checkpoint aggregate：`8cd1d87143942eb76ef5ef53955780f0c7c349dcfb59dd21765384ec2c065f4c`；最终 trainable SHA：`50ef2b1718b1c3bf28fdb8d349aea2943511e2d5652f640630c0b5cffee34efe`。
- HopPAIR run manifest：`73192d2b947a03329e281caafc04ad7944f063ef759327f6bde017d4d628867f`；checkpoint aggregate：`388ae70e0576d513c2e0e522d164815a9a96f5c7f876a9d5c4186204b2c0aa67`；最终 trainable SHA：`ff1de40c45a119ba8983de2966be6c33e8b29523e4917e4263f1ed13f37332f0`。

### 生成、评估与冻结决定

| 工件 | SHA-256 |
|---|---|
| BASE predictions | `bf7da3f62fe2e5e25ea0be13322703826d337df4d8807f4e7d0110390aed79f8` |
| CONTROL predictions | `e7a17c32d045c7834943e66f8971cf9b3dc731411b4a494d9def5dd4e2eb9c03` |
| HopPAIR predictions | `992579e9967900aa6e02a923d1017bc6df920e171bae710070d6e4275344b7ab` |
| BASE evaluation | `d5d32b0f6481669133e0fd260698698265fa1672be1bc62493c2aa1d700b7644` |
| CONTROL evaluation | `9a30472a25be6073d9c9bf9b1ade33a4015bb7590e1b6ae3bb15b60a278bb7ee` |
| HopPAIR evaluation | `3d66abb5f20b30f445e773135fe3b64ec5bf145dc89d45e1540528b51b1039a6` |
| frozen CONTROL→HopPAIR comparison | `ee2259e8d7fcc185de8c1de31c5738c0ad04b2f6cb48695895d995dd03d40f02` |

冻结 comparison 内的 8 项身份/完整性检查全部为 true：arm role、protocol、dataset manifest、split artifact、orbit IDs、schema 以及两臂 run integrity 均匹配。

## 8. 可访问范围与 claim 边界

### 可以说

- 在 **MuSiQue train 内部构造、component-isolated 的 400-orbit dev pilot** 上，SupportOrbit-SFT 相对同一 BASE 在回答、证据、充分性拒答和格式解析上取得显著描述性提升。
- 项目真实完成了数据构造、LoRA 后训练、严格绑定评估、预注册门控与失败诊断。
- HopPAIR 是一个被实验否决的消融；停止规则按预注册协议执行。

### 不可以说

- 不得写成 MuSiQue 官方 dev 或官方 test 结果。
- 不得声称 SOTA、论文级外部有效性或对其他 RAG/Agent 场景已经泛化。
- 不得把 post-hoc component bootstrap 说成预注册分析。
- 不得把单 seed 结果表述为多 seed 稳定复现。
- 不得声称 shadow 已验证；shadow body 保持 sealed、未读。
- 不得把 HopPAIR 写成正增益方法，也不得只挑 M refusal 不变来掩盖其整体失败。

## 9. 最终项目定位

这个 pilot 最可信的创新点不是“复杂损失一定比 SFT 强”，而是把 RAG 的充分性判断变成可训练、可配对评估的三状态 evidence orbit，并用严格的停止规则区分：

1. **可保留的成果**：SupportOrbit 数据设计 + 结构化后训练，在内部 pilot 上获得大幅真实增益；
2. **应公开的负结果**：HopPAIR 的相对 action loss 过度锐化拒答边界，导致更多 false refusal；
3. **工程与研究能力证据**：冻结协议、工件哈希、同初始化公平对照、配对 bootstrap、依赖修正与 honest stop。
