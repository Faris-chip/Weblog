---
title: NPU vs GPU vs CPU（一）
date: 2026-04-09 13:50:00
categories:
  - NPU
tags:
  - NPU设计
  - 闲谈
  - 科技前沿
index_img: /img/landscape4.png
banner_img: /img/landscape4.png
---
去年参加了第一届全国 RISC-V 创新应用大赛，当时设计了一个 16x16 的脉动阵列，当时设计的脉动阵列比较简单，很多问题没有仔细考虑，比如计算精度、数据流动方式、使用场景等问题，现在开始系统性地学习 NPU 设计，加深自己对 AI 加速器设计的认识和理解，同时补充自己的知识体系，这篇文章先从 NPU 的架构特征入手，分析 NPU 和 CPU、GPU的区别~

<!-- more -->

## NPU vs GPU vs CPU：计算密度与专用性分析

NPU、GPU和CPU代表了三种不同的计算架构设计理念，它们在通用性和效率之间做出了不同的权衡。理解这些架构的本质差异对于NPU设计至关重要。

### **CPU（中央处理器）** 追求通用性和灵活性：

CPU的设计哲学是”让任何程序都能高效运行”。现代CPU采用超标量架构，能够同时执行多条指令。以Intel的Golden Cove架构为例，单个核心拥有6个执行端口，可以同时处理不同类型的指令。CPU的复杂控制逻辑占据了芯片面积的很大比例，这些逻辑包括：

* 复杂的控制逻辑，支持乱序执行、分支预测、推测执行
* 大容量缓存层次（L1/L2/L3），典型配置为每核32KB L1，256KB L2，共享8-32MB L3
* 少量高性能核心（4-128核），每核支持SIMD扩展（AVX-512等）
* 计算密度：约0.1-0.5 TFLOPS/W（FP32）

从晶体管利用效率角度分析，CPU的设计存在固有的低效性。根据学术研究，典型的高性能CPU核心中，晶体管分布大致如下：

* 算术逻辑单元（ALU）和浮点单元（FPU）：仅占5-10%
* 控制逻辑（包括指令解码、分支预测、乱序执行引擎）：占30-40%
* 缓存和存储系统：占40-50%
* 其他辅助电路：占10-15%

这种分布反映了通用计算的根本挑战：程序行为的不可预测性。CPU必须为最坏情况做准备，即使这些情况很少发生。例如，分支预测器的TAGE算法维护多个预测表，总容量可达数MB，相当于一个小型缓存。但对于神经网络推理这样的规则workload，分支预测准确率接近100%，这些复杂机制变成了纯粹的开销。

分支预测器是CPU的核心组件之一，现代CPU采用TAGE（TAgged GEometric）预测器，维护多个预测表，每个表使用不同长度的历史信息。预测准确率可达97%以上，但这种复杂性的代价是功耗和面积。一个高端CPU核心中，只有不到10%的晶体管用于实际的算术运算，其余都用于控制、缓存和数据移动。

CPU的内存系统设计也体现了通用性优先的原则。多级缓存系统采用包容性或排他性策略，支持各种访问模式。硬件预取器能够识别顺序、跨步等多种访问模式，但对于神经网络的规则访问模式来说，这些复杂机制显得过度设计。

### **GPU（图形处理器）** 优化并行吞吐量：

GPU的设计源于图形渲染的需求，后来发展成为通用并行计算平台。GPU架构的核心思想是”用大量简单核心替代少量复杂核心”。以NVIDIA的Ampere架构为例，每个SM（Streaming Multiprocessor）包含：

* 大规模SIMT（Single Instruction Multiple Thread）架构
* 数千个简单核心，组织成SM（Streaming Multiprocessor）
* 有限的缓存，更多依赖高带宽内存（HBM）
* 计算密度：约10-30 TFLOPS/W（FP16）

SIMT执行模型是GPU效率的关键。32个线程组成一个warp，共享同一条指令流。这种设计大幅减少了指令获取和解码的开销。然而，当warp内的线程执行不同分支时，会发生分支发散（branch divergence），导致性能下降。对于神经网络推理，这通常不是问题，因为计算模式高度规则。

GPU的内存系统针对高带宽优化，而非低延迟。HBM2E可以提供超过1TB/s的带宽，但访问延迟高达数百个周期。GPU通过大量并发线程隐藏延迟，当一组线程等待内存时，调度器切换到另一组线程。这种设计对于训练很有效，但对于低批量推理，线程数不足以完全隐藏延迟。

内存带宽的有效利用是GPU性能的关键。理论带宽和实际带宽之间的差距可以用以下模型描述：

$$
\text{Effective Bandwidth} = \text{Peak Bandwidth} \times \text{Utilization}
$$

其中利用率受多个因素影响：

* 内存访问粒度：GPU的内存事务通常是32字节或128字节的倍数
* 合并效率：$\eta_{coalesce} = \frac{\text{有用数据量}}{\text{实际传输数据量}}$
* Bank冲突：当多个线程访问同一bank时发生串行化

对于典型的神经网络推理，GPU的内存带宽利用率分析如下：

* 理想的连续访问：利用率可达85-95%
* 跨步访问（stride=2）：利用率降至40-50%
* 随机访问：利用率可能低于10%

Tensor Core是NVIDIA针对AI工作负载的专门优化，本质上是矩阵乘累加单元。一个Tensor Core可以在一个周期内完成4×4矩阵乘法，这已经具有NPU的某些特征。但Tensor Core仍然嵌入在通用GPU架构中，需要通过CUDA核心进行数据准备和后处理。

Tensor Core的计算能力可以表示为：

$$
\text{TFLOPS}_{TC} = N_{SM} \times N_{TC/SM} \times f_{clock} \times \text{Ops/cycle}
$$

以A100为例：

* $N_{SM} = 108$（流处理器数量）
* $N_{TC/SM} = 4$（每个SM的Tensor Core数）
* $f_{clock} = 1.41$ GHz
* Ops/cycle = 256（FP16混合精度）
* 总计：156 TFLOPS（FP16）

但实际性能受限于数据供给能力。Tensor Core的算术强度要求极高，只有大矩阵乘法才能充分利用其计算能力。

### **NPU（神经网络处理器）** 针对AI推理专门优化：

NPU代表了极致专用化的设计方向。通过放弃通用性，NPU能够实现比GPU高10倍、比CPU高100倍的能效。NPU设计的核心洞察是：神经网络推理的计算模式是高度可预测的。

* 专用矩阵乘法单元（脉动阵列或数据流架构）
* 固定的数据流模式，减少控制开销
* 优化的存储层次，匹配神经网络访问模式
* 计算密度：约50-200 TOPS/W（INT8/FP4）

NPU的计算核心通常是大规模矩阵乘法阵列。以Google TPU v4为例，其MXU（Matrix Multiply Unit）是128×128的脉动阵列，每个周期可以完成16384个MAC操作。与GPU的Tensor Core相比，NPU的矩阵单元规模更大，且数据流经过精心设计以最大化重用。

脉动阵列的计算效率可以通过数学模型精确分析。对于$M \times N$的脉动阵列执行$P \times Q \times R$的矩阵乘法：

$$
\text{Cycles}_{compute} = \max(P, M) + \max(R, N) + Q - 1
$$

利用率计算：

$$
\text{Utilization} = \frac{P \times Q \times R}{M \times N \times \text{Cycles}_{compute}}
$$

当矩阵维度是阵列维度的整数倍时，利用率接近100%。例如，128×128阵列处理256×256×256矩阵乘法：

* 计算周期：256 + 256 + 256 - 1 = 767周期
* 总操作数：256³ = 16,777,216
* 阵列容量：128² × 767 = 12,582,912
* 利用率：16,777,216 / 12,582,912 = 133%（通过双缓冲重叠）

控制逻辑的简化是NPU高能效的关键。神经网络的每一层都是确定性的计算，没有数据依赖的分支。这意味着可以在编译时完全确定执行顺序，运行时只需要简单的计数器和状态机。TPU的指令集只有十几条指令，而x86 CPU有上千条指令。

控制开销的量化分析表明，NPU相比CPU在控制逻辑上的节省是巨大的：

* CPU：每条指令需要~100个晶体管用于解码和控制
* GPU：每条指令需要~20个晶体管（SIMT摊销）
* NPU：每条指令需要~5个晶体管（静态调度）

这种简化直接转化为能效提升。根据Amdahl定律的变体： 

$$
\text{Speedup}_{energy} = \frac{1}{f_{control} \times \frac{1}{S_{control}} + (1 - f_{control})}
$$

其中$f_{control}$是控制逻辑的能耗占比（CPU约40%），$S_{control}$是控制逻辑的简化倍数（NPU可达10×），得出能效提升约1.6×，仅从控制简化一项。

NPU的内存系统专门为神经网络访问模式优化。权重在推理过程中是只读的，可以预先加载到片上存储。激活值在层之间流动，具有生产者-消费者模式。这种可预测性允许使用简单的双缓冲或乒乓缓冲策略，避免复杂的缓存一致性协议。

数据重用的数学模型对于理解NPU的优势至关重要。考虑卷积操作的数据重用： 

$$
\text{Reuse Factor} = \frac{\text{Total Operations}}{\text{Unique Data Elements}}
$$

对于卷积层Conv2D(C_in, C_out, K×K, H×W)：

* 计算量：$2 \times C_{in} \times C_{out} \times K^2 \times H_{out} \times W_{out}$
* 输入数据：$C_{in} \times H \times W$
* 权重数据：$C_{in} \times C_{out} \times K^2$
* 输出数据：$C_{out} \times H_{out} \times W_{out}$

重用因子： 

$$
RF = \frac{2 \times C_{in} \times C_{out} \times K^2 \times H_{out} \times W_{out}}{C_{in} \times H \times W + C_{in} \times C_{out} \times K^2 + C_{out} \times H_{out} \times W_{out}}
$$

典型值（C_in=256, C_out=256, K=3, H=W=56）：

$$
RF \approx \frac{2 \times 256 \times 256 \times 9 \times 56 \times 56}{256 \times 58 \times 58 + 256 \times 256 \times 9 + 256 \times 56 \times 56} \approx 40
$$

这意味着每个数据元素平均被重用40次，NPU通过固定的数据流模式可以充分利用这种重用，而CPU/GPU的通用缓存可能无法捕获所有重用机会。
