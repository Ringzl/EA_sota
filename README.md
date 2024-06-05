### 连续优化 (单目标)
* **测试问题**： 
Single Objective Bound Constrained Real-Parameter Numerical Optimisation, IEEE Congress on Evolutionary Computation (CEC) 2022. (F1 - F12) 中  "F2", "F4",  "F6", "F7", "F8", "F9", "F12" 问题 (d = 10维)
<div align="center">
<img src="https://s2.loli.net/2024/05/30/uCywdQxskqYPv91.png" width="600" />
</div>

* **测试算法**：
  * **DE_rand**: 
  $v = x_{r1} + F \times(x_{r2} - x_{r3})$
  * **DE_current_to_best**: 
  $v = x_{i} + F \times (x_{b} - x_{i}) + F \times (x_{r2} - x_{r3})$
  * **RCGA**: 模拟二进制交叉（SBX）和多项式变异（PM）
  * **SHADE**： Success-History Based Parameter Adaptation for Differential Evolution
  * **EA4eig**： Cooperative model of CoBiDE、IDEbd, CMA-ES, and jSO

* **测试结果**： 
进行10次独立实验, 函数评估次数: $10^5$ 次

  * 目标值（误差）优化结果

    | 算法/问题 | F2 | F4 | F6 | F7 | F8 | F9 | F12 |
    |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
    | DE_rand | 7.97e-01(1.59e+00) | 3.39e+01(5.10e+00) | 4.70e-01(7.81e-02) | 2.43e+01(2.99e+00) | 3.05e+00(8.63e-01) | 4.55e-14(1.36e-13) | 7.07e+01(1.51e+02) |
    | DE_current_to_best | **3.99e-01(1.20e+00)** | 2.03e+01(2.56e+00) | 4.63e-01(3.85e-01) | 2.13e+01(5.70e-01) | 1.72e+01(8.15e+00) | 4.09e-13(1.36e-13) | 1.55e+02(4.58e+00) |
    | RCGA | 3.60e+00(3.65e+00) | 3.00e+01(1.16e+01) | 1.11e+04(8.31e+03) | 1.67e+01(7.95e+00) | 1.52e+01(8.53e+00) | 1.10e+02(1.67e+02) | 1.42e+02(1.00e+01) |
    | SHADE | 5.96e+00(2.41e+00) | **1.02e+01(3.39e+00)** | 3.20e-01(1.12e-01) | 1.39e+00(8.43e-01) | 1.53e+00(2.46e-01) | **0.00e+00(0.00e+00)** | **1.04e+02(6.10e+01)** |
    | EA4eig | 7.97e-01(1.59e+00) | 1.12e+01(4.21e+00) | **1.25e-01(1.44e-01)** | **1.36e+00(1.12e+00)** | **8.55e-01(2.37e-01)** | **0.00e+00(0.00e+00)** | 1.07e+02(5.67e+01) |

  * 优化时间

  | 算法/问题              | 运行时间 |
  |--------------------|----|
  | DE_rand            |  74.24 s  |
  | DE_current_to_best |  74.75 s  |
  | RCGA               |  66.64 s  |
  | SHADE              |  110.02 s  |
  | EA4eig             |  89.24 s  |

**观点：** 综合性能和效率， **EA4eig** 算法表现最好。

---

### 连续约束优化 (单目标)

* **测试问题**：
Problem Definitions and Evaluation Criteria for the CEC 2017 Competition on Constrained Real- Parameter Optimization（C01-C28） 节选 C01 - C05 (d = 10维)
<div align="center">
<img src="https://s2.loli.net/2024/05/31/9PuDtTFxJ1VYGls.png" width="500" />
</div>

* **测试算法**
  * **C2oDE** (2018): 
    Composite Differential Evolution for Constrained Evolutionary Optimization
  * **DeCODE** (2019): 
    Decomposition-Based Multiobjective Optimization for Constrained Evolutionary Optimization
  * **CCEF** (2024): 
    A Competitive and Cooperative Evolutionary Framework for Ensemble of Constraint Handling Techniques

* **测试结果**
进行10次独立实验, 函数评估次数: $10^5$ 次

  * 目标值优化结果
  
  | <div style="width: 80pt"> 算法/问题 | <div style="width: 120pt"> C01 | <div style="width: 120pt"> C02 |  <div style="width: 150pt"> C03 | <div style="width: 120pt"> C04 | <div style="width: 120pt"> C05 |
  |:---:|:---:|:---:|:---:|:---:|:---:|
  | C2oDE  | 5.00e-15(4.16e-15) | 1.74e-14(1.61e-14) | 9.43e+02(4.76e+02) *20%* | 2.51e+01(3.74e+00) | 2.48e-10(1.44e-10) |
  | DeCODE | 4.75e-24(4.70e-24) | 3.29e-24(2.80e-24) | **2.40e-02(7.19e-02)** | 1.59e+01(7.46e-01) | 2.38e-17(2.52e-17) |
  | CCEF   | **8.37e-26(7.84e-26)** | **9.17e-26(7.03e-26)** | 1.81e+02(2.06e+02) | **1.51e+01(1.60e+00)** | **7.57e-19(6.27e-19)** |

  * 运行时间

  | 算法/问题 | 运行时间 |
  |--------------------|----|
  | C2oDE  |   70.07 s  |
  | DeCODE |   69.94 s  |
  | CCEF   |   82.10 s  |

**观点**： C2oDE 明显差于其他两种算法，CCEF在含等式约束问题上表现不太好，其他问题相比与DeCODE有略微优势。综合来看，DeCODE值得选择。

---

### 连续多目标优化

* **测试问题** : WFG1-9 （节选WFG1 和 WFG4 进行测试, d = 10维, n_obj=3）
<div align="center">
<img src="https://s2.loli.net/2024/06/03/btlQ9fiqvjws8VI.png" width="500" />
</div>

* **测试算法**
  * **NSGA-II (2002)**: A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II
  * **IBEA (2004)**: Indicator-Based Selection in Multiobjective Search
  * **MOEA/D (2007)**: MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition
  * **Two_arch2 (2015)**: Two_Arch2: An Improved Two-Archive Algorithm for Many-Objective Optimization
  * **RVEA (2016)**: A Reference Vector Guided Evolutionary Algorithm for Many-Objective Optimization

* **测试结果**

  * 评估指标： HV, reference point $y^* = \{3, 5, ..., 2M+1\}$

  * HV指标结果

  | 算法/问题     | WFG1               | WFG4               | DTLZ2              | DTLZ3              |
  |:---------:|:------------------:|:------------------:|:------------------:|:------------------:|
  | NSGA-II   | 5.66e-01(3.86e-02) | 6.58e-01(7.80e-03) | 9.89e-01(3.46e-03) | 9.89e-01(4.85e-03) |
  | IBEA      | 8.06e-01(4.27e-02) | 7.29e-01(5.98e-04) | 9.94e-01(2.13e-05) | 9.90e-01(2.49e-05) |
  | MOEA/D    | 4.76e-01(3.69e-02) | 7.10e-01(1.89e-03) | 9.94e-01(3.56e-06) | 9.40e-01(1.63e-01) |
  | RVEA      | 3.77e-01(2.00e-01) | 7.29e-01(3.94e-04) | 9.94e-01(1.02e-07) | 9.94e-01(1.74e-05) |
  | Two_arch2 | **8.63e-01(5.71e-02)** | **7.30e-01(1.03e-03)** | 9.94e-01(8.01e-06) | 9.94e-01(1.35e-05) |

  * 运行时间

  | 算法/问题 | 运行时间 |
  |--------------------|----|
  | NSGA-II  |   10.27 s + 9.65 s |
  | IBEA     |   189.56 s + 197.18 s  |
  | MOEA/D   |   37.49 s + 30.97 s |
  | Two_arch2|   1187.71 s  |
  | RVEA     |   2.30 s + 2.60 s  |

**观点**： Two_arch2 在所有问题上表现最好，但是运行时间最长， RVEA 运行效率最高，但是在WFG1上问题表现不好。综合来看IBEA相对不错。（需要结合具体需求进行选择）


---


### 组合优化

* **测试算法**
  * **pyomo+SCIP**:
  * **pyomo+Gurobi**:
  * **ACO**:
  * **VNS**:
  * **ALNS**:


* **TSP问题**


* **CVRP问题**

**观点**：

