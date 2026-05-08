const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak, TabStopType, TabStopPosition,
  LevelFormat, ImageRun,
} = require("docx");

// ========== CONSTANTS ==========
const PAGE_W = 11906; // A4
const PAGE_H = 16838;
const ML = 1800; // 3.17cm
const MR = 1800;
const MT = 1440; // 2.54cm
const MB = 1440;
const CW = PAGE_W - ML - MR; // content width ~8306

const FONT_SONG = "SimSun";
const FONT_HEI = "SimHei";
const FONT_EN = "Times New Roman";
const PT = (n) => n * 2; // half-points

// ========== IMAGE HELPERS ==========
const FIG_DIR = path.join(__dirname, "figures");

function loadImg(name) {
  return fs.readFileSync(path.join(FIG_DIR, name));
}

function figImage(fileName, widthPx, heightPx) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 160, after: 60 },
    children: [new ImageRun({
      type: "png",
      data: loadImg(fileName),
      transformation: { width: widthPx, height: heightPx },
      altText: { title: fileName, description: fileName, name: fileName },
    })],
  });
}

// ========== STYLE HELPERS ==========
function cn(text, opts = {}) {
  return new TextRun({ text, font: opts.font || FONT_SONG, size: opts.size || PT(12), bold: opts.bold, italics: opts.italics, ...opts });
}
function en(text, opts = {}) {
  return new TextRun({ text, font: opts.font || FONT_EN, size: opts.size || PT(12), bold: opts.bold, italics: opts.italics, ...opts });
}
function mixed(text, opts = {}) {
  return new TextRun({ text, font: opts.font || FONT_SONG, size: opts.size || PT(12), bold: opts.bold, ...opts });
}

function bodyPara(children, opts = {}) {
  return new Paragraph({
    spacing: { line: 360, before: 60, after: 60 },
    indent: opts.noIndent ? {} : { firstLine: 480 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    children: Array.isArray(children) ? children : [children],
    ...opts,
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 240 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text, font: FONT_HEI, size: PT(16), bold: true })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 180 },
    children: [new TextRun({ text, font: FONT_HEI, size: PT(14), bold: true })],
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 120 },
    children: [new TextRun({ text, font: FONT_HEI, size: PT(12), bold: true })],
  });
}

// ========== TABLE HELPERS ==========
const thinBorder = { style: BorderStyle.SINGLE, size: 1, color: "000000" };
const thickBorder = { style: BorderStyle.SINGLE, size: 3, color: "000000" };
const noBorder = { style: BorderStyle.NONE, size: 0 };

function threeLineTableBorders(isTop, isBottom) {
  return {
    top: isTop ? thickBorder : noBorder,
    bottom: isBottom ? thickBorder : thinBorder,
    left: noBorder,
    right: noBorder,
  };
}

function makeTable(headers, rows, colWidths) {
  const totalW = colWidths.reduce((a, b) => a + b, 0);
  const cellMargins = { top: 40, bottom: 40, left: 80, right: 80 };

  const headerRow = new TableRow({
    children: headers.map((h, i) => new TableCell({
      borders: threeLineTableBorders(true, false),
      width: { size: colWidths[i], type: WidthType.DXA },
      margins: cellMargins,
      shading: { fill: "FFFFFF", type: ShadingType.CLEAR },
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { line: 280 },
        children: [new TextRun({ text: h, font: FONT_HEI, size: PT(10.5), bold: true })],
      })],
    })),
  });

  const dataRows = rows.map((row, ri) => new TableRow({
    children: row.map((cell, ci) => new TableCell({
      borders: threeLineTableBorders(false, ri === rows.length - 1),
      width: { size: colWidths[ci], type: WidthType.DXA },
      margins: cellMargins,
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { line: 280 },
        children: [new TextRun({ text: String(cell), font: FONT_SONG, size: PT(10.5) })],
      })],
    })),
  }));

  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows],
  });
}

function tableCaption(text) {
  return new Paragraph({
    spacing: { before: 200, after: 80 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text, font: FONT_HEI, size: PT(10.5) })],
  });
}

function figCaption(text) {
  return new Paragraph({
    spacing: { before: 80, after: 200 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text, font: FONT_HEI, size: PT(10.5) })],
  });
}

function refTag(tag) {
  return new TextRun({ text: tag, font: FONT_EN, size: PT(10.5), superScript: true });
}

// ========== NUMBERING CONFIG ==========
const numberingConfig = {
  config: [
    {
      reference: "bullets",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } },
      }],
    },
    {
      reference: "numbers",
      levels: [{
        level: 0, format: LevelFormat.DECIMAL, text: "%1.",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } },
      }],
    },
    {
      reference: "numbers2",
      levels: [{
        level: 0, format: LevelFormat.DECIMAL, text: "%1.",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } },
      }],
    },
    {
      reference: "numbers3",
      levels: [{
        level: 0, format: LevelFormat.DECIMAL, text: "%1.",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } },
      }],
    },
  ],
};

function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { line: 340 },
    children: [cn(text, { size: PT(12) })],
  });
}

function numbered(text, ref = "numbers") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { line: 340 },
    children: [cn(text, { size: PT(12) })],
  });
}

// ========== DOCUMENT CONTENT ==========
function buildContent() {
  const children = [];

  // ===== COVER PAGE =====
  children.push(new Paragraph({ spacing: { before: 3000 } }));
  children.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 400 },
    children: [new TextRun({ text: "ChannelHub", font: FONT_EN, size: PT(36), bold: true })],
  }));
  children.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [new TextRun({ text: "5G NR 信道数据工场平台说明书", font: FONT_HEI, size: PT(26), bold: true })],
  }));
  children.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 100 },
    children: [new TextRun({ text: "（数据采集 · 处理 · 管理 · 评估一体化平台）", font: FONT_SONG, size: PT(14) })],
  }));
  children.push(new Paragraph({ spacing: { before: 1500 } }));
  children.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 100 },
    children: [cn("版本：V2.0", { size: PT(14) })],
  }));
  children.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 100 },
    children: [cn("日期：2026 年 5 月", { size: PT(14) })],
  }));
  children.push(new Paragraph({ children: [new PageBreak()] }));

  // ===== CHAPTER 1: OVERVIEW =====
  children.push(h1("第一章  平台概述"));

  children.push(h2("1.1  平台定位与功能"));
  children.push(bodyPara([
    cn("ChannelHub 是面向 5G NR 信道表征学习的数据工场平台，定位为"),
    cn("信道数据工厂", { bold: true }),
    cn("与"),
    cn("模型评估引擎", { bold: true }),
    cn("的一体化系统。平台负责多源信道数据的采集、管理、划分与导出，同时支持外部训练模型的导入、推理与评估。训练过程在外部独立平台完成，平台专注于数据质量与评估公正性的保障。"),
  ]));

  children.push(bodyPara([
    cn("平台的核心工作流包括三个阶段：（1）通过多种数据源采集信道数据，支持 3GPP 38.901 统计模型、射线追踪仿真及 MATLAB QuaDRiGa 引擎；（2）对采集数据进行特征提取、数据集划分与多格式导出，供外部训练框架消费；（3）导入外部训练好的模型，在锁定的测试集上评估并生成排行榜，确保模型对比的公平性。"),
  ]));

  children.push(h2("1.2  技术架构"));
  children.push(bodyPara([
    cn("平台采用前后端分离架构，后端基于 FastAPI 框架提供 RESTful API，前端使用 React 18 与 Ant Design 组件库构建交互界面。任务调度通过 Dramatiq 消息队列实现异步处理，Redis 作为消息代理。数据持久化采用 SQLite 数据库（兼容 PostgreSQL），配合 Alembic 进行数据库迁移管理。"),
  ]));

  children.push(tableCaption("表 1-1  平台技术栈"));
  children.push(makeTable(
    ["层次", "技术组件", "说明"],
    [
      ["前端", "React 18 + TypeScript + Ant Design", "单页应用，响应式布局"],
      ["后端", "FastAPI + SQLAlchemy + Pydantic", "RESTful API，类型安全"],
      ["数据库", "SQLite + Alembic", "轻量级，支持迁移"],
      ["任务队列", "Dramatiq + Redis", "异步任务调度与监控"],
      ["信道仿真", "NumPy + SciPy + Sionna 2.0", "3GPP 信道模型实现"],
      ["ML 框架", "PyTorch", "模型评估与推理"],
      ["配置管理", "Hydra + pydantic-settings", "ML 超参与基础设施配置分离"],
    ],
    [1600, 3800, 2906],
  ));

  children.push(h2("1.3  功能导航"));
  children.push(bodyPara([
    cn("平台前端提供以下主要功能页面，通过左侧导航栏进行切换："),
  ]));
  children.push(tableCaption("表 1-2  平台功能页面"));
  children.push(makeTable(
    ["页面", "路径", "功能说明"],
    [
      ["仪表盘", "/", "系统概览、数据统计"],
      ["数据集", "/datasets", "数据源聚合列表与详情"],
      ["信道浏览", "/channels", "Bridge 处理后的信道可视化"],
      ["数据采集", "/collect", "四步向导式采集配置"],
      ["数据处理", "/process", "Bridge 特征提取任务管理"],
      ["任务", "/jobs", "任务列表、创建与实时监控"],
      ["运行记录", "/runs", "训练/评估运行记录与指标"],
      ["对比", "/compare", "多模型指标对比分析"],
      ["模型", "/models", "模型仓库、上传、评估、排行榜"],
    ],
    [1400, 2000, 4906],
  ));

  children.push(new Paragraph({ children: [new PageBreak()] }));

  // ===== CHAPTER 2: DATA COLLECTION =====
  children.push(h1("第二章  数据采集"));

  children.push(h2("2.1  数据源"));
  children.push(bodyPara([
    cn("平台支持三种信道数据源，各自具有不同的物理建模精度与计算依赖。三种数据源共享统一的输出格式（ChannelSample），确保下游处理流程的一致性。"),
  ]));
  children.push(tableCaption("表 2-1  数据源对比"));
  children.push(makeTable(
    ["数据源", "建模方法", "依赖", "适用场景"],
    [
      ["internal_sim", "3GPP 38.901 统计模型 (TDL/CDL)", "纯 Python", "快速验证、大规模生成"],
      ["sionna_rt", "Sionna 射线追踪 + TDL 回退", "GPU + Sionna 2.0", "真实场景精确建模"],
      ["quadriga_real", "QuaDRiGa 引擎 (MATLAB)", "本地 MATLAB", "3GPP 全场景覆盖"],
    ],
    [1600, 2600, 1800, 2306],
  ));

  children.push(h2("2.2  采集向导操作流程"));
  children.push(bodyPara([
    cn("数据采集通过四步向导（CollectWizard）完成，用户依次选择数据源、配置设备参数、设置信道参数、确认并提交。向导的设计确保参数完整性与物理一致性。"),
  ]));

  children.push(h3("2.2.1  Step 1：选择数据源"));
  children.push(bodyPara([
    cn("用户在第一步选择数据源类型。选择后，后续表单将根据数据源约束动态调整可用选项。例如，室内热点（InH）和农村宏站（RMa）场景的路径损耗模型仅在 QuaDRiGa 数据源下可用。"),
  ]));

  children.push(h3("2.2.2  Step 2：设备与环境配置"));
  children.push(bodyPara([
    cn("第二步通过可折叠面板（Collapse）组织参数，包含拓扑配置、天线阵列、射频参数和终端配置四个面板。顶部提供六个场景预设卡片，点击后自动填充全部参数，用户可在此基础上自由修改。"),
  ]));

  children.push(tableCaption("表 2-2  场景预设"));
  children.push(makeTable(
    ["预设名称", "场景", "频段", "带宽", "ISD", "天线配置"],
    [
      ["城市宏站 64T", "UMa", "3.5 GHz", "100 MHz", "500 m", "8×4×2 = 64T64R"],
      ["城市微站 32T", "UMi", "3.5 GHz", "100 MHz", "200 m", "4×4×2 = 32T32R"],
      ["室内热点 8T", "InH", "3.5 GHz", "50 MHz", "50 m", "2×2×2 = 8T8R"],
      ["农村宏站 4T", "RMa", "700 MHz", "20 MHz", "1732 m", "2×2×1 = 4T4R"],
      ["毫米波 28G", "UMi", "28 GHz", "100 MHz", "100 m", "8×4×2 = 64T64R"],
      ["高铁 350km/h", "UMa-LOS", "2.6 GHz", "100 MHz", "1000 m", "8×4×2 = 64T64R"],
    ],
    [1500, 1000, 1200, 1200, 1000, 2406],
  ));

  children.push(bodyPara([
    cn("拓扑配置面板支持六边形蜂窝与线性轨道两种布局。蜂窝模式下站点数限制为 1/3/7/19/37（对应 0~4 环），线性模式下可自由设置（2~57 站）。天线阵列面板支持 BS 和 UE 各自独立的面阵预设，以及交叉极化鉴别度（XPD）配置。"),
  ]));

  // -- Fig: Hex topology --
  children.push(figImage("fig_hex_topology.png", 420, 390));
  children.push(figCaption("图 2-1  7 小区六边形蜂窝拓扑示意图"));

  children.push(bodyPara([
    cn("射频参数面板中，RB 数根据带宽与子载波间隔自动查询 3GPP TS 38.101"),
    refTag("[1]"),
    cn(" Table 5.3.2-1 标准表确定，无需手动输入。"),
  ]));

  children.push(h3("2.2.3  Step 3：信道配置"));
  children.push(bodyPara([
    cn("信道配置包含核心参数与高级参数两层。核心参数始终可见，包括链路方向、信道估计模式、导频类型、采样数和传播场景。SRS 高级参数（周期、跳频、梳齿、带宽树等）默认折叠，需要时展开配置。"),
  ]));

  children.push(tableCaption("表 2-3  信道估计模式"));
  children.push(makeTable(
    ["模式", "名称", "特点"],
    [
      ["ideal", "理想估计", "直接使用真实信道，无噪声和干扰"],
      ["ls_linear", "LS + 线性插值", "导频位置 LS 估计，频域线性插值"],
      ["ls_mmse", "LS + MMSE 插值", "利用信道统计信息的 MMSE 估计"],
      ["ls_hop_concat", "LS + 跳频拼接", "SRS 逐跳独立估计后按 RB 拼接"],
    ],
    [1600, 2200, 4506],
  ));

  children.push(h3("2.2.4  Step 4：确认与提交"));
  children.push(bodyPara([
    cn("确认页以 Tag 横幅显示关键配置摘要（数据源、站点数、天线规格、频段、带宽、采样数），下方 Descriptions 组件展示完整参数。提交后系统自动创建 simulate 类型任务，并通过 Dramatiq 队列分发至 Worker 执行。"),
  ]));

  children.push(h2("2.3  高铁场景专属能力"));
  children.push(bodyPara([
    cn("平台内置高铁（HSR）场景支持，采用线性拓扑部署模型。站点交替部署在轨道两侧（间距可配），UE 集中分布在列车车厢内随列车整体移动。主要特性如下："),
  ]));
  children.push(tableCaption("表 2-4  高铁场景参数"));
  children.push(makeTable(
    ["特性", "默认值", "说明"],
    [
      ["线性拓扑", "topology_layout: linear", "站点交替部署在轨道两侧"],
      ["轨道偏移", "track_offset_m: 80", "站点到轨道中心线的垂直距离"],
      ["HyperCell 组网", "hypercell_size: 4", "每组 RRH 共享 PCI，减少切换"],
      ["车体穿透损耗", "22 dB", "3GPP TR 38.913 Table 7.4.4-1"],
      ["移动模式", "track", "沿轨道中心线匀速行驶"],
      ["Doppler 计算", "自动", "相对最近基站的径向 Doppler"],
    ],
    [1800, 2600, 3906],
  ));

  // -- Fig: Linear HSR topology --
  children.push(figImage("fig_linear_topology.png", 540, 180));
  children.push(figCaption("图 2-2  高铁线性拓扑部署示意图"));

  children.push(new Paragraph({ children: [new PageBreak()] }));

  // ===== CHAPTER 3: CHANNEL MODELING =====
  children.push(h1("第三章  信道物理建模"));

  children.push(h2("3.1  OFDM 频域资源配置"));
  children.push(bodyPara([
    cn("平台支持 NR FR1 频段的全部标准带宽（5~100 MHz）与子载波间隔（15/30/60/120 kHz）。RB 数量严格按照 3GPP TS 38.101"),
    refTag("[1]"),
    cn(" Table 5.3.2-1 查表确定，以 100 MHz 带宽、30 kHz 子载波间隔为例，对应 273 个 RB。"),
  ]));

  children.push(h2("3.2  TDD 时隙配置"));
  children.push(bodyPara([
    cn("TDD 时隙模式严格按照 3GPP TS 38.213"),
    refTag("[2]"),
    cn(" §11.1 实现，每个样本映射到一个 TDD slot。导频放置遵循时隙方向约束：SRS 仅在上行符号发送，CSI-RS 仅在下行符号发送，保护间隔符号不放置导频。"),
  ]));
  children.push(tableCaption("表 3-1  支持的 TDD 时隙配比"));
  children.push(makeTable(
    ["配比", "时隙序列", "周期", "DL:UL 符号比"],
    [
      ["DDDSU", "D-D-D-S-U", "5 ms", "44:16"],
      ["DDSUU", "D-D-S-U-U", "5 ms", "38:30"],
      ["DDDDDDDSUU", "D×7-S-U-U", "10 ms", "104:32"],
      ["DDDSUDDSUU", "D×3-S-U-D×2-S-U-U", "10 ms", "90:32"],
      ["DSUUD", "D-S-U-U-D", "5 ms", "20:32"],
    ],
    [1800, 2800, 1200, 2506],
  ));

  children.push(h2("3.3  路径损耗模型"));
  children.push(bodyPara([
    cn("路径损耗模型遵循 3GPP TR 38.901"),
    refTag("[3]"),
    cn(" Table 7.4.1-1 规定。系统根据 Table 7.4.2-1 的 LOS 概率公式自动判定每个小区的 LOS/NLOS 状态，并选择对应的路损公式与信道模型。每个小区独立进行 LOS 判定（per-cell LOS），避免使用错误的路损模型。"),
  ]));
  children.push(tableCaption("表 3-2  路径损耗公式"));
  children.push(makeTable(
    ["场景", "条件", "路损公式 PL [dB]", "阴影衰落 σ_SF"],
    [
      ["UMa-LOS", "d₂D ≤ d'BP", "28.0 + 22·lg(d₃D) + 20·lg(fc)", "4 dB"],
      ["UMa-NLOS", "—", "13.54 + 39.08·lg(d₃D) + 20·lg(fc)", "6 dB"],
      ["UMi-LOS", "d₂D ≤ d'BP", "32.4 + 21·lg(d₃D) + 20·lg(fc)", "4 dB"],
      ["UMi-NLOS", "—", "22.4 + 35.3·lg(d₃D) + 21.3·lg(fc)", "7.82 dB"],
      ["InF", "—", "31.84 + 21.50·lg(d₃D) + 19.0·lg(fc)", "7.56 dB"],
    ],
    [1300, 1300, 3706, 2000],
  ));
  children.push(bodyPara([
    cn("其中 d'BP = 4·h'BS·h'UT·fc/c 为断点距离，fc 以 Hz 为单位，d₃D 为 3D 距离，lg 为以 10 为底的对数。"),
  ]));

  children.push(h2("3.4  信道模型"));
  children.push(bodyPara([
    cn("平台支持 3GPP 38.901"),
    refTag("[3]"),
    cn(" 定义的 TDL（Tapped Delay Line）和 CDL（Cluster Delay Line）两类信道模型。NLOS 场景默认使用 TDL-A/B/C，LOS 场景自动切换至 TDL-D/E（含 Rician K 因子）。信道的时延扩展（τ_rms）和阴影衰落（SF）等大尺度参数从 Table 7.5-6 采样，并通过位置量化确保空间一致性。"),
  ]));

  children.push(h2("3.5  干扰建模"));
  children.push(bodyPara([
    cn("平台采用物理层级的干扰建模，区别于等效高斯噪声近似。上行干扰通过邻区 UE 按 3GPP TS 38.211"),
    refTag("[4]"),
    cn(" §6.4.1.4 标准 SRS 序列发送实现，各 UE 使用独立的 n_SRS_ID 和循环移位。下行干扰通过邻区各自 PCI 对应的 CSI-RS Gold 序列叠加实现。LS 估计器使用服务小区导频解调，干扰残留在估计信道中，真实反映干扰对信道估计的影响。"),
  ]));

  children.push(h3("3.5.1  邻区预编码投影"));
  children.push(bodyPara([
    cn("在实际系统中，邻区基站会对其调度用户施加预编码，因此对目标用户的干扰并非全秩。平台通过邻区预编码投影机制模拟这一效果：对每个邻区 BS_k，生成其到自身调度用户 P_k 的下行信道 H(BS_k→P_k)，通过 SVD 计算预编码矩阵 W_k，然后将干扰信道 H(BS_k→Q) 投影到 W_k 的列空间中。投影后干扰功率降低约 10·lg(rank/BS_ant) dB。"),
  ]));

  // -- Fig: Interference model --
  children.push(figImage("fig_interference.png", 420, 380));
  children.push(figCaption("图 3-1  邻区 DL 预编码投影干扰建模示意图"));

  children.push(h3("3.5.2  多 UE 干扰信道"));
  children.push(bodyPara([
    cn("上行干扰建模使用独立的 per-UE 信道。每个邻区生成最多 num_interfering_ues 个独立 UE，各 UE 位置在邻区覆盖范围内均匀分布，并独立生成 H(UE_kn→BS_serving) 信道。每个样本随机激活部分 UE 以模拟调度碰撞的不确定性。"),
  ]));

  children.push(h2("3.6  SRS 频域跳频"));
  children.push(bodyPara([
    cn("SRS 频域跳频严格按照 3GPP TS 38.211"),
    refTag("[4]"),
    cn(" §6.4.1.4.3 的带宽树结构实现。配置索引 C_SRS 确定带宽树参数 m_SRS[0..3] 和 N[0..3]，层级 B_SRS 决定每次发送的 RB 数。SRS 在不同时隙跳到不同频率位置，完整跳频周期覆盖 m_SRS[0] 个 RB。基站累积一个完整跳频周期的所有 SRS 观测后再进行信道估计，避免在 SRS 带宽内做插值。"),
  ]));
  children.push(tableCaption("表 3-3  SRS 配置参数"));
  children.push(makeTable(
    ["参数", "默认值", "3GPP 来源", "说明"],
    [
      ["srs_periodicity", "10", "Table 6.4.1.4.4-1", "SRS 周期（时隙）"],
      ["srs_comb", "2", "§6.4.1.4.2", "传输梳齿 K_TC ∈ {2, 4, 8}"],
      ["srs_c_srs", "3", "Table 6.4.1.4.3-1", "带宽配置索引"],
      ["srs_b_srs", "1", "§6.4.1.4.3", "带宽层级 0~3"],
      ["srs_b_hop", "0", "§6.4.1.4.3", "频域跳频参数"],
      ["srs_n_rrc", "0", "§6.4.1.4.3", "频域起始 RB 位置"],
    ],
    [1600, 1200, 2306, 3200],
  ));

  children.push(h2("3.7  DL 预编码"));
  children.push(bodyPara([
    cn("TDD 系统下，基站通过上行 SRS 估计信道，利用 TDD 互易性推导下行信道，再通过 SVD 计算预编码权值。具体流程为：（1）从 SRS 导频估计 H_UL_est；（2）利用 TDD 互易 H_DL = conj(H_UL)；（3）对 H_DL 进行 SVD 分解，取 W_DL = U[:, :rank]；（4）基于奇异值分布自动确定传输 rank（σ_i > 0.1·σ_max 的层数）。预编码矩阵 W_DL 的 shape 为 [RB, BS_ant, rank]，每个 RB 独立计算。"),
  ]));

  children.push(h2("3.8  SSB 波束管理"));
  children.push(bodyPara([
    cn("平台实现了 SSB 波束扫描测量，支持 4/8/16 波束配置。每个小区使用 DFT 波束权值进行波束扫描，计算各波束的 SS-RSRP、RSRQ 和 SS-SINR，选取最佳波束。测量结果存储在 ChannelSample 的 SSB 字段中，可用于波束管理、小区选择等 ML 任务。"),
  ]));

  children.push(h2("3.9  面阵天线模型"));
  children.push(bodyPara([
    cn("平台支持 3GPP 定义的双极化面阵天线（Panel Array）模型，取代传统单极化 ULA。面阵配置通过 [N_H, N_V, N_P] 三元组指定，如 [8, 4, 2] 表示 8 列×4 行×2 极化 = 64 天线端口。空间相关矩阵采用 Kronecker 结构 R = R_H ⊗ R_V ⊗ R_P，其中 R_H/R_V 使用指数衰减模型 ρ^|i-j|，R_P 为极化相关矩阵 [[1, μ], [μ, 1]]，μ = 10^(-XPD/10)。"),
  ]));

  children.push(h2("3.10  移动性建模"));
  children.push(bodyPara([
    cn("平台支持五种 UE 移动模式，覆盖静止、匀速、随机游走、随机航路点和轨道固定轨迹等典型场景。轨迹生成模块为每个 UE 生成连续位置序列，多普勒频移从位置差分自动推导。"),
  ]));
  children.push(tableCaption("表 3-4  移动模式"));
  children.push(makeTable(
    ["模式", "运动特征", "适用场景"],
    [
      ["static", "静止，UE 位置固定", "室内固定终端、基准测试"],
      ["linear", "匀速直线运动，随机方向", "高速公路、直线道路"],
      ["random_walk", "随机游走，每步随机转向", "步行用户、城市漫游"],
      ["random_waypoint", "随机航路点模型", "通用移动场景"],
      ["track", "沿轨道中心线匀速行驶", "高铁场景"],
    ],
    [1800, 2800, 3706],
  ));

  children.push(h2("3.11  信道固有属性保障"));
  children.push(bodyPara([
    cn("为确保生成信道数据的物理可信性，平台在信道生成阶段实施了多项保障机制，覆盖空间一致性、时频互易性、LOS 判定和信道溯源等关键属性。"),
  ]));
  children.push(tableCaption("表 3-5  信道固有属性保障机制"));
  children.push(makeTable(
    ["属性", "保障机制", "技术要点"],
    [
      ["空间一致性", "位置量化确定性采样", "大尺度参数使用位置量化到相关距离网格的确定性 RNG"],
      ["TDD 互易性", "频域平滑校准噪声", "H_UL = conj(H_DL^T) + 频域低秩插值噪声"],
      ["per-cell LOS", "独立 LOS 判定", "每小区独立计算 LOS 概率，避免错误路损选择"],
      ["信道溯源", "generation_mode 标记", "区分 sionna_rt 和 tdl_fallback 信道"],
      ["理想信道纯净性", "无干扰保证", "h_serving_true 取自单 BS→UE 纯净信道"],
    ],
    [1500, 2000, 4806],
  ));

  children.push(bodyPara([
    cn("大尺度参数的空间相关距离遵循 3GPP 38.901"),
    refTag("[3]"),
    cn(" Table 7.5-6 规定。UMa-NLOS 场景下时延扩展（DS）的相关距离为 40 m，UMi-NLOS 为 10 m。平台将 UE 位置量化到对应场景的最小相关距离网格，确保网格内的相邻采样点共享相同的大尺度参数。"),
  ]));

  children.push(new Paragraph({ children: [new PageBreak()] }));

  // ===== CHAPTER 4: DATA MANAGEMENT =====
  children.push(h1("第四章  数据管理"));

  children.push(h2("4.1  数据集列表与详情"));
  children.push(bodyPara([
    cn("数据集页面（/datasets）按数据源聚合显示统计信息，包括每个数据源的样本总数、SNR/SIR/SINR 均值、链路类型分布和处理阶段分布（raw/bridged）。点击数据源可进入详情页，查看 SINR 分布直方图和样本级明细。"),
  ]));

  children.push(h2("4.2  信道样本结构"));
  children.push(bodyPara([
    cn("每条信道记录（ChannelSample）包含信道矩阵、标量指标、SSB 测量和元数据四大类字段。核心信道数据的张量维度为 [T, RB, BS_ant, UE_ant]，其中 T 为 OFDM 符号数，RB 为资源块数，BS_ant 和 UE_ant 分别为基站和终端天线数。"),
  ]));
  children.push(tableCaption("表 4-1  ChannelSample 核心字段"));
  children.push(makeTable(
    ["字段", "类型", "Shape", "说明"],
    [
      ["h_serving_true", "complex64", "[T, RB, BS, UE]", "理想服务小区信道（无干扰）"],
      ["h_serving_est", "complex64", "[T, RB, BS, UE]", "含干扰的估计服务小区信道"],
      ["h_interferers", "complex64", "[K-1, T, RB, BS, UE]", "干扰小区信道（可选）"],
      ["snr_dB", "float", "标量", "信噪比"],
      ["sir_dB", "float|None", "标量", "信干比"],
      ["sinr_dB", "float", "标量", "信干噪比"],
      ["ue_position", "float64", "[3]", "UE 位置 (x, y, z) 米"],
      ["w_dl", "complex64", "[RB, BS, rank]", "DL 预编码权值矩阵"],
    ],
    [1600, 1200, 2506, 3000],
  ));

  children.push(h2("4.3  信道浏览器"));
  children.push(bodyPara([
    cn("信道浏览器（/channels）提供 Bridge 处理后样本的交互式可视化功能。用户可浏览信道幅度热力图（理想信道、估计信道、误差信道），查看 16 个 Token 的特征分布，以及 SNR/SIR/SINR 等元数据。"),
  ]));

  children.push(h2("4.4  Bridge 特征提取"));
  children.push(bodyPara([
    cn("Bridge 模块将原始 ChannelSample 转换为定长 Token 序列，作为下游 ChannelMAE 编码器的输入。处理流程分为三个阶段：（1）特征提取：从信道矩阵计算 16 个 Token 字段和 8 个门控字段；（2）归一化：物理值映射到 [-1, 1] 区间；（3）嵌入：归一化特征经门控和线性投影生成 [B, 16, 128] 的 Token 序列。"),
  ]));
  children.push(tableCaption("表 4-2  16 Token 特征说明"));
  children.push(makeTable(
    ["Token 索引", "名称", "来源", "物理含义"],
    [
      ["0", "PDP", "UL 信道", "功率时延谱"],
      ["1~4", "SRS", "UL 信道", "SRS 信道估计特征"],
      ["5~8", "PMI", "DL 信道", "38.214 Type I 码本搜索"],
      ["9~12", "DFT", "UL 信道", "DFT 域信道表示"],
      ["13", "RSRP_SRS", "UL 信道", "SRS 参考信号功率"],
      ["14", "RSRP_CB", "UL 信道", "码本参考信号功率"],
      ["15", "Cell RSRP", "SSB 测量", "服务小区 SS-RSRP"],
    ],
    [1200, 1400, 1400, 4306],
  ));

  children.push(h2("4.5  数据集划分与锁定"));
  children.push(bodyPara([
    cn("平台提供 train/val/test 三分法的数据集划分功能。划分支持三种策略：随机划分（random）、按位置划分（by_position）和按波束划分（by_beam）。测试集一旦锁定，后续新采集的数据自动进入训练集，不影响已锁定的测试集，确保评估基准的一致性。"),
  ]));
  children.push(bodyPara([
    cn("锁定操作通过前端“数据集划分与导出”卡片完成，用户选择划分策略、随机种子和比例后，可通过开关控制是否立即锁定。锁定后 split_version 自增，评估报告绑定版本号。解锁操作需要二次确认。"),
  ]));

  children.push(h2("4.6  数据导出"));
  children.push(bodyPara([
    cn("平台支持三种导出格式，每种格式的导出包均为自包含（含契约版本、Split 版本号、维度信息和使用示例）。"),
  ]));
  children.push(tableCaption("表 4-3  数据导出格式"));
  children.push(makeTable(
    ["格式", "文件类型", "特点", "适用场景"],
    [
      ["HDF5", ".h5 单文件", "gzip 压缩，跨语言", "PyTorch/JAX 训练"],
      ["WebDataset", ".tar 分片", "流式读取，支持分片", "大规模分布式训练"],
      ["pt_dir", "目录 + .pt 文件", "与平台内部格式一致", "快速原型开发"],
    ],
    [1200, 1600, 2706, 2800],
  ));

  children.push(new Paragraph({ children: [new PageBreak()] }));

  // ===== CHAPTER 5: MODEL MANAGEMENT =====
  children.push(h1("第五章  模型管理与评估"));

  children.push(h2("5.1  模型上传"));
  children.push(bodyPara([
    cn("模型仓库页面（/models）提供模型上传入口。用户通过拖拽上传 .pt/.pth/.ckpt 格式的 Checkpoint 文件，系统自动执行 ChannelMAE 兼容性验证（检查 encoder/decoder/latent_proj 等关键 key），并创建 Run 记录和 ModelArtifact 注册。上传完成后可选择立即在锁定测试集上触发评估。"),
  ]));

  children.push(h2("5.2  模型评估"));
  children.push(bodyPara([
    cn("评估流程在锁定的测试集上执行，主要计算信道图谱（Channel Charting）相关指标，包括 KNN 一致性准确率（KNN Acc）、归一化均方误差（NMSE）、可信度（Trustworthiness）和连续性（Continuity）等。评估结果以 metrics.json 形式存储，嵌入向量以 embeddings.parquet 格式输出。"),
  ]));

  children.push(h2("5.3  模型推理"));
  children.push(bodyPara([
    cn("推理功能允许用户在指定数据集分区上批量运行模型，生成信道嵌入向量。用户可选择数据划分（train/val/test）、batch size 和样本数量上限。推理结果可用于下游任务，如信道图谱构建、位置估计和波束预测等。"),
  ]));

  children.push(h2("5.4  模型排行榜"));
  children.push(bodyPara([
    cn("排行榜（/models → 排行榜 Tab）汇总所有已评估模型的指标，默认按 KNN Acc 降序排列。表格包含排名、运行 ID、标签、兼容性状态、KNN Acc、NMSE (dB)、Continuity、Trustworthiness 和评估时间等列。前三名以金/银/铜色标签标注。测试集版本号绑定每次评估，确保不同版本测试集的评估结果可追溯。"),
  ]));

  children.push(h2("5.5  端到端工作流"));
  children.push(bodyPara([
    cn("平台的完整工作流如下："),
  ]));
  children.push(numbered("通过数据采集向导采集多源信道数据", "numbers2"));
  children.push(numbered("划分数据集并锁定测试集（版本 v1）", "numbers2"));
  children.push(numbered("将训练集导出为 HDF5/WebDataset 格式，供外部平台下载", "numbers2"));
  children.push(numbered("在外部平台完成模型训练", "numbers2"));
  children.push(numbered("继续采集新数据，自动归入训练集（测试集不变）", "numbers2"));
  children.push(numbered("将训练好的 Checkpoint 上传至平台", "numbers2"));
  children.push(numbered("平台在锁定测试集 v1 上自动评估，输出指标", "numbers2"));
  children.push(numbered("通过排行榜对比选取最优模型", "numbers2"));

  // -- Fig: Platform workflow --
  children.push(figImage("fig_workflow.png", 560, 195));
  children.push(figCaption("图 5-1  平台端到端工作流"));

  children.push(new Paragraph({ children: [new PageBreak()] }));

  // ===== CHAPTER 6: TASK SYSTEM =====
  children.push(h1("第六章  任务系统"));

  children.push(h2("6.1  任务类型"));
  children.push(bodyPara([
    cn("平台通过异步任务系统管理所有耗时操作。每种任务类型对应一个 Dramatiq Actor，通过 Redis 消息队列分发至 Worker 进程执行。"),
  ]));
  children.push(tableCaption("表 6-1  任务类型"));
  children.push(makeTable(
    ["类型", "名称", "说明"],
    [
      ["simulate", "信道仿真", "执行信道数据采集"],
      ["convert", "数据转换", "格式转换（如 .mat → .pt）"],
      ["bridge", "特征提取", "Bridge 处理，生成 Token"],
      ["eval", "模型评估", "在测试集上评估模型"],
      ["infer", "模型推理", "批量生成嵌入向量"],
      ["export", "模型导出", "导出 ONNX/TorchScript 格式"],
      ["report", "报告生成", "生成评估报告"],
      ["dataset_export", "数据集导出", "导出 HDF5/WebDataset/pt_dir"],
    ],
    [1600, 1600, 5106],
  ));

  children.push(h2("6.2  任务管理"));
  children.push(bodyPara([
    cn("任务页面（/jobs）展示所有任务的状态列表，支持按类型和状态筛选。任务创建页面（/jobs/new）提供表单式任务配置，支持批量运行模式。任务详情页面（/jobs/:jobId）显示实时日志输出和运行指标。任务状态包括 pending（等待）、running（运行中）、done（完成）和 failed（失败）四种。"),
  ]));

  children.push(new Paragraph({ children: [new PageBreak()] }));

  // ===== CHAPTER 7: API REFERENCE =====
  children.push(h1("第七章  API 参考"));
  children.push(bodyPara([
    cn("平台后端提供 RESTful API，基于 FastAPI 自动生成 OpenAPI 文档（/docs）。以下列出核心 API 端点。"),
  ]));

  children.push(h2("7.1  数据集 API"));
  children.push(tableCaption("表 7-1  数据集相关 API"));
  children.push(makeTable(
    ["方法", "路径", "说明"],
    [
      ["GET", "/api/datasets", "数据源聚合列表"],
      ["GET", "/api/datasets/:source/samples", "样本分页查询"],
      ["POST", "/api/datasets/collect", "触发数据采集任务"],
      ["DELETE", "/api/datasets/:source", "删除数据源"],
      ["GET", "/api/datasets/split/status", "查看 Split 状态"],
      ["POST", "/api/datasets/split", "计算划分并可选锁定"],
      ["POST", "/api/datasets/export", "提交数据集导出任务"],
      ["GET", "/api/datasets/exports", "列出已完成的导出包"],
      ["GET", "/api/datasets/exports/:name/download", "下载导出文件"],
    ],
    [800, 3706, 3800],
  ));

  children.push(h2("7.2  模型 API"));
  children.push(tableCaption("表 7-2  模型相关 API"));
  children.push(makeTable(
    ["方法", "路径", "说明"],
    [
      ["GET", "/api/models", "模型列表"],
      ["POST", "/api/models/upload", "上传模型 Checkpoint"],
      ["POST", "/api/models/{run_id}/evaluate", "触发模型评估"],
      ["POST", "/api/models/{run_id}/infer", "触发模型推理"],
      ["GET", "/api/models/{run_id}/meta", "获取模型元数据"],
      ["GET", "/api/models/leaderboard", "模型排行榜"],
    ],
    [800, 3706, 3800],
  ));

  children.push(h2("7.3  任务与其他 API"));
  children.push(tableCaption("表 7-3  任务与辅助 API"));
  children.push(makeTable(
    ["方法", "路径", "说明"],
    [
      ["GET", "/api/jobs", "任务列表"],
      ["POST", "/api/jobs", "创建任务"],
      ["GET", "/api/jobs/:jobId", "任务详情与日志"],
      ["POST", "/api/topology/preview", "拓扑预览渲染"],
      ["GET", "/api/channels", "Bridge 处理后的样本列表"],
      ["GET", "/api/channels/:index", "单个样本完整数据"],
    ],
    [800, 3706, 3800],
  ));

  children.push(new Paragraph({ children: [new PageBreak()] }));

  // ===== CHAPTER 8: EXTERNAL TRAINING =====
  children.push(h1("第八章  外部训练对接"));

  children.push(h2("8.1  数据导出与下载"));
  children.push(bodyPara([
    cn("导出完成后，用户可通过 API 或前端直接下载导出文件。HDF5 文件直接下载，目录格式（pt_dir）自动打包为 .zip。前端导出列表已集成下载链接。"),
  ]));

  children.push(h2("8.2  训练数据消费"));
  children.push(bodyPara([
    cn("以 HDF5 格式为例，外部训练平台可通过以下代码加载训练数据："),
  ]));
  children.push(new Paragraph({
    spacing: { before: 120, after: 120 },
    indent: { left: 480 },
    children: [new TextRun({ text: 'with h5py.File("msg_export_train_hdf5.h5", "r") as f:', font: "Consolas", size: PT(9) })],
  }));
  children.push(new Paragraph({
    indent: { left: 480 },
    children: [new TextRun({ text: '    sample = ChannelSample.from_hdf5_group(f["samples/0"])', font: "Consolas", size: PT(9) })],
  }));
  children.push(new Paragraph({
    indent: { left: 480 },
    children: [new TextRun({ text: '    h_true = sample.h_serving_true  # [T, RB, BS, UE]', font: "Consolas", size: PT(9) })],
  }));
  children.push(new Paragraph({
    spacing: { after: 120 },
    indent: { left: 480 },
    children: [new TextRun({ text: '    h_est  = sample.h_serving_est   # with interference', font: "Consolas", size: PT(9) })],
  }));

  children.push(h2("8.3  模型回传与评估"));
  children.push(bodyPara([
    cn("训练完成后，用户通过 /api/models/upload 端点上传 Checkpoint 文件。上传时系统自动验证 ChannelMAE 兼容性。验证通过后可调用 /api/models/{run_id}/evaluate 在锁定测试集上触发评估。评估完成后结果自动进入排行榜。"),
  ]));

  children.push(h2("8.4  兼容性要求"));
  children.push(bodyPara([
    cn("上传的 Checkpoint 须满足以下条件：（1）文件格式为 PyTorch 的 .pt/.pth/.ckpt；（2）state_dict 中包含 ChannelMAE 的 encoder、decoder 和 latent_proj 相关 key；（3）或以 {\"model\": state_dict} 格式封装。不满足条件的 Checkpoint 仍可上传，但会被标记为“兼容性未确认”。"),
  ]));

  children.push(new Paragraph({ children: [new PageBreak()] }));

  // ===== REFERENCES =====
  children.push(h1("参考文献"));
  const refs = [
    ["[1]", "3GPP TS 38.101-1, \"NR; User Equipment (UE) radio transmission and reception; Part 1: Range 1 Standalone,\" V18.6.0, 2024."],
    ["[2]", "3GPP TS 38.213, \"NR; Physical layer procedures for control,\" V18.5.0, 2024."],
    ["[3]", "3GPP TR 38.901, \"Study on channel model for frequencies from 0.5 to 100 GHz,\" V17.1.0, 2023."],
    ["[4]", "3GPP TS 38.211, \"NR; Physical channels and modulation,\" V18.5.0, 2024."],
    ["[5]", "3GPP TS 38.214, \"NR; Physical layer procedures for data,\" V18.5.0, 2024."],
    ["[6]", "3GPP TR 38.913, \"Study on scenarios and requirements for next generation access technologies,\" V17.0.0, 2022."],
    ["[7]", "3GPP TS 38.331, \"NR; Radio Resource Control (RRC) protocol specification,\" V18.5.0, 2024."],
  ];
  refs.forEach(([tag, text]) => {
    children.push(new Paragraph({
      spacing: { line: 340, after: 60 },
      indent: { left: 480, hanging: 480 },
      children: [
        en(tag + " ", { size: PT(10.5) }),
        en(text, { size: PT(10.5) }),
      ],
    }));
  });

  return children;
}

// ========== BUILD DOCUMENT ==========
const doc = new Document({
  styles: {
    default: {
      document: { run: { font: FONT_SONG, size: PT(12) } },
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: PT(16), bold: true, font: FONT_HEI },
        paragraph: { spacing: { before: 360, after: 240 }, outlineLevel: 0, alignment: AlignmentType.CENTER },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: PT(14), bold: true, font: FONT_HEI },
        paragraph: { spacing: { before: 280, after: 180 }, outlineLevel: 1 },
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: PT(12), bold: true, font: FONT_HEI },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 },
      },
    ],
  },
  numbering: numberingConfig,
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: PAGE_H },
        margin: { top: MT, bottom: MB, left: ML, right: MR },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "ChannelHub 5G NR 信道数据工场平台说明书", font: FONT_SONG, size: PT(9), color: "888888" })],
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC", space: 4 } },
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT_EN, size: PT(10) })],
        })],
      }),
    },
    children: buildContent(),
  }],
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("D:/MSG平台_cc/docs/ChannelHub平台说明书.docx", buffer);
  console.log("Done: ChannelHub平台说明书.docx generated successfully");
});
