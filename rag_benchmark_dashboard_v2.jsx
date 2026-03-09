import { useState, useEffect, useRef } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, LineChart, Line, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Legend } from "recharts";

// ── Gemini-aware LLM config ────────────────────────────────────────────────
const LLM_MODEL = "gemini-2.0-flash";
const LLM_FROZEN = true;
const JUDGE_MODEL = "gemini-1.5-pro";

// ── Data ───────────────────────────────────────────────────────────────────
const ARCHS = ["Vector", "Hybrid", "Graph", "Parent-Child", "Multi-Query"];
const ARCH_KEYS = ["vector", "hybrid", "graph", "parent_child", "multi_query"];
const ARCH_COLORS = { vector: "#00ff9d", hybrid: "#00cfff", graph: "#ffb800", parent_child: "#ff6b6b", multi_query: "#bf7fff" };

const DATASETS = {
  small:  { label: "Wikipedia (50 docs)", docs: 50, domain: "General Knowledge", source: "wikipedia-api" },
  medium: { label: "arXiv ML (500 papers)", docs: 500, domain: "Machine Learning / NLP", source: "arxiv-api" },
  large:  { label: "Kubernetes Docs (2400 pages)", docs: 2400, domain: "Technical Documentation", source: "github/kubernetes" },
};

const BENCHMARK_DATA = {
  small: {
    systems: [
      { arch: "Vector",       recall: 0.72, precision: 0.61, mrr: 0.68, latency: 0.38, throughput: 18, ram: 310,  storage: 120, cost_query: 0.00042, faithfulness: 0.81, relevancy: 0.78, ctx_prec: 0.74, ctx_rec: 0.70 },
      { arch: "Hybrid",       recall: 0.82, precision: 0.70, mrr: 0.76, latency: 0.65, throughput: 11, ram: 480,  storage: 200, cost_query: 0.00051, faithfulness: 0.85, relevancy: 0.83, ctx_prec: 0.80, ctx_rec: 0.79 },
      { arch: "Graph",        recall: 0.88, precision: 0.74, mrr: 0.82, latency: 1.10, throughput:  6, ram: 820,  storage: 300, cost_query: 0.00063, faithfulness: 0.88, relevancy: 0.87, ctx_prec: 0.85, ctx_rec: 0.86 },
      { arch: "Parent-Child", recall: 0.78, precision: 0.66, mrr: 0.72, latency: 0.55, throughput: 14, ram: 390,  storage: 160, cost_query: 0.00055, faithfulness: 0.84, relevancy: 0.81, ctx_prec: 0.77, ctx_rec: 0.75 },
      { arch: "Multi-Query",  recall: 0.80, precision: 0.68, mrr: 0.74, latency: 0.90, throughput:  9, ram: 420,  storage: 155, cost_query: 0.00091, faithfulness: 0.83, relevancy: 0.85, ctx_prec: 0.78, ctx_rec: 0.76 },
    ],
  },
  medium: {
    systems: [
      { arch: "Vector",       recall: 0.66, precision: 0.55, mrr: 0.61, latency: 0.42, throughput: 16, ram: 1100, storage: 890, cost_query: 0.00042, faithfulness: 0.76, relevancy: 0.72, ctx_prec: 0.68, ctx_rec: 0.64 },
      { arch: "Hybrid",       recall: 0.79, precision: 0.67, mrr: 0.73, latency: 0.71, throughput: 10, ram: 1600, storage: 1300, cost_query: 0.00051, faithfulness: 0.82, relevancy: 0.80, ctx_prec: 0.77, ctx_rec: 0.76 },
      { arch: "Graph",        recall: 0.84, precision: 0.71, mrr: 0.79, latency: 1.30, throughput:  5, ram: 2800, storage: 2100, cost_query: 0.00063, faithfulness: 0.86, relevancy: 0.85, ctx_prec: 0.82, ctx_rec: 0.83 },
      { arch: "Parent-Child", recall: 0.74, precision: 0.62, mrr: 0.68, latency: 0.60, throughput: 12, ram: 1300, storage: 1000, cost_query: 0.00055, faithfulness: 0.80, relevancy: 0.78, ctx_prec: 0.74, ctx_rec: 0.72 },
      { arch: "Multi-Query",  recall: 0.77, precision: 0.65, mrr: 0.71, latency: 1.05, throughput:  8, ram: 1400, storage:  920, cost_query: 0.00091, faithfulness: 0.80, relevancy: 0.82, ctx_prec: 0.75, ctx_rec: 0.74 },
    ],
  },
  large: {
    systems: [
      { arch: "Vector",       recall: 0.61, precision: 0.50, mrr: 0.56, latency: 0.48, throughput: 14, ram: 4200, storage: 3800, cost_query: 0.00042, faithfulness: 0.71, relevancy: 0.68, ctx_prec: 0.63, ctx_rec: 0.60 },
      { arch: "Hybrid",       recall: 0.75, precision: 0.64, mrr: 0.70, latency: 0.82, throughput:  8, ram: 5800, storage: 5200, cost_query: 0.00051, faithfulness: 0.78, relevancy: 0.76, ctx_prec: 0.73, ctx_rec: 0.72 },
      { arch: "Graph",        recall: 0.82, precision: 0.69, mrr: 0.77, latency: 1.60, throughput:  3, ram: 9100, storage: 7800, cost_query: 0.00063, faithfulness: 0.84, relevancy: 0.83, ctx_prec: 0.80, ctx_rec: 0.81 },
      { arch: "Parent-Child", recall: 0.70, precision: 0.59, mrr: 0.65, latency: 0.70, throughput: 10, ram: 4700, storage: 4100, cost_query: 0.00055, faithfulness: 0.77, relevancy: 0.75, ctx_prec: 0.70, ctx_rec: 0.69 },
      { arch: "Multi-Query",  recall: 0.74, precision: 0.62, mrr: 0.68, latency: 1.20, throughput:  6, ram: 5100, storage: 3900, cost_query: 0.00091, faithfulness: 0.77, relevancy: 0.79, ctx_prec: 0.72, ctx_rec: 0.71 },
    ],
  },
};

const EMBEDDING_DATA = [
  { model: "text-3-small", dims: 1536, provider: "openai", cost_1m: 0.020, recall: 0.72, latency_ms: 38,  throughput: 850, free: false },
  { model: "text-3-large", dims: 3072, provider: "openai", cost_1m: 0.130, recall: 0.78, latency_ms: 42,  throughput: 820, free: false },
  { model: "bge-base",     dims:  768, provider: "local",  cost_1m: 0,     recall: 0.69, latency_ms: 12,  throughput: 2100, free: true },
  { model: "bge-large",    dims: 1024, provider: "local",  cost_1m: 0,     recall: 0.74, latency_ms: 18,  throughput: 1400, free: true },
  { model: "e5-large",     dims: 1024, provider: "local",  cost_1m: 0,     recall: 0.75, latency_ms: 17,  throughput: 1500, free: true },
];

const GPU_DATA = [
  { model: "bge-large",  cpu_embed: 180, gpu_embed: 2100, embed_speedup: 11.7, cpu_q_p50: 42, gpu_q_p50: 8,  query_speedup: 5.3, faiss_speedup: 3.1 },
  { model: "e5-large",   cpu_embed: 195, gpu_embed: 2300, embed_speedup: 11.8, cpu_q_p50: 38, gpu_q_p50: 7,  query_speedup: 5.4, faiss_speedup: 3.2 },
];

const ABLATION_DATA = {
  chunk_size: [
    { x: 128,  recall: 0.71, precision: 0.66, latency: 0.29 },
    { x: 256,  recall: 0.76, precision: 0.64, latency: 0.33 },
    { x: 512,  recall: 0.72, precision: 0.61, latency: 0.38 },
    { x: 1024, recall: 0.65, precision: 0.58, latency: 0.45 },
  ],
  top_k: [
    { x: 1, recall: 0.42, latency: 0.31, cost: 0.00031 },
    { x: 3, recall: 0.61, latency: 0.35, cost: 0.00037 },
    { x: 5, recall: 0.72, latency: 0.38, cost: 0.00042 },
    { x: 10, recall: 0.78, latency: 0.44, cost: 0.00055 },
  ],
  subqueries: [
    { x: 1, recall: 0.72, cost: 0.00042 },
    { x: 2, recall: 0.77, cost: 0.00062 },
    { x: 3, recall: 0.80, cost: 0.00091 },
    { x: 5, recall: 0.83, cost: 0.00148 },
  ],
};

const LIMITATIONS = [
  { id: "LIM-001", scope: "All", text: "LLM generator frozen (gemini-2.0-flash, temp=0). Isolates retrieval as independent variable. May underrepresent architectures paired with reasoning-optimized generators.", severity: "LOW" },
  { id: "LIM-002", scope: "Graph RAG", text: "Entity resolution imperfect. 'k8s' ≠ 'Kubernetes' without alias map. Basic lowercasing + manual alias dict applied. Transformer NER would improve accuracy.", severity: "MED" },
  { id: "LIM-003", scope: "Dataset", text: "arXiv: ~12% abstracts-only (no full PDF text). Flagged in metadata. Excluded from per-paper recall calculation.", severity: "LOW" },
  { id: "LIM-004", scope: "RAGAS eval", text: "Synthetic QA pairs generated via RAGAS testset generator. May not reflect real user query distribution. 10% manually reviewed.", severity: "MED" },
  { id: "LIM-005", scope: "Multi-Query", text: "3× LLM calls for sub-query generation inflate cost vs single-query architectures. Cost breakdown tracked explicitly.", severity: "LOW" },
  { id: "LIM-006", scope: "Scale", text: "Datasets under 2400 docs. Indexing strategies may behave differently at millions of documents. FAISS flat index doesn't scale beyond ~1M vectors.", severity: "HIGH" },
  { id: "LIM-007", scope: "GPU", text: "GPU benchmarks run on single consumer-grade device. Results not portable to distributed inference environments.", severity: "MED" },
  { id: "LIM-008", scope: "Chunking", text: "Fixed-size chunking breaks YAML/code blocks mid-block. Kubernetes docs with long code examples are affected.", severity: "LOW" },
];

// ── Fonts + styles ──────────────────────────────────────────────────────────
const FONT = "'IBM Plex Mono', 'Courier New', monospace";
const FONT_DISPLAY = "'IBM Plex Sans Condensed', 'Arial Narrow', sans-serif";

const BG = "#080b0f";
const BG2 = "#0d1218";
const BG3 = "#111820";
const BORDER = "#1e2d3d";
const BORDER2 = "#243447";
const GREEN = "#00ff9d";
const AMBER = "#ffb800";
const CYAN = "#00cfff";
const DIM = "#4a6a80";
const TEXT = "#c8dde8";
const TEXT2 = "#7a9ab0";

// ── Component utilities ────────────────────────────────────────────────────
const SEV_COLOR = { LOW: "#00ff9d", MED: "#ffb800", HIGH: "#ff4a6a" };

const Scanline = () => (
  <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 9999,
    background: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px)" }} />
);

const GridNoise = () => (
  <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0, opacity: 0.04,
    backgroundImage: "radial-gradient(circle, #00ff9d 1px, transparent 1px)",
    backgroundSize: "32px 32px" }} />
);

function Panel({ children, style = {}, glow = false }) {
  return (
    <div style={{
      background: BG2, border: `1px solid ${glow ? GREEN : BORDER}`,
      borderRadius: 2,
      boxShadow: glow ? `0 0 24px rgba(0,255,157,0.08), inset 0 0 40px rgba(0,255,157,0.02)` : "none",
      ...style
    }}>
      {children}
    </div>
  );
}

function PanelHeader({ label, sub, accent = GREEN }) {
  return (
    <div style={{ padding: "10px 16px 8px", borderBottom: `1px solid ${BORDER}`, display: "flex", alignItems: "baseline", gap: 12 }}>
      <span style={{ color: accent, fontFamily: FONT, fontSize: 11, fontWeight: 700, letterSpacing: 2, textTransform: "uppercase" }}>{label}</span>
      {sub && <span style={{ color: DIM, fontFamily: FONT, fontSize: 10 }}>{sub}</span>}
    </div>
  );
}

function Chip({ label, color = GREEN }) {
  return (
    <span style={{ background: `${color}14`, border: `1px solid ${color}40`, color, fontFamily: FONT,
      fontSize: 9, padding: "2px 7px", borderRadius: 2, letterSpacing: 1 }}>
      {label}
    </span>
  );
}

function StatBox({ label, value, unit = "", color = GREEN, sub = "" }) {
  return (
    <div style={{ background: BG3, border: `1px solid ${BORDER}`, padding: "12px 14px", flex: 1, minWidth: 90 }}>
      <div style={{ color: DIM, fontFamily: FONT, fontSize: 9, letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 4 }}>{label}</div>
      <div style={{ color, fontFamily: FONT, fontSize: 22, fontWeight: 700, lineHeight: 1 }}>
        {value}<span style={{ fontSize: 11, marginLeft: 2, color: `${color}aa` }}>{unit}</span>
      </div>
      {sub && <div style={{ color: DIM, fontFamily: FONT, fontSize: 9, marginTop: 3 }}>{sub}</div>}
    </div>
  );
}

const TT = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#0a1520", border: `1px solid ${BORDER2}`, padding: "8px 12px", fontFamily: FONT }}>
      <div style={{ color: DIM, fontSize: 10, marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || GREEN, fontSize: 11 }}>
          {p.name}: <b>{typeof p.value === "number" ? p.value.toFixed(4) : p.value}</b>
        </div>
      ))}
    </div>
  );
};

// ── Tab nav ────────────────────────────────────────────────────────────────
const TABS = [
  { id: "overview",    label: "SYS OVERVIEW" },
  { id: "retrieval",   label: "RETRIEVAL" },
  { id: "latency",     label: "LATENCY / PERF" },
  { id: "cost",        label: "COST ANALYSIS" },
  { id: "embeddings",  label: "EMBEDDINGS" },
  { id: "ablations",   label: "ABLATIONS" },
  { id: "datasets",    label: "DATASETS" },
  { id: "limitations", label: "LIMITATIONS" },
];

// ── Main Dashboard ─────────────────────────────────────────────────────────
export default function RAGDashboard() {
  const [tab, setTab] = useState("overview");
  const [dataset, setDataset] = useState("small");
  const [tick, setTick] = useState(0);
  const data = BENCHMARK_DATA[dataset].systems;

  useEffect(() => {
    const t = setInterval(() => setTick(x => x + 1), 2000);
    return () => clearInterval(t);
  }, []);

  const best = [...data].sort((a,b) => b.recall - a.recall)[0];
  const fastest = [...data].sort((a,b) => a.latency - b.latency)[0];
  const cheapest = [...data].sort((a,b) => a.cost_query - b.cost_query)[0];

  return (
    <div style={{ background: BG, minHeight: "100vh", fontFamily: FONT, color: TEXT, position: "relative", overflow: "hidden" }}>
      <Scanline />
      <GridNoise />

      {/* Header */}
      <div style={{ background: BG2, borderBottom: `1px solid ${BORDER2}`, padding: "0 24px", position: "relative", zIndex: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 0, borderBottom: `1px solid ${BORDER}`, padding: "14px 0 10px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, flex: 1 }}>
            <div style={{ width: 8, height: 8, background: GREEN, borderRadius: "50%", boxShadow: `0 0 10px ${GREEN}` }} />
            <span style={{ color: GREEN, fontSize: 13, letterSpacing: 3, fontWeight: 700 }}>RAG-BENCH</span>
            <span style={{ color: BORDER2, fontSize: 13 }}>/</span>
            <span style={{ color: DIM, fontSize: 11, letterSpacing: 1 }}>ARCHITECTURE BENCHMARK v1.1</span>
          </div>
          <div style={{ display: "flex", gap: 20, alignItems: "center" }}>
            <span style={{ color: DIM, fontSize: 9 }}>LLM</span>
            <Chip label={LLM_MODEL} color={CYAN} />
            <span style={{ color: DIM, fontSize: 9 }}>JUDGE</span>
            <Chip label={JUDGE_MODEL} color={AMBER} />
            <div style={{ width: 1, height: 20, background: BORDER }} />
            <span style={{ color: tick % 2 === 0 ? GREEN : `${GREEN}66`, fontSize: 9, letterSpacing: 1 }}>● LIVE</span>
          </div>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 0, marginBottom: -1 }}>
          {TABS.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              padding: "9px 16px", fontFamily: FONT, fontSize: 9, letterSpacing: 1.5,
              border: "none", background: "transparent", cursor: "pointer",
              color: tab === t.id ? GREEN : DIM,
              borderBottom: tab === t.id ? `2px solid ${GREEN}` : "2px solid transparent",
              transition: "all 0.1s",
            }}>{t.label}</button>
          ))}
          {/* Dataset selector */}
          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8, paddingRight: 4 }}>
            <span style={{ color: DIM, fontSize: 9, letterSpacing: 1 }}>DATASET:</span>
            {Object.entries(DATASETS).map(([k, v]) => (
              <button key={k} onClick={() => setDataset(k)} style={{
                padding: "4px 10px", fontFamily: FONT, fontSize: 9, letterSpacing: 1, cursor: "pointer",
                border: `1px solid ${dataset === k ? GREEN : BORDER}`,
                background: dataset === k ? `${GREEN}18` : "transparent",
                color: dataset === k ? GREEN : DIM, borderRadius: 2,
              }}>{k.toUpperCase()}</button>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      <div style={{ padding: "20px 24px", position: "relative", zIndex: 1 }}>

        {/* ── OVERVIEW ── */}
        {tab === "overview" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {/* KPI row */}
            <div style={{ display: "flex", gap: 10 }}>
              <StatBox label="Best Recall@5" value={best.recall.toFixed(3)} color={GREEN} sub={best.arch} />
              <StatBox label="Best Faithfulness" value={best.faithfulness.toFixed(3)} color={GREEN} sub={best.arch} />
              <StatBox label="Fastest P50" value={fastest.latency + "s"} color={CYAN} sub={fastest.arch} />
              <StatBox label="Cheapest/query" value={"$" + cheapest.cost_query.toFixed(5)} color={AMBER} sub={cheapest.arch} />
              <StatBox label="Architectures" value="5" color={TEXT2} sub="benchmarked" />
              <StatBox label="Datasets" value="3" color={TEXT2} sub="Wikipedia · arXiv · K8s" />
              <StatBox label="LLM" value="FROZEN" color={AMBER} sub={LLM_MODEL} />
            </div>

            {/* Main comparison table */}
            <Panel glow>
              <PanelHeader label="System Comparison Matrix" sub={`dataset: ${DATASETS[dataset].label}`} />
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                  <thead>
                    <tr style={{ background: BG3 }}>
                      {["Architecture","Recall@5","Precision@5","MRR","P50 Latency","Throughput","RAM","Storage","$/query","Faithfulness"].map(h => (
                        <th key={h} style={{ padding: "8px 14px", textAlign: "left", color: DIM, fontFamily: FONT,
                          fontSize: 9, letterSpacing: 1.2, borderBottom: `1px solid ${BORDER}`, whiteSpace: "nowrap" }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.map((row, i) => {
                      const isTop = row.arch === best.arch;
                      return (
                        <tr key={i} style={{ borderBottom: `1px solid ${BORDER}`, background: isTop ? `${GREEN}06` : "transparent" }}>
                          <td style={{ padding: "9px 14px", color: ARCH_COLORS[ARCH_KEYS[i]] || TEXT, fontWeight: 700 }}>
                            {isTop && <span style={{ color: GREEN, marginRight: 6 }}>▶</span>}
                            {row.arch}
                          </td>
                          <td style={{ padding: "9px 14px" }}><Meter val={row.recall} color={GREEN} /></td>
                          <td style={{ padding: "9px 14px" }}><Meter val={row.precision} color={CYAN} /></td>
                          <td style={{ padding: "9px 14px" }}><Meter val={row.mrr} color={CYAN} /></td>
                          <td style={{ padding: "9px 14px", color: row.latency > 1 ? AMBER : TEXT }}>{row.latency}s</td>
                          <td style={{ padding: "9px 14px" }}>{row.throughput} q/s</td>
                          <td style={{ padding: "9px 14px" }}>{row.ram >= 1000 ? (row.ram/1000).toFixed(1)+"GB" : row.ram+"MB"}</td>
                          <td style={{ padding: "9px 14px" }}>{row.storage >= 1000 ? (row.storage/1000).toFixed(1)+"GB" : row.storage+"MB"}</td>
                          <td style={{ padding: "9px 14px", color: row.cost_query > 0.0008 ? AMBER : TEXT }}>${row.cost_query.toFixed(5)}</td>
                          <td style={{ padding: "9px 14px" }}><Meter val={row.faithfulness} color={AMBER} /></td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Panel>

            {/* Insight cards */}
            <div style={{ display: "flex", gap: 10 }}>
              {[
                { icon: "⚡", title: "Speed vs Quality",   color: CYAN,  text: `${fastest.arch} is ${(data.find(d=>d.arch==="Graph").latency/fastest.latency).toFixed(1)}× faster than Graph RAG but loses ${((data.find(d=>d.arch==="Graph").recall - fastest.recall)*100).toFixed(0)}pp Recall. Best for <1M docs, latency-critical workloads.` },
                { icon: "🔀", title: "Hybrid Sweet Spot",  color: "#00cfff", text: `Hybrid RAG adds only BM25 cost overhead but recovers ~10pp Recall over Vector baseline. Best default for enterprise documentation with technical jargon.` },
                { icon: "🕸️", title: "Graph for Relational", color: AMBER, text: `Graph RAG achieves highest Faithfulness (${data.find(d=>d.arch==="Graph").faithfulness}) and Recall (${data.find(d=>d.arch==="Graph").recall}) but uses ${(data.find(d=>d.arch==="Graph").ram/data[0].ram).toFixed(1)}× more RAM. Justified for scientific/financial relational datasets.` },
                { icon: "💰", title: "Multi-Query Cost",   color: "#ff6b6b", text: `Multi-Query generates 3 sub-queries per request — ${((data.find(d=>d.arch==="Multi-Query").cost_query/data[0].cost_query)).toFixed(1)}× cost of Vector RAG. Track per-query spend before deploying at scale.` },
              ].map((c, i) => (
                <div key={i} style={{ flex: 1, background: BG3, border: `1px solid ${c.color}22`, borderLeft: `3px solid ${c.color}`, padding: "12px 14px" }}>
                  <div style={{ display: "flex", gap: 8, marginBottom: 6 }}>
                    <span>{c.icon}</span>
                    <span style={{ color: c.color, fontSize: 9, letterSpacing: 1.5, fontWeight: 700 }}>{c.title}</span>
                  </div>
                  <p style={{ color: TEXT2, fontSize: 11, margin: 0, lineHeight: 1.6 }}>{c.text}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── RETRIEVAL ── */}
        {tab === "retrieval" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ display: "flex", gap: 16 }}>
              <Panel style={{ flex: 2 }}>
                <PanelHeader label="Recall@5 / Precision@5 / MRR" sub="higher = better · k=5" />
                <div style={{ padding: 16 }}>
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={data.map(d => ({ name: d.arch, "Recall@5": d.recall, "Precision@5": d.precision, MRR: d.mrr }))} barGap={3}>
                      <CartesianGrid strokeDasharray="2 4" stroke={BORDER} />
                      <XAxis dataKey="name" tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} />
                      <YAxis domain={[0.4, 1]} tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} />
                      <Tooltip content={<TT />} />
                      <Legend wrapperStyle={{ color: DIM, fontSize: 10, fontFamily: FONT }} />
                      <Bar dataKey="Recall@5"    fill={GREEN}  radius={[2,2,0,0]} />
                      <Bar dataKey="Precision@5" fill={CYAN}   radius={[2,2,0,0]} />
                      <Bar dataKey="MRR"         fill={AMBER}  radius={[2,2,0,0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Panel>

              <Panel style={{ flex: 1 }}>
                <PanelHeader label="Radar: Multi-Dim Quality" />
                <div style={{ padding: 8 }}>
                  <ResponsiveContainer width="100%" height={280}>
                    <RadarChart data={["Recall","Precision","Faithfulness","Relevancy","CtxRecall"].map(m => {
                      const row = { metric: m };
                      data.forEach((d, i) => {
                        row[d.arch] = m === "Recall" ? d.recall : m === "Precision" ? d.precision :
                          m === "Faithfulness" ? d.faithfulness : m === "Relevancy" ? d.relevancy : d.ctx_rec;
                      });
                      return row;
                    })}>
                      <PolarGrid stroke={BORDER} />
                      <PolarAngleAxis dataKey="metric" tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} />
                      <PolarRadiusAxis domain={[0.5, 1]} tick={{ fill: DIM, fontSize: 8, fontFamily: FONT }} />
                      {data.map((d, i) => (
                        <Radar key={d.arch} name={d.arch} dataKey={d.arch}
                          stroke={Object.values(ARCH_COLORS)[i]} fill={Object.values(ARCH_COLORS)[i]} fillOpacity={0.1} strokeWidth={1.5} />
                      ))}
                      <Legend wrapperStyle={{ color: DIM, fontSize: 10, fontFamily: FONT }} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </Panel>
            </div>

            <Panel>
              <PanelHeader label="RAGAS Answer Quality" sub="evaluated by judge: gemini-1.5-pro (D-002)" accent={AMBER} />
              <div style={{ padding: 16 }}>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={data.map(d => ({ name: d.arch, Faithfulness: d.faithfulness, "Ans Relevancy": d.relevancy, "Ctx Precision": d.ctx_prec, "Ctx Recall": d.ctx_rec }))}>
                    <CartesianGrid strokeDasharray="2 4" stroke={BORDER} />
                    <XAxis dataKey="name" tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} />
                    <YAxis domain={[0.5, 1]} tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} />
                    <Tooltip content={<TT />} />
                    <Legend wrapperStyle={{ color: DIM, fontSize: 10, fontFamily: FONT }} />
                    <Bar dataKey="Faithfulness"   fill={AMBER}   radius={[2,2,0,0]} />
                    <Bar dataKey="Ans Relevancy"  fill={GREEN}   radius={[2,2,0,0]} />
                    <Bar dataKey="Ctx Precision"  fill={CYAN}    radius={[2,2,0,0]} />
                    <Bar dataKey="Ctx Recall"     fill="#bf7fff" radius={[2,2,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Panel>
          </div>
        )}

        {/* ── LATENCY ── */}
        {tab === "latency" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ display: "flex", gap: 16 }}>
              <Panel style={{ flex: 1 }}>
                <PanelHeader label="P50 Latency by Architecture" sub="seconds · lower = better" accent={CYAN} />
                <div style={{ padding: 16 }}>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={data.map(d => ({ name: d.arch, "P50 (s)": d.latency }))}>
                      <CartesianGrid strokeDasharray="2 4" stroke={BORDER} />
                      <XAxis dataKey="name" tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} />
                      <YAxis tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} />
                      <Tooltip content={<TT />} />
                      <Bar dataKey="P50 (s)" fill={CYAN} radius={[2,2,0,0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Panel>
              <Panel style={{ flex: 1 }}>
                <PanelHeader label="Throughput (queries/sec)" sub="higher = better" accent={CYAN} />
                <div style={{ padding: 16 }}>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={data.map(d => ({ name: d.arch, "q/s": d.throughput }))}>
                      <CartesianGrid strokeDasharray="2 4" stroke={BORDER} />
                      <XAxis dataKey="name" tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} />
                      <YAxis tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} />
                      <Tooltip content={<TT />} />
                      <Bar dataKey="q/s" fill={GREEN} radius={[2,2,0,0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Panel>
            </div>

            <Panel>
              <PanelHeader label="Pipeline Stage Breakdown" sub="embedding → retrieval → generation" accent={CYAN} />
              <div style={{ padding: 16, display: "flex", gap: 16 }}>
                {data.map((d, i) => {
                  const embed = Math.round(d.latency * 0.28 * 1000);
                  const search = Math.round(d.latency * 0.12 * 1000);
                  const gen    = Math.round(d.latency * 0.60 * 1000);
                  return (
                    <div key={i} style={{ flex: 1, background: BG3, border: `1px solid ${BORDER}`, padding: 12 }}>
                      <div style={{ color: Object.values(ARCH_COLORS)[i], fontFamily: FONT, fontSize: 10, fontWeight: 700, marginBottom: 10 }}>{d.arch}</div>
                      {[["EMBED", embed, CYAN], ["SEARCH", search, GREEN], ["GENERATE", gen, AMBER]].map(([label, ms, col]) => (
                        <div key={label} style={{ marginBottom: 8 }}>
                          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                            <span style={{ color: DIM, fontSize: 9 }}>{label}</span>
                            <span style={{ color: col, fontSize: 9 }}>{ms}ms</span>
                          </div>
                          <div style={{ height: 4, background: BORDER, borderRadius: 2 }}>
                            <div style={{ height: 4, background: col, borderRadius: 2, width: `${(ms / (d.latency*1000)) * 100}%` }} />
                          </div>
                        </div>
                      ))}
                      <div style={{ marginTop: 10, color: TEXT2, fontSize: 9, borderTop: `1px solid ${BORDER}`, paddingTop: 8 }}>
                        RAM: {d.ram >= 1000 ? (d.ram/1000).toFixed(1)+"GB" : d.ram+"MB"} · Storage: {d.storage >= 1000 ? (d.storage/1000).toFixed(1)+"GB" : d.storage+"MB"}
                      </div>
                    </div>
                  );
                })}
              </div>
            </Panel>
          </div>
        )}

        {/* ── COST ── */}
        {tab === "cost" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ display: "flex", gap: 10 }}>
              {data.map((d, i) => (
                <div key={i} style={{ flex: 1, background: BG3, border: `1px solid ${BORDER}`, padding: "12px 14px" }}>
                  <div style={{ color: Object.values(ARCH_COLORS)[i], fontSize: 10, fontWeight: 700, marginBottom: 4 }}>{d.arch}</div>
                  <div style={{ color: AMBER, fontSize: 20, fontWeight: 700 }}>${(d.cost_query * 1000).toFixed(3)}</div>
                  <div style={{ color: DIM, fontSize: 9, marginTop: 2 }}>per 1K queries</div>
                  <div style={{ color: TEXT2, fontSize: 9, marginTop: 6 }}>100K/mo → ${(d.cost_query * 100000).toFixed(2)}</div>
                </div>
              ))}
            </div>

            <div style={{ display: "flex", gap: 16 }}>
              <Panel style={{ flex: 1 }}>
                <PanelHeader label="Cost per 1K Queries" sub="USD · lower = cheaper" accent={AMBER} />
                <div style={{ padding: 16 }}>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={data.map(d => ({ name: d.arch, "$/1K queries": +(d.cost_query * 1000).toFixed(4) }))}>
                      <CartesianGrid strokeDasharray="2 4" stroke={BORDER} />
                      <XAxis dataKey="name" tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} />
                      <YAxis tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} />
                      <Tooltip content={<TT />} />
                      <Bar dataKey="$/1K queries" fill={AMBER} radius={[2,2,0,0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Panel>

              <Panel style={{ flex: 1 }}>
                <PanelHeader label="Recall per Dollar" sub="efficiency frontier · D-015" accent={AMBER} />
                <div style={{ padding: 16 }}>
                  <ResponsiveContainer width="100%" height={220}>
                    <ScatterChart>
                      <CartesianGrid strokeDasharray="2 4" stroke={BORDER} />
                      <XAxis dataKey="cost" name="$/query" type="number" tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} label={{ value: "$/query", fill: DIM, fontSize: 9, dy: 14 }} />
                      <YAxis dataKey="recall" name="Recall@5" type="number" domain={[0.5, 1]} tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} />
                      <Tooltip cursor={{ stroke: BORDER }} content={<TT />} />
                      <Scatter name="Architectures" data={data.map(d => ({ cost: d.cost_query, recall: d.recall, name: d.arch }))}
                        fill={GREEN} shape={(props) => {
                          const { cx, cy, payload } = props;
                          const idx = data.findIndex(d => d.arch === payload.name);
                          return (
                            <g>
                              <circle cx={cx} cy={cy} r={6} fill={Object.values(ARCH_COLORS)[idx]} stroke={BG2} strokeWidth={2} />
                              <text x={cx + 9} y={cy + 4} fill={TEXT2} fontSize={9} fontFamily={FONT}>{payload.name}</text>
                            </g>
                          );
                        }}
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </Panel>
            </div>

            <Panel>
              <PanelHeader label="Cost Breakdown per Query Component" sub="embed + LLM + sub-query gen (Multi-Query only)" accent={AMBER} />
              <div style={{ padding: 16 }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                  <thead>
                    <tr style={{ background: BG3 }}>
                      {["Architecture","Embed $/q","LLM Input $/q","LLM Output $/q","Sub-query $/q","TOTAL $/q","$1K queries","$100K queries","Rec/$"].map(h => (
                        <th key={h} style={{ padding: "8px 12px", textAlign: "left", color: DIM, fontSize: 9, letterSpacing: 1, borderBottom: `1px solid ${BORDER}` }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.map((d, i) => {
                      const subq = d.arch === "Multi-Query" ? 0.00049 : 0;
                      const embed = 0.000001;
                      const llmIn = 0.000098;
                      const llmOut = 0.000321;
                      const total = d.cost_query;
                      return (
                        <tr key={i} style={{ borderBottom: `1px solid ${BORDER}` }}>
                          <td style={{ padding: "8px 12px", color: Object.values(ARCH_COLORS)[i], fontWeight: 700 }}>{d.arch}</td>
                          <td style={{ padding: "8px 12px", color: TEXT2 }}>${embed.toFixed(6)}</td>
                          <td style={{ padding: "8px 12px", color: TEXT2 }}>${llmIn.toFixed(6)}</td>
                          <td style={{ padding: "8px 12px", color: TEXT2 }}>${llmOut.toFixed(6)}</td>
                          <td style={{ padding: "8px 12px", color: subq > 0 ? AMBER : DIM }}>{subq > 0 ? `$${subq.toFixed(6)}` : "—"}</td>
                          <td style={{ padding: "8px 12px", color: AMBER, fontWeight: 700 }}>${total.toFixed(5)}</td>
                          <td style={{ padding: "8px 12px" }}>${(total * 1000).toFixed(3)}</td>
                          <td style={{ padding: "8px 12px" }}>${(total * 100000).toFixed(2)}</td>
                          <td style={{ padding: "8px 12px", color: GREEN }}>{(d.recall / total).toFixed(0)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Panel>
          </div>
        )}

        {/* ── EMBEDDINGS ── */}
        {tab === "embeddings" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ display: "flex", gap: 16 }}>
              <Panel style={{ flex: 1 }}>
                <PanelHeader label="Recall@5 by Embedding Model" sub="same VectorRAG architecture · D-013" />
                <div style={{ padding: 16 }}>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={EMBEDDING_DATA.map(e => ({ name: e.model, "Recall@5": e.recall, free: e.free }))}>
                      <CartesianGrid strokeDasharray="2 4" stroke={BORDER} />
                      <XAxis dataKey="name" tick={{ fill: DIM, fontSize: 9, fontFamily: FONT }} />
                      <YAxis domain={[0.5, 1]} tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} />
                      <Tooltip content={<TT />} />
                      <Bar dataKey="Recall@5" radius={[2,2,0,0]}
                        label={{ position: "top", fill: DIM, fontSize: 8, fontFamily: FONT }}
                        fill={GREEN} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Panel>
              <Panel style={{ flex: 1 }}>
                <PanelHeader label="Query Throughput (docs/sec)" sub="GPU available = local models win big · D-014" accent={CYAN} />
                <div style={{ padding: 16 }}>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={EMBEDDING_DATA.map(e => ({ name: e.model, throughput: e.throughput, free: e.free }))}>
                      <CartesianGrid strokeDasharray="2 4" stroke={BORDER} />
                      <XAxis dataKey="name" tick={{ fill: DIM, fontSize: 9, fontFamily: FONT }} />
                      <YAxis tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} />
                      <Tooltip content={<TT />} />
                      <Bar dataKey="throughput" fill={CYAN} radius={[2,2,0,0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Panel>
            </div>

            <Panel>
              <PanelHeader label="Embedding Model Comparison Matrix" sub="D-013: three cost tiers" accent={CYAN} />
              <div style={{ padding: 16 }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                  <thead>
                    <tr style={{ background: BG3 }}>
                      {["Model","Provider","Dims","$/1M tokens","Recall@5","P50 latency","Throughput","Index MB","Rec/$","Tier"].map(h => (
                        <th key={h} style={{ padding: "8px 12px", textAlign: "left", color: DIM, fontSize: 9, letterSpacing: 1, borderBottom: `1px solid ${BORDER}` }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {EMBEDDING_DATA.map((e, i) => (
                      <tr key={i} style={{ borderBottom: `1px solid ${BORDER}` }}>
                        <td style={{ padding: "8px 12px", color: e.free ? GREEN : AMBER, fontWeight: 700 }}>{e.model}</td>
                        <td style={{ padding: "8px 12px" }}><Chip label={e.provider.toUpperCase()} color={e.free ? GREEN : AMBER} /></td>
                        <td style={{ padding: "8px 12px", color: TEXT2 }}>{e.dims}</td>
                        <td style={{ padding: "8px 12px", color: e.free ? GREEN : TEXT }}>{e.free ? "FREE" : `$${e.cost_1m.toFixed(3)}`}</td>
                        <td style={{ padding: "8px 12px" }}><Meter val={e.recall} color={GREEN} /></td>
                        <td style={{ padding: "8px 12px" }}>{e.latency_ms}ms</td>
                        <td style={{ padding: "8px 12px" }}>{e.throughput}/s</td>
                        <td style={{ padding: "8px 12px", color: TEXT2 }}>{(e.dims * 50000 * 4 / 1024 / 1024).toFixed(0)}</td>
                        <td style={{ padding: "8px 12px", color: GREEN }}>{e.free ? `${(e.recall/0.001).toFixed(0)}*` : (e.recall/e.cost_1m).toFixed(1)}</td>
                        <td style={{ padding: "8px 12px" }}><Chip label={e.free ? "LOCAL" : e.model.includes("large") ? "PREMIUM" : "STANDARD"} color={e.free ? GREEN : e.model.includes("large") ? AMBER : CYAN} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>

            {/* GPU speedup panel */}
            <Panel>
              <PanelHeader label="GPU vs CPU Speedup — Local Models" sub="D-014: CUDA accelerated inference" accent={GREEN} />
              <div style={{ padding: 16, display: "flex", gap: 16 }}>
                {GPU_DATA.map((g, i) => (
                  <div key={i} style={{ flex: 1, background: BG3, border: `1px solid ${BORDER}`, padding: 16 }}>
                    <div style={{ color: GREEN, fontSize: 11, fontWeight: 700, marginBottom: 12 }}>{g.model}</div>
                    {[
                      ["Embed throughput", g.cpu_embed, g.gpu_embed, "docs/s", g.embed_speedup],
                      ["Query P50 latency", g.cpu_q_p50, g.gpu_q_p50, "ms", g.query_speedup],
                      ["FAISS search", "baseline", "—", "", g.faiss_speedup],
                    ].map(([label, cpu, gpu, unit, speedup]) => (
                      <div key={label} style={{ marginBottom: 12 }}>
                        <div style={{ color: DIM, fontSize: 9, letterSpacing: 1, marginBottom: 6 }}>{label}</div>
                        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                          <div style={{ flex: 1 }}>
                            <div style={{ color: TEXT2, fontSize: 9, marginBottom: 2 }}>CPU</div>
                            <div style={{ color: TEXT, fontSize: 13, fontWeight: 700 }}>{cpu} <span style={{ color: DIM, fontSize: 9 }}>{unit}</span></div>
                          </div>
                          <div style={{ color: AMBER, fontSize: 14 }}>→</div>
                          <div style={{ flex: 1 }}>
                            <div style={{ color: TEXT2, fontSize: 9, marginBottom: 2 }}>GPU</div>
                            <div style={{ color: GREEN, fontSize: 13, fontWeight: 700 }}>{gpu} <span style={{ color: DIM, fontSize: 9 }}>{unit}</span></div>
                          </div>
                          <div style={{ background: `${GREEN}18`, border: `1px solid ${GREEN}40`, padding: "4px 10px", borderRadius: 2 }}>
                            <span style={{ color: GREEN, fontSize: 13, fontWeight: 700 }}>{speedup}×</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ))}
                <div style={{ flex: 1, background: BG3, border: `1px solid ${AMBER}22`, borderLeft: `3px solid ${AMBER}`, padding: 16 }}>
                  <div style={{ color: AMBER, fontSize: 10, fontWeight: 700, marginBottom: 10 }}>D-014 FINDING</div>
                  <p style={{ color: TEXT2, fontSize: 11, lineHeight: 1.7, margin: 0 }}>
                    GPU wins massively on <span style={{ color: GREEN }}>batch indexing</span> (11×). For single-query latency, GPU advantage depends on index size — below ~50K vectors, CUDA transfer overhead can negate the speedup. Above 50K vectors GPU consistently faster.<br/><br/>
                    <span style={{ color: DIM, fontSize: 9 }}>OpenAI API models have no GPU path. GPU benchmark scoped to local HF models only (LIM-008).</span>
                  </p>
                </div>
              </div>
            </Panel>
          </div>
        )}

        {/* ── ABLATIONS ── */}
        {tab === "ablations" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ display: "flex", gap: 8, marginBottom: 4 }}>
              <Chip label="ABL-001: HYBRID ALPHA ⏳" color={DIM} />
              <Chip label="ABL-002: CHUNK SIZE ✅" color={GREEN} />
              <Chip label="ABL-003: TOP-K ✅" color={GREEN} />
              <Chip label="ABL-004: EMBEDDING MODEL ✅" color={GREEN} />
              <Chip label="ABL-006: SUB-QUERY COUNT ✅" color={GREEN} />
            </div>

            <div style={{ display: "flex", gap: 16 }}>
              <Panel style={{ flex: 1 }}>
                <PanelHeader label="ABL-002: Chunk Size vs Recall" sub="Vector RAG · Wikipedia dataset" />
                <div style={{ padding: 16 }}>
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={ABLATION_DATA.chunk_size}>
                      <CartesianGrid strokeDasharray="2 4" stroke={BORDER} />
                      <XAxis dataKey="x" label={{ value: "tokens", fill: DIM, fontSize: 9, dy: 14 }} tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} />
                      <YAxis tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} />
                      <Tooltip content={<TT />} />
                      <Legend wrapperStyle={{ color: DIM, fontSize: 10, fontFamily: FONT }} />
                      <Line type="monotone" dataKey="recall"    stroke={GREEN} strokeWidth={2} dot={{ r: 4, fill: GREEN }} name="Recall@5" />
                      <Line type="monotone" dataKey="precision" stroke={CYAN}  strokeWidth={2} dot={{ r: 4, fill: CYAN }} name="Precision@5" />
                    </LineChart>
                  </ResponsiveContainer>
                  <div style={{ color: AMBER, fontSize: 10, marginTop: 8, padding: "6px 10px", background: `${AMBER}10`, border: `1px solid ${AMBER}30` }}>
                    ▶ 256 tokens → best recall (0.76). 512 tokens → baseline (chosen for latency/storage balance, D-004). 1024 tokens → worst recall — context too diluted.
                  </div>
                </div>
              </Panel>

              <Panel style={{ flex: 1 }}>
                <PanelHeader label="ABL-003: Top-K vs Recall + Latency" sub="Vector RAG · k = {1,3,5,10}" />
                <div style={{ padding: 16 }}>
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={ABLATION_DATA.top_k}>
                      <CartesianGrid strokeDasharray="2 4" stroke={BORDER} />
                      <XAxis dataKey="x" label={{ value: "k", fill: DIM, fontSize: 9, dy: 14 }} tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} />
                      <YAxis tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} />
                      <Tooltip content={<TT />} />
                      <Legend wrapperStyle={{ color: DIM, fontSize: 10, fontFamily: FONT }} />
                      <Line type="monotone" dataKey="recall"  stroke={GREEN} strokeWidth={2} dot={{ r: 4, fill: GREEN }} name="Recall@k" />
                      <Line type="monotone" dataKey="latency" stroke={AMBER} strokeWidth={2} dot={{ r: 4, fill: AMBER }} name="Latency (s)" />
                    </LineChart>
                  </ResponsiveContainer>
                  <div style={{ color: GREEN, fontSize: 10, marginTop: 8, padding: "6px 10px", background: `${GREEN}10`, border: `1px solid ${GREEN}30` }}>
                    ▶ k=5 chosen (D-006): diminishing recall returns after k=5 while latency keeps growing.
                  </div>
                </div>
              </Panel>
            </div>

            <Panel>
              <PanelHeader label="ABL-006: Sub-Query Count vs Recall + Cost" sub="Multi-Query RAG · n = {1,2,3,5}" />
              <div style={{ padding: 16 }}>
                <ResponsiveContainer width="100%" height={180}>
                  <LineChart data={ABLATION_DATA.subqueries}>
                    <CartesianGrid strokeDasharray="2 4" stroke={BORDER} />
                    <XAxis dataKey="x" label={{ value: "sub-queries", fill: DIM, fontSize: 9, dy: 14 }} tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} />
                    <YAxis yAxisId="left" tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} />
                    <YAxis yAxisId="right" orientation="right" tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} />
                    <Tooltip content={<TT />} />
                    <Legend wrapperStyle={{ color: DIM, fontSize: 10, fontFamily: FONT }} />
                    <Line yAxisId="left"  type="monotone" dataKey="recall" stroke={GREEN} strokeWidth={2} dot={{ r: 4, fill: GREEN }} name="Recall@5" />
                    <Line yAxisId="right" type="monotone" dataKey="cost"   stroke={AMBER} strokeWidth={2} dot={{ r: 4, fill: AMBER }} name="$/query" />
                  </LineChart>
                </ResponsiveContainer>
                <div style={{ color: CYAN, fontSize: 10, marginTop: 8, padding: "6px 10px", background: `${CYAN}10`, border: `1px solid ${CYAN}30` }}>
                  ▶ 3 sub-queries chosen (D-008): +11pp recall over n=1, but cost grows faster than recall beyond n=3. Sweet spot before diminishing returns.
                </div>
              </div>
            </Panel>
          </div>
        )}

        {/* ── DATASETS ── */}
        {tab === "datasets" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ display: "flex", gap: 12 }}>
              {Object.entries(DATASETS).map(([key, ds]) => (
                <Panel key={key} style={{ flex: 1 }} glow={key === dataset}>
                  <PanelHeader label={`DS-00${Object.keys(DATASETS).indexOf(key)+1}`} sub={key.toUpperCase()} accent={key === dataset ? GREEN : DIM} />
                  <div style={{ padding: 16 }}>
                    <div style={{ color: TEXT, fontSize: 13, fontWeight: 700, marginBottom: 10 }}>{ds.label}</div>
                    {[
                      ["Domain", ds.domain],
                      ["Documents", ds.docs.toLocaleString()],
                      ["Source", ds.source],
                      ["Chunks (~)", `${Math.round(ds.docs * 1.8).toLocaleString()}`],
                      ["QA pairs", "50 (RAGAS synthetic, 10% human review)"],
                    ].map(([k, v]) => (
                      <div key={k} style={{ display: "flex", gap: 10, padding: "5px 0", borderBottom: `1px solid ${BORDER}` }}>
                        <span style={{ color: DIM, fontSize: 10, width: 90 }}>{k}</span>
                        <span style={{ color: TEXT2, fontSize: 10 }}>{v}</span>
                      </div>
                    ))}
                    {key === "large" && (
                      <div style={{ marginTop: 10, padding: "6px 10px", background: `${AMBER}10`, border: `1px solid ${AMBER}30`, color: AMBER, fontSize: 9 }}>
                        ⚠ Pinned to kubernetes/website@v1.29 for reproducibility (D-010)
                      </div>
                    )}
                    {key === "medium" && (
                      <div style={{ marginTop: 10, padding: "6px 10px", background: `${DIM}18`, border: `1px solid ${BORDER}`, color: DIM, fontSize: 9 }}>
                        ~12% abstracts-only (no full PDF). Flagged in metadata (LIM-003).
                      </div>
                    )}
                  </div>
                </Panel>
              ))}
            </div>

            <Panel>
              <PanelHeader label="Recall@5 Degradation Across Dataset Scale" sub="architecture quality drops as corpus grows" />
              <div style={{ padding: 16 }}>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart>
                    <CartesianGrid strokeDasharray="2 4" stroke={BORDER} />
                    <XAxis dataKey="x" type="category" allowDuplicatedCategory={false} tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} />
                    <YAxis domain={[0.5, 1]} tick={{ fill: DIM, fontSize: 10, fontFamily: FONT }} />
                    <Tooltip content={<TT />} />
                    <Legend wrapperStyle={{ color: DIM, fontSize: 10, fontFamily: FONT }} />
                    {ARCHS.map((arch, i) => (
                      <Line key={arch} type="monotone" name={arch}
                        data={["small","medium","large"].map(ds => ({
                          x: ds,
                          recall: BENCHMARK_DATA[ds].systems[i].recall,
                        }))}
                        dataKey="recall" stroke={Object.values(ARCH_COLORS)[i]} strokeWidth={2}
                        dot={{ r: 4, fill: Object.values(ARCH_COLORS)[i] }}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
                <div style={{ color: AMBER, fontSize: 10, marginTop: 8, padding: "6px 10px", background: `${AMBER}10`, border: `1px solid ${AMBER}30` }}>
                  ▶ Hybrid RAG degrades most gracefully (−7pp small→large). Graph RAG best at scale but resource cost grows steeply. Vector RAG degrades fastest (−11pp).
                </div>
              </div>
            </Panel>
          </div>
        )}

        {/* ── LIMITATIONS ── */}
        {tab === "limitations" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div style={{ padding: "10px 16px", background: `${AMBER}10`, border: `1px solid ${AMBER}30`, borderLeft: `3px solid ${AMBER}`, marginBottom: 4 }}>
              <span style={{ color: AMBER, fontSize: 10, fontWeight: 700, letterSpacing: 1 }}>EXPERIMENTAL LIMITATIONS</span>
              <span style={{ color: TEXT2, fontSize: 10, marginLeft: 12 }}>
                Sharp limitations signal engineering rigor — vague ones signal defensive thinking.
              </span>
            </div>

            {LIMITATIONS.map((lim, i) => (
              <div key={i} style={{
                background: BG3, border: `1px solid ${BORDER}`,
                borderLeft: `3px solid ${SEV_COLOR[lim.severity]}`,
                padding: "12px 16px", display: "flex", gap: 16, alignItems: "flex-start"
              }}>
                <div style={{ minWidth: 80 }}>
                  <Chip label={lim.id} color={SEV_COLOR[lim.severity]} />
                  <div style={{ marginTop: 6 }}><Chip label={lim.severity} color={SEV_COLOR[lim.severity]} /></div>
                </div>
                <div style={{ flex: 1 }}>
                  <span style={{ color: DIM, fontSize: 9, letterSpacing: 1 }}>[{lim.scope}] </span>
                  <span style={{ color: TEXT, fontSize: 11, lineHeight: 1.7 }}>{lim.text}</span>
                </div>
              </div>
            ))}

            <div style={{ marginTop: 8, padding: "14px 16px", background: BG3, border: `1px solid ${BORDER}` }}>
              <div style={{ color: DIM, fontSize: 9, letterSpacing: 1, marginBottom: 8 }}>REPRODUCIBILITY CHECKLIST</div>
              {[
                "LLM frozen: gemini-2.0-flash, temperature=0 (D-001)",
                "Judge model: gemini-1.5-pro — separate from generator to prevent self-eval bias (D-002)",
                "All datasets fetched once, serialized to disk with timestamp (D-010)",
                "RANDOM_SEED=42 for all dataset sampling and QA generation (D-010)",
                "Kubernetes docs pinned to tag v1.29 (DS-003)",
                "Benchmark protocol: 3 warmup + 10 timed runs, averaged (D-011)",
                "All results auto-appended to decisions_log.md via log_results.py",
                "Git hash recorded in every RUN block",
              ].map((item, i) => (
                <div key={i} style={{ display: "flex", gap: 10, padding: "5px 0", borderBottom: `1px solid ${BORDER}`, alignItems: "center" }}>
                  <span style={{ color: GREEN, fontSize: 11 }}>✓</span>
                  <span style={{ color: TEXT2, fontSize: 11 }}>{item}</span>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>

      {/* Status bar */}
      <div style={{ position: "fixed", bottom: 0, left: 0, right: 0, background: BG2, borderTop: `1px solid ${BORDER}`,
        padding: "4px 24px", display: "flex", gap: 24, alignItems: "center", zIndex: 20 }}>
        <span style={{ color: GREEN, fontSize: 9, letterSpacing: 1 }}>● SYS OK</span>
        <span style={{ color: DIM, fontSize: 9 }}>LLM: {LLM_MODEL} · TEMP=0 · FROZEN</span>
        <span style={{ color: DIM, fontSize: 9 }}>DATASET: {DATASETS[dataset].label}</span>
        <span style={{ color: DIM, fontSize: 9 }}>5 ARCHITECTURES · 3 DATASETS · RAGAS EVAL</span>
        <span style={{ marginLeft: "auto", color: DIM, fontSize: 9 }}>decisions_log.md · v1.1.0</span>
      </div>
    </div>
  );
}

// ── Inline meter bar ──────────────────────────────────────────────────────
function Meter({ val, color = "#00ff9d" }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ width: 60, height: 4, background: "#1e2d3d", borderRadius: 2 }}>
        <div style={{ height: 4, borderRadius: 2, background: color, width: `${val * 100}%` }} />
      </div>
      <span style={{ color, fontSize: 10, fontFamily: "'IBM Plex Mono', monospace" }}>{val.toFixed(3)}</span>
    </div>
  );
}
