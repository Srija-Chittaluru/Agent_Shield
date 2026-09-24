"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  ShieldCheck,
  Upload,
  FileCode2,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  Workflow,
  ArrowRight,
  History,
  Target,
  Wand2,
  RotateCcw,
  Info,
  Mic,
  MessageSquare,
  Save,
  PlayCircle,
  Trash2,
} from "lucide-react";
import {
  getInventory,
  getAgent,
  uploadAgentFlow,
  getAgentFlows,
  getFlow,
  generateNodeScript,
  saveNodeTest,
  runNodeTest,
  getRun,
  getReport,
  InventoryAgent,
  FlowSummary,
  FlowDetail,
  NodeScriptTurn,
  SavedNodeTest,
  Report,
} from "../lib/api";
import { TranscriptDetails } from "../dashboard/page";

// Phase 1: upload -> parse -> store -> display nodes.
// Phase 2: select a node -> generate a test goal + a deterministic caller script ->
// review/edit it.
// Phase 3 (this file also covers it): save the reviewed script as a real test case,
// then run it through the EXISTING Temporal execution pipeline — the same
// AgentTestWorkflow/RunGroupWorkflow, scripted runner, voice protocols, and Judge
// every other test uses (see backend/app/routers/flows.py and
// app/core/node_script.py). This introduces no second voice-testing engine. Results
// are shown here by reusing the dashboard's own transcript view (TranscriptDetails,
// imported from ../dashboard/page) rather than duplicating it.

interface UploadDiagnostics {
  top_level_keys?: string[];
  top_level_summary?: Record<string, string>;
  possible_sections?: string[];
}

function extractErrorBody(message: string): { errors: string[]; message?: string; diagnostics?: UploadDiagnostics } {
  const idx = message.indexOf(": ");
  const rest = idx >= 0 ? message.slice(idx + 2) : message;
  try {
    const parsed = JSON.parse(rest);
    if (parsed && Array.isArray(parsed.errors)) {
      return { errors: parsed.errors, message: parsed.message, diagnostics: parsed.diagnostics };
    }
  } catch {
    /* not a JSON error payload — fall through to the raw message */
  }
  return { errors: [message] };
}

// Compatibility wrapper for the call sites that only ever showed a flat error list
// (script generation, save, run) — unchanged behavior for those.
function extractErrors(message: string): string[] {
  return extractErrorBody(message).errors;
}

export default function FlowsPage() {
  const [agents, setAgents] = useState<InventoryAgent[]>([]);
  const [selectedAgentId, setSelectedAgentId] = useState<number | null>(null);
  const [flows, setFlows] = useState<FlowSummary[]>([]);
  const [flow, setFlow] = useState<FlowDetail | null>(null);

  const [flowsLoading, setFlowsLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadErrors, setUploadErrors] = useState<string[] | null>(null);
  const [uploadDiagnostics, setUploadDiagnostics] = useState<UploadDiagnostics | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  // The upload control is hidden by default once a flow already exists for the
  // selected agent (Part 2/9) — "Upload a Different / Updated Flow" reveals it.
  const [showUpload, setShowUpload] = useState(false);

  // Phase 2 — selected node + its generated/edited test goal & script (draft only,
  // nothing here is persisted).
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [testGoal, setTestGoal] = useState("");
  const [script, setScript] = useState<NodeScriptTurn[] | null>(null);
  const [generating, setGenerating] = useState(false);
  const [genErrors, setGenErrors] = useState<string[] | null>(null);
  // `dirty` tracks whether the current draft differs from whatever was last saved,
  // so Save/Run know whether a (re)save is needed and, when it is, whether to insert
  // a new test case or update the one already saved for this node.
  const [dirty, setDirty] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  // Brings the Node Test section into view the moment a node is selected, instead of
  // requiring a manual scroll past the node list — never fires on initial page load
  // since selectedNodeId starts null and only changes via an explicit node click.
  const nodeTestRef = useRef<HTMLDivElement | null>(null);

  // Phase 3 — save the reviewed script, then run it through the existing pipeline.
  const [savedTest, setSavedTest] = useState<SavedNodeTest | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveErrors, setSaveErrors] = useState<string[] | null>(null);
  const [runId, setRunId] = useState<number | null>(null);
  const [runStatus, setRunStatus] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [runErrors, setRunErrors] = useState<string[] | null>(null);
  const [report, setReport] = useState<Report | null>(null);

  useEffect(() => {
    if (selectedNodeId) nodeTestRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [selectedNodeId]);

  useEffect(() => {
    getInventory().then(async (inv) => {
      const voiceAgents = inv.customers
        .flatMap((c) => c.agents)
        .filter((a): a is InventoryAgent & { agent_id: number } => a.modality === "voice" && a.agent_id != null);
      const seen = new Set<number>();
      const deduped = voiceAgents.filter((a) => (seen.has(a.agent_id) ? false : (seen.add(a.agent_id), true)));

      // Arriving from the dashboard's "Flow-Based Scripts" option carries the agent
      // that was just connected. A brand-new agent has no customer_agents row yet, so
      // it isn't in inventory yet — fetch it directly so it still appears and gets
      // preselected instead of falling back to some other agent.
      let list = deduped;
      const requested = Number(new URLSearchParams(window.location.search).get("agentId"));
      if (requested && !deduped.some((a) => a.agent_id === requested)) {
        try {
          const agent = await getAgent(requested);
          if (agent.modality === "voice") {
            list = [
              { key: `agent-${agent.id}`, name: agent.name, customer_agent_id: null, agent_id: agent.id, modality: "voice" },
              ...deduped,
            ];
          }
        } catch {
          /* agent not found — fall back to the inventory list below */
        }
      }

      setAgents(list);
      const match = requested && list.find((a) => a.agent_id === requested);
      if (match) setSelectedAgentId(match.agent_id);
      else if (list.length > 0) setSelectedAgentId(list[0].agent_id);
    });
  }, []);

  useEffect(() => {
    if (selectedAgentId == null) {
      setFlows([]);
      setFlow(null);
      setShowUpload(false);
      return;
    }
    setFlow(null);
    setShowUpload(false);
    setFlowsLoading(true);
    getAgentFlows(selectedAgentId).then((r) => {
      setFlows(r.flows);
      // Already uploaded a flow for this agent before — load it right away instead of
      // asking to upload again (newest first, since list_agent_flows orders id DESC).
      // showUpload stays false, so the upload control stays hidden behind "Upload a
      // Different / Updated Flow" until the user explicitly asks for it.
      if (r.flows.length > 0) {
        getFlow(r.flows[0].id)
          .then(setFlow)
          .finally(() => setFlowsLoading(false));
      } else {
        setFlowsLoading(false);
      }
    });
  }, [selectedAgentId]);

  const selectedAgentLabel = useMemo(
    () => agents.find((a) => a.agent_id === selectedAgentId)?.name ?? "",
    [agents, selectedAgentId]
  );

  async function onFileSelected(file: File) {
    if (selectedAgentId == null) return;
    setUploading(true);
    setUploadErrors(null);
    setUploadDiagnostics(null);
    setUploadSuccess(null);
    try {
      const content = await file.text();
      const result = await uploadAgentFlow(selectedAgentId, content, file.name);
      setFlow({
        id: result.flow_id,
        agent_id: result.agent_id,
        name: result.name,
        source_format: result.source_format,
        extraction_method: result.extraction_method,
        created_at: new Date().toISOString(),
        nodes: result.nodes,
        edges: result.edges,
      });
      setUploadSuccess(`Parsed "${result.agent_name}" — ${result.nodes.length} node(s), ${result.edges.length} edge(s).`);
      setShowUpload(false); // fold the upload control back away now that we have a flow
      const refreshed = await getAgentFlows(selectedAgentId);
      setFlows(refreshed.flows);
    } catch (err) {
      const body = extractErrorBody(err instanceof Error ? err.message : String(err));
      setUploadErrors(body.message ? [body.message, ...body.errors] : body.errors);
      setUploadDiagnostics(body.diagnostics ?? null);
    } finally {
      setUploading(false);
    }
  }

  async function onSelectStoredFlow(flowId: number) {
    setUploadErrors(null);
    setUploadDiagnostics(null);
    setUploadSuccess(null);
    const detail = await getFlow(flowId);
    setFlow(detail);
    setShowUpload(false);
  }

  function nodeName(id: string): string {
    return flow?.nodes.find((n) => n.id === id)?.name ?? id;
  }

  const selectedNode = useMemo(
    () => flow?.nodes.find((n) => n.id === selectedNodeId) ?? null,
    [flow, selectedNodeId]
  );

  // Explicit script state (see the pasted UX spec's "SCRIPT STATE" section): drives
  // which helper text/badge shows without adding a separate state-management layer —
  // it's derived from the existing script/savedTest/dirty pieces above.
  type ScriptStatus = "none" | "draft" | "saved" | "modified";
  const scriptStatus: ScriptStatus = !script ? "none" : !savedTest ? "draft" : dirty ? "modified" : "saved";

  function resetSaveAndRunState() {
    setSavedTest(null);
    setSaveErrors(null);
    setRunId(null);
    setRunStatus(null);
    setRunning(false);
    setRunErrors(null);
    setReport(null);
    setDirty(false);
    setShowDeleteConfirm(false);
  }

  function selectNode(nodeId: string) {
    if (nodeId === selectedNodeId) return;
    setSelectedNodeId(nodeId);
    setTestGoal("");
    setScript(null);
    setGenErrors(null);
    resetSaveAndRunState();
  }

  async function onGenerateScript() {
    if (!flow || !selectedNodeId) return;
    setGenerating(true);
    setGenErrors(null);
    resetSaveAndRunState(); // a (re)generated script is unreviewed and unsaved again
    try {
      const result = await generateNodeScript(flow.id, selectedNodeId, testGoal.trim() || undefined);
      setTestGoal(result.test_goal);
      setScript(result.script);
    } catch (err) {
      setGenErrors(extractErrors(err instanceof Error ? err.message : String(err)));
    } finally {
      setGenerating(false);
    }
  }

  function updateTurn(index: number, field: keyof NodeScriptTurn, value: string) {
    setScript((prev) => {
      if (!prev) return prev;
      const next = [...prev];
      next[index] = { ...next[index], [field]: value };
      return next;
    });
    setDirty(true); // no-op while there's nothing saved yet (scriptStatus ignores it until then)
  }

  function onTestGoalChange(value: string) {
    setTestGoal(value);
    setDirty(true);
  }

  function onDeleteDraft() {
    setScript(null);
    setGenErrors(null);
    resetSaveAndRunState(); // clears the local draft/save link only — the persisted
    // test_cases row (if any) is left untouched, per the "never silently delete a
    // saved test case" requirement.
  }

  // Shared by Save and the Run auto-save path — persists the current draft, updating
  // the row from a prior save of THIS node instead of inserting a duplicate.
  async function persistScript(): Promise<SavedNodeTest> {
    if (!flow || !selectedNodeId || !script || !testGoal.trim()) {
      throw new Error("Generate a script and enter a test goal before saving.");
    }
    const result = await saveNodeTest(flow.id, selectedNodeId, testGoal.trim(), script, savedTest?.test_id);
    setSavedTest(result);
    setDirty(false);
    return result;
  }

  async function onSaveTest() {
    if (!script || !testGoal.trim()) return;
    setSaving(true);
    setSaveErrors(null);
    setRunId(null);
    setRunStatus(null);
    setReport(null);
    setRunErrors(null);
    try {
      await persistScript();
    } catch (err) {
      setSaveErrors(extractErrors(err instanceof Error ? err.message : String(err)));
    } finally {
      setSaving(false);
    }
  }

  // RUN: save first only if there's nothing saved yet, or the draft has changed since
  // the last save — otherwise reuse the existing saved test id as-is. Either way, the
  // existing save/run APIs are the only path; there's no second execution route.
  async function onRunTest() {
    if (!flow || !selectedNodeId || !script || !testGoal.trim()) return;
    setRunning(true);
    setRunErrors(null);
    setReport(null);
    try {
      let testId = savedTest?.test_id ?? null;
      if (!savedTest || dirty) {
        setSaving(true);
        const saved = await persistScript();
        testId = saved.test_id;
        setSaving(false);
      }
      const result = await runNodeTest(flow.id, selectedNodeId, testId as number);
      setRunStatus("queued");
      setRunId(result.run_id);
    } catch (err) {
      setRunErrors(extractErrors(err instanceof Error ? err.message : String(err)));
      setRunning(false);
      setSaving(false);
    }
  }

  // Poll the EXISTING run/report endpoints (same ones the dashboard already polls)
  // until the run finishes, then load its full report.
  useEffect(() => {
    if (runId == null) return;
    let cancelled = false;

    const poll = async () => {
      try {
        const status = await getRun(runId);
        if (cancelled) return;
        setRunStatus(status.status);
        if (status.status === "done" || status.status === "error") {
          const rep = await getReport(runId);
          if (!cancelled) {
            setReport(rep);
            setRunning(false);
          }
          return true; // stop polling
        }
      } catch (err) {
        if (!cancelled) {
          setRunErrors(extractErrors(err instanceof Error ? err.message : String(err)));
          setRunning(false);
        }
        return true;
      }
      return false;
    };

    const interval = setInterval(async () => {
      if (await poll()) clearInterval(interval);
    }, 2000);
    poll(); // check immediately rather than waiting a full interval

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [runId]);

  return (
    <main className="min-h-screen bg-[#0B0B0F] pb-24 text-[#F8FAFC]">
      <header className="sticky top-0 z-50 border-b border-white/12 bg-[#0B0B0F]/70 backdrop-blur-md">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
          <Link href="/" className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/20 bg-white/4 text-slate-200">
              <ShieldCheck className="h-5 w-5" strokeWidth={1.5} />
            </div>
            <span className="font-logo text-lg font-extrabold tracking-tight">AgentShield</span>
          </Link>
          <div className="flex items-center gap-2 text-sm text-[#9CA3AF]">
            <Workflow className="h-4 w-4" strokeWidth={1.5} />
            Flow-Aware Voice Testing
          </div>
        </div>
      </header>

      <section className="mx-auto max-w-5xl px-6 pt-10">
        <h1 className="font-heading text-2xl font-medium tracking-tight sm:text-3xl">
          Flow-Based Script Testing
        </h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-[#9CA3AF]">
          Upload the voice agent&apos;s JSON/YAML flow definition to analyze its nodes and
          generate node-specific test scripts. AgentShield adapts whatever structure your
          file uses — it does not need to match any particular schema.
        </p>

        {/* Step 1: agent selection */}
        <div className="mt-8 rounded-xl border border-white/12 bg-white/2 p-6">
          <label className="block text-sm font-medium text-[#F8FAFC]">1. Select a voice agent</label>
          {agents.length === 0 ? (
            <p className="mt-2 text-sm text-[#9CA3AF]">
              No voice agents found yet. Connect one first from the{" "}
              <Link href="/start?modality=voice" className="underline hover:text-white">
                Voice Agent
              </Link>{" "}
              flow.
            </p>
          ) : (
            <select
              className="mt-2 w-full rounded-lg border border-white/15 bg-black/40 px-3 py-2 text-sm text-[#F8FAFC] outline-none focus:border-white/40"
              value={selectedAgentId ?? ""}
              onChange={(e) => setSelectedAgentId(Number(e.target.value))}
            >
              {agents.map((a) => (
                <option key={a.agent_id} value={a.agent_id ?? ""}>
                  {a.name}
                </option>
              ))}
            </select>
          )}
        </div>

        {/* Step 2: Voice Agent Flow Definition — an already-uploaded flow is the
            default view; uploading is an explicit, optional action, never a forced
            re-ask (Part 2/9). */}
        <div className="mt-6 rounded-xl border border-white/12 bg-white/2 p-6">
          <label className="block text-sm font-medium text-[#F8FAFC]">
            2. Voice Agent Flow Definition {selectedAgentLabel && `for ${selectedAgentLabel}`}
          </label>
          <p className="mt-1 text-xs text-[#9CA3AF]">
            The voice agent&apos;s own JSON/YAML flow definition — analyzed for nodes so
            you can generate node-specific test scripts. This is separate from Agent
            Knowledge (uploaded when connecting the agent).
          </p>

          {flowsLoading ? (
            <div className="mt-3 flex items-center gap-2 text-sm text-[#9CA3AF]">
              <Loader2 className="h-4 w-4 animate-spin" strokeWidth={1.5} />
              Checking for an existing flow…
            </div>
          ) : flow && !showUpload ? (
            <div className="mt-3 rounded-lg border border-emerald-400/25 bg-emerald-400/5 p-4">
              <div className="flex items-center gap-2 text-sm font-medium text-emerald-300">
                <CheckCircle2 className="h-4 w-4 shrink-0" strokeWidth={1.5} />
                Flow detected
              </div>
              <p className="mt-1 text-sm text-[#F8FAFC]">{flow.name}</p>
              <p className="mt-0.5 text-xs text-[#9CA3AF]">
                Uploaded {new Date(flow.created_at).toLocaleString()} · Parsed using{" "}
                {flow.extraction_method === "llm" ? "LLM-assisted extraction" : "deterministic extraction"}
              </p>
              <button
                onClick={() => setShowUpload(true)}
                className="mt-3 inline-flex items-center gap-1.5 rounded-full border border-white/15 px-3 py-1.5 text-xs font-medium text-[#9CA3AF] transition hover:border-white/30 hover:text-white"
              >
                <Upload className="h-3.5 w-3.5" strokeWidth={1.5} />
                Upload a Different / Updated Flow
              </button>
            </div>
          ) : (
            <p className="mt-3 text-sm text-[#9CA3AF]">
              No flow definition uploaded for this agent. Upload a JSON or YAML flow
              definition to begin.
            </p>
          )}

          {(showUpload || (!flow && !flowsLoading)) && (
            <>
              <label
                className={`mt-3 flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-dashed border-white/20 px-4 py-8 text-sm text-[#9CA3AF] transition hover:border-white/40 hover:text-white ${
                  selectedAgentId == null ? "pointer-events-none opacity-40" : ""
                }`}
              >
                <Upload className="h-4 w-4" strokeWidth={1.5} />
                {uploading ? "Uploading…" : "Choose a .json or .yaml/.yml file"}
                <input
                  type="file"
                  accept=".json,.yaml,.yml,application/json,text/yaml"
                  className="hidden"
                  disabled={selectedAgentId == null || uploading}
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) onFileSelected(file);
                    e.target.value = "";
                  }}
                />
              </label>

              {showUpload && flow && (
                <button
                  onClick={() => setShowUpload(false)}
                  className="mt-2 text-xs text-[#9CA3AF] underline hover:text-white"
                >
                  Cancel — keep using the existing flow
                </button>
              )}
            </>
          )}

          {uploading && (
            <div className="mt-3 flex items-center gap-2 text-sm text-[#9CA3AF]">
              <Loader2 className="h-4 w-4 animate-spin" strokeWidth={1.5} />
              Parsing flow…
            </div>
          )}

          {uploadSuccess && (
            <div className="mt-3 flex items-start gap-2 rounded-lg border border-emerald-400/30 bg-emerald-400/10 px-3 py-2 text-sm text-emerald-300">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" strokeWidth={1.5} />
              {uploadSuccess}
            </div>
          )}

          {uploadErrors && (
            <div className="mt-3 rounded-lg border border-rose-400/30 bg-rose-400/10 px-3 py-2 text-sm text-rose-300">
              <div className="flex items-center gap-2 font-medium">
                <AlertTriangle className="h-4 w-4 shrink-0" strokeWidth={1.5} />
                Could not parse this flow
              </div>
              <ul className="mt-1.5 list-disc space-y-0.5 pl-6">
                {uploadErrors.map((e, i) => (
                  <li key={i}>{e}</li>
                ))}
              </ul>
              {uploadDiagnostics && (
                <div className="mt-3 border-t border-rose-400/20 pt-2 text-xs text-rose-200/80">
                  {uploadDiagnostics.top_level_keys && uploadDiagnostics.top_level_keys.length > 0 && (
                    <p>
                      Detected top-level keys:{" "}
                      <span className="font-mono">{uploadDiagnostics.top_level_keys.join(", ")}</span>
                    </p>
                  )}
                  {uploadDiagnostics.possible_sections && uploadDiagnostics.possible_sections.length > 0 && (
                    <p className="mt-1">
                      Possible structured sections:{" "}
                      <span className="font-mono">{uploadDiagnostics.possible_sections.join(", ")}</span>
                    </p>
                  )}
                </div>
              )}
            </div>
          )}

          {(showUpload || !flow) && flows.length > 0 && (
            <div className="mt-5 border-t border-white/10 pt-4">
              <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-[#9CA3AF]">
                <History className="h-3.5 w-3.5" strokeWidth={1.5} />
                Previously uploaded
              </div>
              <div className="flex flex-wrap gap-2">
                {flows.map((f) => (
                  <button
                    key={f.id}
                    onClick={() => onSelectStoredFlow(f.id)}
                    className={`rounded-full border px-3 py-1 text-xs transition ${
                      flow?.id === f.id
                        ? "border-white/50 bg-white/10 text-white"
                        : "border-white/15 text-[#9CA3AF] hover:border-white/30 hover:text-white"
                    }`}
                  >
                    {f.name} · {f.source_format} · {new Date(f.created_at).toLocaleString()}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Step 3: parsed nodes/edges */}
        {flow && (
          <div className="mt-6 rounded-xl border border-white/12 bg-white/2 p-6">
            <label className="block text-sm font-medium text-[#F8FAFC]">3. Detected nodes</label>
            {flow.extraction_method && (
              <p className="mt-1 text-xs text-[#9CA3AF]">
                Parsed using{" "}
                {flow.extraction_method === "llm" ? "LLM-assisted extraction" : "deterministic extraction"}
              </p>
            )}

            <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
              {flow.nodes.map((n) => (
                <button
                  key={n.id}
                  onClick={() => selectNode(n.id)}
                  className={`rounded-lg border p-4 text-left transition ${
                    selectedNodeId === n.id
                      ? "border-white/50 bg-white/10"
                      : "border-white/12 bg-black/30 hover:border-white/25 hover:bg-white/5"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <h3 className="font-medium text-[#F8FAFC]">{n.name}</h3>
                    {n.type && (
                      <span className="rounded-full border border-white/15 px-2 py-0.5 text-[10px] uppercase tracking-wide text-[#9CA3AF]">
                        {n.type}
                      </span>
                    )}
                  </div>
                  <p className="mt-1 text-xs text-[#9CA3AF]">
                    id: {n.id}
                    {n.source_path && <span className="text-slate-600"> · {n.source_path}</span>}
                  </p>
                  {n.purpose && <p className="mt-2 text-sm text-[#D1D5DB]">{n.purpose}</p>}
                  {n.expected_inputs.length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {n.expected_inputs.map((inp) => (
                        <span
                          key={inp}
                          className="rounded-md border border-white/10 bg-white/5 px-1.5 py-0.5 text-[11px] text-[#9CA3AF]"
                        >
                          {inp}
                        </span>
                      ))}
                    </div>
                  )}
                  <div className="mt-3 flex items-center gap-1.5 text-xs text-[#9CA3AF]">
                    <Target className="h-3.5 w-3.5" strokeWidth={1.5} />
                    {selectedNodeId === n.id ? "Selected for testing" : "Click to test this node"}
                  </div>
                </button>
              ))}
            </div>

            <label className="mt-6 block text-sm font-medium text-[#F8FAFC]">Edges</label>
            {flow.edges.length === 0 ? (
              <p className="mt-2 text-sm text-[#9CA3AF]">This flow has no edges.</p>
            ) : (
              <ul className="mt-2 space-y-1.5">
                {flow.edges.map((e, i) => (
                  <li key={i} className="flex items-center gap-2 text-sm text-[#D1D5DB]">
                    <FileCode2 className="h-3.5 w-3.5 shrink-0 text-[#9CA3AF]" strokeWidth={1.5} />
                    {nodeName(e.from)}
                    <ArrowRight className="h-3.5 w-3.5 shrink-0 text-[#9CA3AF]" strokeWidth={1.5} />
                    {nodeName(e.to)}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {/* Step 4: selected node -> test goal -> deterministic script (Phase 2) */}
        {selectedNode && (
          <div ref={nodeTestRef} className="mt-6 scroll-mt-24 rounded-xl border border-white/12 bg-white/2 p-6">
            <label className="block text-sm font-medium text-[#F8FAFC]">4. Node test</label>

            <div className="mt-3 rounded-lg border border-white/10 bg-black/30 p-4">
              <p className="text-[10px] font-medium uppercase tracking-wide text-[#9CA3AF]">Selected node</p>
              <div className="mt-1.5 flex items-center gap-2">
                <Target className="h-4 w-4 text-[#9CA3AF]" strokeWidth={1.5} />
                <h3 className="font-medium text-[#F8FAFC]">{selectedNode.name}</h3>
                <span className="text-xs text-[#9CA3AF]">({selectedNode.id})</span>
              </div>
              {selectedNode.purpose && (
                <p className="mt-2 text-sm text-[#D1D5DB]">{selectedNode.purpose}</p>
              )}
            </div>

            <label className="mt-5 block text-sm font-medium text-[#F8FAFC]">Test goal</label>
            <p className="mt-1 text-xs text-[#9CA3AF]">
              Describe what this test should verify, or leave it blank and let generation infer
              one from the node&apos;s purpose.
            </p>
            <textarea
              value={testGoal}
              onChange={(e) => onTestGoalChange(e.target.value)}
              placeholder="e.g. Verify the agent rejects an incorrect name before accepting the correct one."
              rows={2}
              className="mt-2 w-full resize-y rounded-lg border border-white/15 bg-black/40 px-3 py-2 text-sm text-[#F8FAFC] outline-none focus:border-white/40"
            />

            {!script && (
              <button
                onClick={onGenerateScript}
                disabled={generating}
                className="mt-3 inline-flex items-center gap-2 rounded-lg border border-white/20 bg-white/8 px-4 py-2 text-sm font-medium text-[#F8FAFC] transition hover:bg-white/15 disabled:opacity-50"
              >
                {generating ? (
                  <Loader2 className="h-4 w-4 animate-spin" strokeWidth={1.5} />
                ) : (
                  <Wand2 className="h-4 w-4" strokeWidth={1.5} />
                )}
                {generating ? "Generating…" : "Generate Test Script"}
              </button>
            )}

            {genErrors && (
              <div className="mt-3 rounded-lg border border-rose-400/30 bg-rose-400/10 px-3 py-2 text-sm text-rose-300">
                <div className="flex items-center gap-2 font-medium">
                  <AlertTriangle className="h-4 w-4 shrink-0" strokeWidth={1.5} />
                  Could not generate a valid script
                </div>
                <ul className="mt-1.5 list-disc space-y-0.5 pl-6">
                  {genErrors.map((e, i) => (
                    <li key={i}>{e}</li>
                  ))}
                </ul>
              </div>
            )}

            {script && (
              <div className="mt-6">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <label className="block text-sm font-medium text-[#F8FAFC]">Generated test script</label>
                  {scriptStatus === "saved" && (
                    <span className="rounded-full border border-emerald-400/30 bg-emerald-400/10 px-2.5 py-0.5 text-[11px] font-medium text-emerald-300">
                      Saved as test #{savedTest?.test_id}
                    </span>
                  )}
                  {scriptStatus === "modified" && (
                    <span className="rounded-full border border-amber-400/30 bg-amber-400/10 px-2.5 py-0.5 text-[11px] font-medium text-amber-300">
                      Unsaved changes
                    </span>
                  )}
                  {scriptStatus === "draft" && (
                    <span className="rounded-full border border-white/15 px-2.5 py-0.5 text-[11px] font-medium text-[#9CA3AF]">
                      Draft — not saved yet
                    </span>
                  )}
                </div>

                <div className="mt-2 flex items-start gap-2 rounded-lg border border-sky-400/25 bg-sky-400/10 px-3 py-2 text-xs text-sky-200">
                  <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={1.5} />
                  Caller lines are executed exactly as written during the test. The Voice
                  Agent&apos;s actual responses are not scripted — they are captured live and
                  checked against &quot;Expected Agent Behavior&quot; when this test runs.
                </div>

                <div className="mt-4 space-y-4">
                  {script.map((turn, i) => (
                    <div key={i} className="rounded-lg border border-white/12 bg-black/30 p-4">
                      <p className="text-xs font-medium uppercase tracking-wide text-[#9CA3AF]">
                        Turn {i + 1}
                      </p>

                      <div className="mt-3">
                        <label className="flex items-center gap-1.5 text-xs font-medium text-amber-300">
                          <ShieldCheck className="h-3.5 w-3.5" strokeWidth={1.5} />
                          Expected Agent Behavior (evaluation only — never spoken)
                        </label>
                        <textarea
                          value={turn.expected_agent_behavior}
                          onChange={(e) => updateTurn(i, "expected_agent_behavior", e.target.value)}
                          rows={2}
                          className="mt-1.5 w-full resize-y rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-sm text-[#F8FAFC] outline-none focus:border-amber-400/50"
                        />
                      </div>

                      <div className="mt-3">
                        <label className="flex items-center gap-1.5 text-xs font-medium text-emerald-300">
                          <Mic className="h-3.5 w-3.5" strokeWidth={1.5} />
                          Caller (spoken verbatim during the test)
                        </label>
                        <textarea
                          value={turn.caller_line}
                          onChange={(e) => updateTurn(i, "caller_line", e.target.value)}
                          rows={2}
                          className="mt-1.5 w-full resize-y rounded-lg border border-emerald-400/20 bg-emerald-400/5 px-3 py-2 text-sm text-[#F8FAFC] outline-none focus:border-emerald-400/50"
                        />
                      </div>
                    </div>
                  ))}
                </div>

                <p className="mt-4 flex items-center gap-1.5 text-xs text-[#9CA3AF]">
                  <MessageSquare className="h-3.5 w-3.5" strokeWidth={1.5} />
                  Caller lines are executed exactly as written during the test&#8202;— edit
                  them directly above, then save and run against the Voice Agent.
                </p>

                {/* Compact secondary actions: Regenerate / Delete. */}
                <div className="mt-4 flex flex-wrap items-center gap-2">
                  <button
                    onClick={onGenerateScript}
                    disabled={generating}
                    className="inline-flex items-center gap-1.5 rounded-full border border-white/15 px-3 py-1.5 text-xs font-medium text-[#9CA3AF] transition hover:border-white/30 hover:text-white disabled:opacity-50"
                  >
                    {generating ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" strokeWidth={1.5} />
                    ) : (
                      <RotateCcw className="h-3.5 w-3.5" strokeWidth={1.5} />
                    )}
                    {generating ? "Regenerating…" : "Regenerate"}
                  </button>
                  <button
                    onClick={() => setShowDeleteConfirm(true)}
                    className="inline-flex items-center gap-1.5 rounded-full border border-rose-400/25 px-3 py-1.5 text-xs font-medium text-rose-300 transition hover:border-rose-400/50 hover:bg-rose-400/10"
                  >
                    <Trash2 className="h-3.5 w-3.5" strokeWidth={1.5} />
                    Delete
                  </button>
                </div>

                {showDeleteConfirm && (
                  <div className="mt-3 rounded-lg border border-rose-400/30 bg-rose-400/10 px-3 py-3 text-sm text-rose-200">
                    <p>
                      {savedTest
                        ? `This clears the draft here in the workspace. Saved test #${savedTest.test_id} stays stored and can still be run or found elsewhere — it will not be deleted.`
                        : "Discard this generated draft? It hasn't been saved anywhere."}
                    </p>
                    <div className="mt-2 flex gap-2">
                      <button
                        onClick={() => {
                          onDeleteDraft();
                        }}
                        className="rounded-full border border-rose-400/40 px-3 py-1 text-xs font-medium text-rose-200 transition hover:bg-rose-400/20"
                      >
                        Clear Draft
                      </button>
                      <button
                        onClick={() => setShowDeleteConfirm(false)}
                        className="rounded-full border border-white/15 px-3 py-1 text-xs text-[#9CA3AF] transition hover:border-white/30 hover:text-white"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}

                {/* Primary actions: Save + Run are both visible as soon as a draft exists. */}
                <div className="mt-5 flex flex-wrap items-center gap-3">
                  <button
                    onClick={onSaveTest}
                    disabled={saving || !testGoal.trim()}
                    className="inline-flex items-center gap-2 rounded-lg border border-white/20 bg-white/8 px-4 py-2 text-sm font-medium text-[#F8FAFC] transition hover:bg-white/15 disabled:opacity-50"
                  >
                    {saving ? (
                      <Loader2 className="h-4 w-4 animate-spin" strokeWidth={1.5} />
                    ) : (
                      <Save className="h-4 w-4" strokeWidth={1.5} />
                    )}
                    {saving ? "Saving…" : "Save Test"}
                  </button>
                  <button
                    onClick={onRunTest}
                    disabled={running || !testGoal.trim()}
                    className="inline-flex items-center gap-2 rounded-lg border border-emerald-400/40 bg-emerald-400/15 px-4 py-2 text-sm font-medium text-emerald-200 transition hover:bg-emerald-400/25 disabled:opacity-50"
                  >
                    {running ? (
                      <Loader2 className="h-4 w-4 animate-spin" strokeWidth={1.5} />
                    ) : (
                      <PlayCircle className="h-4 w-4" strokeWidth={1.5} />
                    )}
                    {running ? "Running…" : "Run Test"}
                  </button>
                </div>
                <p className="mt-2 text-xs text-[#9CA3AF]">
                  Run Test automatically saves changes before starting.
                </p>

                {saveErrors && (
                  <div className="mt-3 rounded-lg border border-rose-400/30 bg-rose-400/10 px-3 py-2 text-sm text-rose-300">
                    <div className="flex items-center gap-2 font-medium">
                      <AlertTriangle className="h-4 w-4 shrink-0" strokeWidth={1.5} />
                      Could not save this test
                    </div>
                    <ul className="mt-1.5 list-disc space-y-0.5 pl-6">
                      {saveErrors.map((e, i) => (
                        <li key={i}>{e}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {runErrors && (
                  <div className="mt-3 rounded-lg border border-rose-400/30 bg-rose-400/10 px-3 py-2 text-sm text-rose-300">
                    <div className="flex items-center gap-2 font-medium">
                      <AlertTriangle className="h-4 w-4 shrink-0" strokeWidth={1.5} />
                      Could not run this test
                    </div>
                    <ul className="mt-1.5 list-disc space-y-0.5 pl-6">
                      {runErrors.map((e, i) => (
                        <li key={i}>{e}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {running && !report && (
                  <div className="mt-4 flex items-center gap-2 text-sm text-[#9CA3AF]">
                    <Loader2 className="h-4 w-4 animate-spin" strokeWidth={1.5} />
                    {runStatus === "queued" || runStatus === "running"
                      ? "Running against the Voice Agent through AgentShield's existing voice pipeline…"
                      : "Starting…"}
                  </div>
                )}

                {/* Result — reuses the dashboard's own transcript view. */}
                {report && report.conversations[0] && (
                  <div className="mt-6 rounded-lg border border-white/12 bg-black/30 p-4">
                    <div className="flex items-center justify-between gap-2">
                      <h4 className="font-medium text-[#F8FAFC]">Result</h4>
                      <span
                        className={`rounded-full border px-2.5 py-0.5 text-xs font-medium uppercase tracking-wide ${
                          report.conversations[0].verdict === "pass"
                            ? "border-emerald-400/40 text-emerald-300"
                            : "border-rose-400/40 text-rose-300"
                        }`}
                      >
                        {report.conversations[0].verdict ?? "unjudged"}
                        {report.conversations[0].severity ? ` · ${report.conversations[0].severity}` : ""}
                      </span>
                    </div>
                    {typeof report.reliability_score === "number" && (
                      <p className="mt-1 text-xs text-[#9CA3AF]">
                        Reliability score: {report.reliability_score}
                      </p>
                    )}
                    {report.conversations[0].explanation && (
                      <p className="mt-3 text-sm text-[#D1D5DB]">
                        <span className="font-medium text-[#F8FAFC]">Why: </span>
                        {report.conversations[0].explanation}
                      </p>
                    )}
                    {report.conversations[0].suggested_fix && (
                      <p className="mt-2 text-sm text-[#D1D5DB]">
                        <span className="font-medium text-[#F8FAFC]">Suggested fix: </span>
                        {report.conversations[0].suggested_fix}
                      </p>
                    )}
                    <TranscriptDetails
                      messages={report.conversations[0].messages}
                      label="Transcript"
                    />
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </section>
    </main>
  );
}
