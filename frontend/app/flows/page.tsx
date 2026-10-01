"use client";

import Link from "next/link";
import React, { useEffect, useMemo, useRef, useState } from "react";
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
  Save,
  PlayCircle,
  Trash2,
  Pencil,
  ChevronRight,
  ChevronDown,
  ArrowDown,
  Zap,
} from "lucide-react";
import {
  getInventory,
  getAgent,
  uploadAgentFlow,
  getAgentFlows,
  getFlow,
  getRun,
  getReport,
  previewFlowScenarios,
  listSavedFlowScenarios,
  generateFlowScenarioScript,
  saveFlowScenarioScript,
  deleteFlowScenario,
  runFlowScenario,
  saveFlowScenariosById,
  FlowScenarioPreview,
  FlowScriptTurn,
  FlowInterruptScenario,
  FlowScriptSetup,
  FlowScriptExpectations,
  InventoryAgent,
  FlowSummary,
  FlowDetail,
  Report,
} from "../lib/api";
import { TranscriptDetails } from "../dashboard/page";

// Upload -> parse -> store -> show the flow's structure, then plan its scenarios
// (normal paths + interrupts), generate/edit/save each one's deterministic caller
// script, and run it through the EXISTING Temporal execution pipeline — the same
// AgentTestWorkflow/RunGroupWorkflow, scripted runner, voice protocols, and Judge
// every other test uses (see backend/app/routers/flows.py and
// app/core/node_script.py). Results reuse the dashboard's own transcript view
// (TranscriptDetails, imported from ../dashboard/page) rather than duplicating it.

// One scenario's script in the UI. `saved`: a copy exists server-side. `dirty`: the
// local copy differs from it (always true for a never-saved draft).
interface FlowScriptWork {
  testGoal: string;
  turns: FlowScriptTurn[];
  saved: boolean;
  dirty: boolean;
  // Interrupt scripts only.
  setup?: FlowScriptSetup;
  expectations?: FlowScriptExpectations;
}

// "user.first_name: Daniel" lines -> {"user.first_name": "Daniel"}; blank values dropped.
function parseRecord(text: string): Record<string, string> {
  const record: Record<string, string> = {};
  for (const line of text.split("\n")) {
    const at = line.indexOf(":");
    if (at <= 0) continue;
    const key = line.slice(0, at).trim();
    const value = line.slice(at + 1).trim();
    if (key && value) record[key] = value;
  }
  return record;
}

type FlowScriptBusy = "generating" | "saving" | "deleting" | "running";

// One execution of a saved flow scenario, polled through the existing run endpoints.
interface FlowScenarioRun {
  runId: number | null;
  status: string | null;
  report: Report | null;
}

interface FlowScriptFailure {
  action: FlowScriptBusy;
  errors: string[];
}

const FAILURE_TITLES: Record<FlowScriptBusy, string> = {
  generating: "Could not generate a valid script",
  saving: "Could not save this script",
  deleting: "Could not delete this scenario",
  running: "Could not run this scenario",
};

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
  // Edges start collapsed — a large flow's edge list is long and rarely needed at a
  // glance. Purely a display toggle; the parsed edge data itself is untouched.

  // Flow-level scenario preview — read-only planning output for the loaded flow.
  const [scenarioPreview, setScenarioPreview] = useState<FlowScenarioPreview | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewErrors, setPreviewErrors] = useState<string[] | null>(null);
  const [expandedScenarioId, setExpandedScenarioId] = useState<string | null>(null);
  // Per-scenario conversation scripts (drafts + saved copies), keyed by scenario id.
  const [flowScripts, setFlowScripts] = useState<Record<string, FlowScriptWork>>({});
  const [flowScriptBusy, setFlowScriptBusy] = useState<Record<string, FlowScriptBusy>>({});
  const [flowScriptErrors, setFlowScriptErrors] = useState<Record<string, FlowScriptFailure>>({});
  const [editingScenarioId, setEditingScenarioId] = useState<string | null>(null);
  const [confirmDeleteScenarioId, setConfirmDeleteScenarioId] = useState<string | null>(null);
  const [flowRuns, setFlowRuns] = useState<Record<string, FlowScenarioRun>>({});
  // Interrupt scenarios are read-only for now: they can only be saved and deleted.
  const [savedInterruptIds, setSavedInterruptIds] = useState<Set<string>>(new Set());
  const [interruptBusy, setInterruptBusy] = useState<string | null>(null);
  const [interruptErrors, setInterruptErrors] = useState<string[] | null>(null);
  const [expandedInterruptId, setExpandedInterruptId] = useState<string | null>(null);
  // Test data for the flow's user.* record fields, used verbatim by interrupt scripts.
  const [recordText, setRecordText] = useState("");

  const flowId = flow?.id ?? null;
  const loadedFlowIdRef = useRef<number | null>(null);
  useEffect(() => {
    loadedFlowIdRef.current = flowId;
    setScenarioPreview(null);
    setPreviewErrors(null);
    setExpandedScenarioId(null);
    setFlowScripts({});
    setFlowScriptBusy({});
    setFlowScriptErrors({});
    setEditingScenarioId(null);
    setConfirmDeleteScenarioId(null);
    setFlowRuns({});
    setSavedInterruptIds(new Set());
    setInterruptBusy(null);
    setInterruptErrors(null);
    setExpandedInterruptId(null);
    setRecordText("");
  }, [flowId]);

  // Stops any in-flight flow-scenario run polling once the page goes away.
  useEffect(() => () => {
    loadedFlowIdRef.current = null;
  }, []);

  async function onPreviewScenarios() {
    if (flowId == null) return;
    setPreviewLoading(true);
    setPreviewErrors(null);
    setExpandedScenarioId(null);
    try {
      const [result, saved] = await Promise.all([
        previewFlowScenarios(flowId),
        listSavedFlowScenarios(flowId),
      ]);
      // Drop a response that lands after the user switched to another flow.
      if (loadedFlowIdRef.current !== result.flow_id) return;
      setScenarioPreview(result);
      setSavedInterruptIds(new Set(saved.scenarios.filter((s) => s.category === "interrupt").map((s) => s.id)));
      const testDataFields = [...(result.record_fields ?? []), ...(result.call_fields ?? [])];
      if (testDataFields.length) {
        setRecordText((cur) => cur || testDataFields.map((f) => `${f}: `).join("\n"));
      }
      // Show already-saved scripts in their cards. A local draft that hasn't been saved
      // yet is kept rather than overwritten.
      setFlowScripts((prev) => {
        const next = { ...prev };
        for (const s of saved.scenarios) {
          if (s.turns && !(prev[s.id]?.dirty)) {
            next[s.id] = {
              testGoal: s.test_goal ?? "", turns: s.turns, saved: true, dirty: false,
              setup: s.setup ?? undefined, expectations: s.expectations ?? undefined,
            };
          }
        }
        return next;
      });
    } catch (err) {
      if (loadedFlowIdRef.current !== flowId) return;
      setScenarioPreview(null);
      setPreviewErrors(extractErrors(err instanceof Error ? err.message : String(err)));
    } finally {
      setPreviewLoading(false);
    }
  }

  function setScriptBusy(id: string, busy: FlowScriptBusy | null) {
    setFlowScriptBusy((prev) => {
      const next = { ...prev };
      if (busy) next[id] = busy;
      else delete next[id];
      return next;
    });
  }

  function setScriptErrors(id: string, failure: FlowScriptFailure | null) {
    setFlowScriptErrors((prev) => {
      const next = { ...prev };
      if (failure) next[id] = failure;
      else delete next[id];
      return next;
    });
  }

  function failure(action: FlowScriptBusy, err: unknown): FlowScriptFailure {
    return { action, errors: extractErrors(err instanceof Error ? err.message : String(err)) };
  }

  // Save interrupt scenarios by id. The server re-plans them from the stored flow and
  // ignores anything but the ids; saving an already-saved one is a no-op.
  async function onSaveInterrupts(ids: string[], busyKey: string) {
    if (flowId == null || ids.length === 0) return;
    const requestFlowId = flowId;
    setInterruptBusy(busyKey);
    setInterruptErrors(null);
    try {
      const result = await saveFlowScenariosById(requestFlowId, ids);
      if (loadedFlowIdRef.current !== requestFlowId) return;
      setSavedInterruptIds(new Set(result.scenarios.filter((s) => s.category === "interrupt").map((s) => s.id)));
    } catch (err) {
      if (loadedFlowIdRef.current === requestFlowId) {
        setInterruptErrors(extractErrors(err instanceof Error ? err.message : String(err)));
      }
    } finally {
      setInterruptBusy(null);
    }
  }

  async function onDeleteInterrupt(id: string) {
    if (flowId == null) return;
    const requestFlowId = flowId;
    setConfirmDeleteScenarioId(null);
    setInterruptBusy(id);
    setInterruptErrors(null);
    try {
      await deleteFlowScenario(requestFlowId, id);
      if (loadedFlowIdRef.current !== requestFlowId) return;
      setSavedInterruptIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    } catch (err) {
      if (loadedFlowIdRef.current === requestFlowId) {
        setInterruptErrors(extractErrors(err instanceof Error ? err.message : String(err)));
      }
    } finally {
      setInterruptBusy(null);
    }
  }

  // Generate, or regenerate, the script for one scenario. The server resolves the path
  // from the scenario id, so this is always the SAME path. The draft is replaced only
  // on success.
  async function onGenerateFlowScript(id: string, testData?: Record<string, string>) {
    if (flowId == null) return;
    const requestFlowId = flowId;
    setScriptBusy(id, "generating");
    setScriptErrors(id, null);
    setConfirmDeleteScenarioId(null);
    try {
      const result = await generateFlowScenarioScript(requestFlowId, id, testData);
      if (loadedFlowIdRef.current !== requestFlowId) return;
      setFlowScripts((prev) => ({
        ...prev,
        [id]: {
          testGoal: result.test_goal, turns: result.turns, saved: prev[id]?.saved ?? false, dirty: true,
          setup: result.setup, expectations: result.expectations,
        },
      }));
      setEditingScenarioId((cur) => (cur === id ? null : cur));
    } catch (err) {
      if (loadedFlowIdRef.current !== requestFlowId) return;
      setScriptErrors(id, failure("generating", err));
    } finally {
      setScriptBusy(id, null);
    }
  }

  function updateFlowScript(id: string, change: (work: FlowScriptWork) => FlowScriptWork) {
    setFlowScripts((prev) => (prev[id] ? { ...prev, [id]: { ...change(prev[id]), dirty: true } } : prev));
  }

  async function onSaveFlowScript(id: string) {
    const work = flowScripts[id];
    if (flowId == null || !work) return;
    const requestFlowId = flowId;
    setScriptBusy(id, "saving");
    setScriptErrors(id, null);
    try {
      const saved = await saveFlowScenarioScript(
        requestFlowId, id, work.testGoal, work.turns,
        work.setup || work.expectations ? { setup: work.setup, expectations: work.expectations } : undefined
      );
      if (loadedFlowIdRef.current !== requestFlowId) return;
      setFlowScripts((prev) => ({
        ...prev,
        [id]: {
          testGoal: saved.test_goal ?? "", turns: saved.turns ?? [], saved: true, dirty: false,
          setup: saved.setup ?? undefined, expectations: saved.expectations ?? undefined,
        },
      }));
      if (saved.category === "interrupt") setSavedInterruptIds((prev) => new Set(prev).add(id));
      setEditingScenarioId((cur) => (cur === id ? null : cur));
    } catch (err) {
      if (loadedFlowIdRef.current !== requestFlowId) return;
      setScriptErrors(id, failure("saving", err));
    } finally {
      setScriptBusy(id, null);
    }
  }

  // Deletes the SAVED scenario row (if any) and clears the card back to its path. The
  // flow, its nodes/edges, test cases and runs are never touched.
  async function onDeleteFlowScript(id: string) {
    const work = flowScripts[id];
    if (flowId == null || !work) return;
    const requestFlowId = flowId;
    setConfirmDeleteScenarioId(null);
    if (work.saved) {
      setScriptBusy(id, "deleting");
      setScriptErrors(id, null);
      try {
        await deleteFlowScenario(requestFlowId, id);
      } catch (err) {
        if (loadedFlowIdRef.current === requestFlowId) {
          setScriptErrors(id, failure("deleting", err));
        }
        return;
      } finally {
        setScriptBusy(id, null);
      }
      if (loadedFlowIdRef.current !== requestFlowId) return;
    }
    setFlowScripts((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
    setScriptErrors(id, null);
    setEditingScenarioId((cur) => (cur === id ? null : cur));
    setFlowRuns((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
    setSavedInterruptIds((prev) => {
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
  }

  // Run a SAVED scenario's SAVED script, then poll the EXISTING run/report endpoints —
  // the same ones the dashboard polls — until it finishes.
  async function onRunFlowScenario(id: string) {
    const work = flowScripts[id];
    if (flowId == null || !work?.saved || work.dirty) return;
    const requestFlowId = flowId;
    const stillHere = () => loadedFlowIdRef.current === requestFlowId;
    setScriptBusy(id, "running");
    setScriptErrors(id, null);
    setConfirmDeleteScenarioId(null);
    setEditingScenarioId((cur) => (cur === id ? null : cur));
    setFlowRuns((prev) => ({ ...prev, [id]: { runId: null, status: null, report: null } }));
    try {
      const { run_id } = await runFlowScenario(requestFlowId, id);
      if (!stillHere()) return;
      setFlowRuns((prev) => ({ ...prev, [id]: { runId: run_id, status: "queued", report: null } }));
      for (;;) {
        await new Promise((resolve) => setTimeout(resolve, 2000));
        if (!stillHere()) return;
        const status = await getRun(run_id);
        if (!stillHere()) return;
        if (status.status === "done" || status.status === "error") {
          const report = await getReport(run_id);
          if (!stillHere()) return;
          setFlowRuns((prev) => ({ ...prev, [id]: { runId: run_id, status: status.status, report } }));
          return;
        }
        setFlowRuns((prev) => ({ ...prev, [id]: { runId: run_id, status: status.status, report: null } }));
      }
    } catch (err) {
      if (!stillHere()) return;
      setScriptErrors(id, failure("running", err));
      setFlowRuns((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
    } finally {
      setScriptBusy(id, null);
    }
  }

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

        {/* Step 3: parsed nodes/edges — a summary, with the details behind a toggle */}
        {flow && (
          <FlowStructure key={flow.id} flow={flow} nodeName={nodeName} />
        )}

        {/* Possible test scenarios — read-only preview of the backend planner's
            candidate paths (app.core.flow_graph). Nothing here saves or runs. */}
        {flow && (
          <div className="mt-6 rounded-xl border border-white/12 bg-white/2 p-6">
            <label className="block text-sm font-medium text-[#F8FAFC]">4. Possible test scenarios</label>
            <p className="mt-1 text-xs text-[#9CA3AF]">
              Paths through this flow worth testing, planned from its nodes and edges alone.
              Preview only — nothing is saved or run.
            </p>

            <button
              onClick={onPreviewScenarios}
              disabled={previewLoading}
              className="mt-3 inline-flex items-center gap-2 rounded-lg border border-white/20 bg-white/8 px-4 py-2 text-sm font-medium text-[#F8FAFC] transition hover:bg-white/15 disabled:opacity-50"
            >
              {previewLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" strokeWidth={1.5} />
              ) : (
                <Workflow className="h-4 w-4" strokeWidth={1.5} />
              )}
              {previewLoading ? "Planning…" : scenarioPreview ? "Refresh Preview" : "Preview Scenarios"}
            </button>

            {previewErrors && (
              <div className="mt-3 rounded-lg border border-rose-400/30 bg-rose-400/10 px-3 py-2 text-sm text-rose-300">
                <div className="flex items-center gap-2 font-medium">
                  <AlertTriangle className="h-4 w-4 shrink-0" strokeWidth={1.5} />
                  Could not preview scenarios
                </div>
                <ul className="mt-1.5 list-disc space-y-0.5 pl-6">
                  {previewErrors.map((e, i) => (
                    <li key={i}>{e}</li>
                  ))}
                </ul>
              </div>
            )}

            {scenarioPreview && (() => {
              const label = (id: string) => scenarioPreview.node_names[id] ?? id;
              const { unreachable } = scenarioPreview.graph;
              return (
                <>
                  <p className="mt-4 text-xs text-[#9CA3AF]">
                    {scenarioPreview.scenario_count} scenario{scenarioPreview.scenario_count === 1 ? "" : "s"}
                    {scenarioPreview.scenario_count === scenarioPreview.max_scenarios &&
                      ` — the preview limit; lower-priority paths may be omitted`}
                  </p>

                  {unreachable.length > 0 && (
                    <p className="mt-2 flex items-start gap-1.5 text-xs text-amber-300">
                      <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={1.5} />
                      Not reachable from the flow&apos;s start, so in no scenario:{" "}
                      {unreachable.map(label).join(", ")}
                    </p>
                  )}

                  {scenarioPreview.interrupt_scenarios && (
                    <p className="mt-4 text-xs font-medium uppercase tracking-wide text-[#9CA3AF]">Normal scenarios</p>
                  )}

                  {scenarioPreview.scenario_count === 0 ? (
                    <p className="mt-3 text-sm text-[#9CA3AF]">No scenarios could be planned for this flow.</p>
                  ) : (
                    <div className="mt-3 space-y-2">
                      {scenarioPreview.scenarios.map((s) => {
                        const expanded = expandedScenarioId === s.id;
                        return (
                          <div key={s.id} className="rounded-lg border border-white/12 bg-black/30">
                            <button
                              onClick={() => setExpandedScenarioId(expanded ? null : s.id)}
                              aria-expanded={expanded}
                              className="w-full rounded-lg p-4 text-left transition hover:bg-white/5"
                            >
                              <div className="flex items-center justify-between gap-2">
                                <div className="flex min-w-0 items-center gap-1.5">
                                  {expanded ? (
                                    <ChevronDown className="h-3.5 w-3.5 shrink-0 text-[#9CA3AF]" strokeWidth={1.5} />
                                  ) : (
                                    <ChevronRight className="h-3.5 w-3.5 shrink-0 text-[#9CA3AF]" strokeWidth={1.5} />
                                  )}
                                  <h3 className="truncate font-medium text-[#F8FAFC]">{s.name}</h3>
                                </div>
                                <div className="flex shrink-0 items-center gap-1.5">
                                  {flowScripts[s.id] && <FlowScriptStatus work={flowScripts[s.id]} />}
                                  <span className="rounded-full border border-white/15 px-2 py-0.5 text-[10px] uppercase tracking-wide text-[#9CA3AF]">
                                    {s.category}
                                  </span>
                                </div>
                              </div>
                              <p className="mt-1 pl-5 text-xs text-[#9CA3AF]">
                                {s.path_length} node{s.path_length === 1 ? "" : "s"}
                                {" · "}
                                <span className={s.contains_retry ? "text-amber-300" : undefined}>
                                  Retry: {s.contains_retry ? "Yes" : "No"}
                                </span>
                                {" · "}
                                Terminal:{" "}
                                {s.covered_terminals.length > 0
                                  ? s.covered_terminals.map(label).join(", ")
                                  : "none reached"}
                              </p>
                              {!expanded && (
                                <div className="mt-2 flex flex-wrap items-center gap-x-1.5 gap-y-1 pl-5 text-sm text-[#D1D5DB]">
                                  {s.path.map((id, i) => (
                                    <span key={i} className="inline-flex items-center gap-1.5">
                                      {i > 0 && (
                                        <ArrowRight className="h-3.5 w-3.5 shrink-0 text-[#9CA3AF]" strokeWidth={1.5} />
                                      )}
                                      {label(id)}
                                    </span>
                                  ))}
                                </div>
                              )}
                            </button>

                            {expanded && (
                              <ol className="px-4 pb-4 pl-9">
                                {s.path.map((id, i) => {
                                  const revisit = s.path.indexOf(id) < i;
                                  return (
                                    <li key={i}>
                                      {i > 0 && (
                                        <ArrowDown className="my-1 ml-1 h-3.5 w-3.5 text-[#9CA3AF]" strokeWidth={1.5} />
                                      )}
                                      <div className="flex flex-wrap items-baseline gap-x-2">
                                        <span className="text-sm text-[#F8FAFC]">{label(id)}</span>
                                        <span className="text-xs text-slate-600">{id}</span>
                                        {revisit && (
                                          <span className="rounded-full border border-amber-400/30 px-1.5 text-[10px] uppercase tracking-wide text-amber-300">
                                            revisit
                                          </span>
                                        )}
                                      </div>
                                    </li>
                                  );
                                })}
                              </ol>
                            )}

                            {expanded && (
                              <FlowScenarioScriptPanel
                                label={label}
                                work={flowScripts[s.id]}
                                busy={flowScriptBusy[s.id]}
                                errors={flowScriptErrors[s.id]}
                                editing={editingScenarioId === s.id}
                                confirmingDelete={confirmDeleteScenarioId === s.id}
                                onGenerate={() => onGenerateFlowScript(s.id)}
                                onToggleEdit={() =>
                                  setEditingScenarioId((cur) => (cur === s.id ? null : s.id))
                                }
                                onChangeGoal={(value) => updateFlowScript(s.id, (w) => ({ ...w, testGoal: value }))}
                                onChangeTurn={(index, field, value) =>
                                  updateFlowScript(s.id, (w) => ({
                                    ...w,
                                    turns: w.turns.map((t, i) => (i === index ? { ...t, [field]: value } : t)),
                                  }))
                                }
                                onSave={() => onSaveFlowScript(s.id)}
                                onAskDelete={() => setConfirmDeleteScenarioId(s.id)}
                                onCancelDelete={() => setConfirmDeleteScenarioId(null)}
                                onDelete={() => onDeleteFlowScript(s.id)}
                                run={flowRuns[s.id]}
                                onRun={() => onRunFlowScenario(s.id)}
                              />
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {scenarioPreview.interrupt_scenarios && (() => {
                    const interrupts = scenarioPreview.interrupt_scenarios;
                    const unsaved = interrupts.filter((s) => s.plannable && !savedInterruptIds.has(s.id)).map((s) => s.id);
                    const base = scenarioPreview.base_path;
                    return (
                      <div className="mt-6">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <p className="text-xs font-medium uppercase tracking-wide text-[#9CA3AF]">Interrupt scenarios</p>
                          {unsaved.length > 0 && (
                            <button
                              onClick={() => onSaveInterrupts(unsaved, "all")}
                              disabled={interruptBusy !== null}
                              className="inline-flex items-center gap-1.5 rounded-full border border-white/15 px-3 py-1.5 text-xs font-medium text-[#9CA3AF] transition hover:border-white/30 hover:text-white disabled:opacity-50"
                            >
                              {interruptBusy === "all" ? (
                                <Loader2 className="h-3.5 w-3.5 animate-spin" strokeWidth={1.5} />
                              ) : (
                                <Save className="h-3.5 w-3.5" strokeWidth={1.5} />
                              )}
                              Save all ({unsaved.length})
                            </button>
                          )}
                        </div>
                        <p className="mt-1 text-xs text-[#9CA3AF]">
                          Each fires one of the flow&apos;s interrupts at a planned point of the base path. Where it fires
                          is a planning convention — the flow does not declare it. Generate and save a script to run one.
                        </p>
                        {base && (
                          <p className="mt-2 text-xs text-[#9CA3AF]">
                            Base path: <span className="text-[#D1D5DB]">{base.name}</span> ({base.path.length} steps:{" "}
                            {label(base.path[0])} → … → {label(base.path[base.path.length - 1])})
                          </p>
                        )}
                        {(scenarioPreview.interrupt_scenario_total ?? 0) > interrupts.length && (
                          <p className="mt-1 text-xs text-[#9CA3AF]">
                            Showing {interrupts.length} of {scenarioPreview.interrupt_scenario_total} — the preview limit.
                          </p>
                        )}
                        {(scenarioPreview.record_fields?.length ?? 0) > 0 && (
                          <div className="mt-3">
                            <p className="text-xs font-medium text-[#D1D5DB]">Test data for generated scripts</p>
                            <p className="mt-0.5 text-xs text-[#9CA3AF]">
                              The flow&apos;s record fields — values here are spoken verbatim. Leave a value blank and the
                              script will declare any value it has to assume. <span className="font-mono">call.*</span> values
                              are sent to the voice agent when the call is created (e.g. which test patient it loads).
                            </p>
                            <textarea
                              value={recordText}
                              onChange={(e) => setRecordText(e.target.value)}
                              rows={Math.min(7, (scenarioPreview.record_fields?.length ?? 0) + (scenarioPreview.call_fields?.length ?? 0) + 1)}
                              className="mt-1.5 w-full resize-y rounded-lg border border-white/15 bg-black/40 px-3 py-2 font-mono text-xs text-[#F8FAFC] outline-none focus:border-white/40"
                            />
                          </div>
                        )}
                        {interruptErrors && <ScriptErrors title="Could not update interrupt scenarios" errors={interruptErrors} />}
                        {interrupts.length === 0 ? (
                          <p className="mt-3 text-sm text-[#9CA3AF]">This flow declares no interrupts.</p>
                        ) : (
                          <div className="mt-3 space-y-2">
                            {interrupts.map((s) => (
                              <InterruptScenarioCard
                                key={s.id}
                                scenario={s}
                                label={label}
                                expanded={expandedInterruptId === s.id}
                                onToggle={() => setExpandedInterruptId((cur) => (cur === s.id ? null : s.id))}
                                saved={savedInterruptIds.has(s.id)}
                                busy={interruptBusy === s.id || interruptBusy === "all"}
                                disabled={interruptBusy !== null}
                                confirmingDelete={confirmDeleteScenarioId === s.id}
                                onSave={() => onSaveInterrupts([s.id], s.id)}
                                onAskDelete={() => setConfirmDeleteScenarioId(s.id)}
                                onCancelDelete={() => setConfirmDeleteScenarioId(null)}
                                onDelete={() => onDeleteInterrupt(s.id)}
                                hasScript={!!flowScripts[s.id]}
                                scriptPanel={
                                  <FlowScenarioScriptPanel
                                    label={label}
                                    work={flowScripts[s.id]}
                                    busy={flowScriptBusy[s.id]}
                                    errors={flowScriptErrors[s.id]}
                                    editing={editingScenarioId === s.id}
                                    confirmingDelete={confirmDeleteScenarioId === s.id}
                                    onGenerate={() => onGenerateFlowScript(s.id, parseRecord(recordText))}
                                    onToggleEdit={() => setEditingScenarioId((cur) => (cur === s.id ? null : s.id))}
                                    onChangeGoal={(value) => updateFlowScript(s.id, (w) => ({ ...w, testGoal: value }))}
                                    onChangeTurn={(index, field, value) =>
                                      updateFlowScript(s.id, (w) => ({
                                        ...w,
                                        turns: w.turns.map((t, i) => (i === index ? { ...t, [field]: value } : t)),
                                      }))
                                    }
                                    onChangeExpectations={(expectations) =>
                                      updateFlowScript(s.id, (w) => ({ ...w, expectations }))
                                    }
                                    onSave={() => onSaveFlowScript(s.id)}
                                    onAskDelete={() => setConfirmDeleteScenarioId(s.id)}
                                    onCancelDelete={() => setConfirmDeleteScenarioId(null)}
                                    onDelete={() => onDeleteFlowScript(s.id)}
                                    run={flowRuns[s.id]}
                                    onRun={() => onRunFlowScenario(s.id)}
                                  />
                                }
                              />
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })()}
                </>
              );
            })()}
          </div>
        )}

      </section>
    </main>
  );
}

const OUTCOME_LABELS: Record<string, string> = { goto: "Goto", end: "End", resume: "Resume" };

// Readable lines from a node's `purpose`. A modular flow step's purpose is its prompt
// block as stored by the parser (e.g. "{'en-US': {'ask': {'text': '...'}, ...}}"); the
// spoken text / intent of each part is pulled out of it. Any other purpose is plain
// text and shown as is.
const PROMPT_PART_LABELS: Record<string, string> = {
  say: "Says",
  ask: "Asks",
  reask: "Asks again",
  clarify: "If the caller asks something else",
  readback: "Reads back",
  confirm: "Confirms",
  declined: "If the caller declines",
  invite: "Invites",
};

function purposeLines(purpose: string): { label: string; text: string }[] {
  const text = (purpose || "").trim();
  if (!text) return [];
  if (!text.startsWith("{")) return [{ label: "Purpose", text }];
  const part = /'(\w+)':\s*\{\s*'(text|intent)':\s*(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")/g;
  const lines: { label: string; text: string }[] = [];
  let m: RegExpExecArray | null;
  while ((m = part.exec(text)) !== null) {
    const spoken = (m[3] ?? m[4] ?? "").replace(/\\'/g, "'").replace(/\\"/g, '"');
    const base = PROMPT_PART_LABELS[m[1]] ?? m[1];
    lines.push({ label: m[2] === "intent" ? `${base} (in its own words)` : base, text: spoken });
  }
  return lines.length ? lines : [{ label: "Details", text }];
}

function FlowStructure({ flow, nodeName }: { flow: FlowDetail; nodeName: (id: string) => string }) {
  const [open, setOpen] = useState(false);
  const [typeFilter, setTypeFilter] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [edgesOpen, setEdgesOpen] = useState(false);

  const typeCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const n of flow.nodes) if (n.type) counts.set(n.type, (counts.get(n.type) ?? 0) + 1);
    return [...counts.entries()];
  }, [flow.nodes]);
  const shown = typeFilter ? flow.nodes.filter((n) => n.type === typeFilter) : flow.nodes;

  const chip = (active: boolean) =>
    `rounded-full border px-2.5 py-1 text-xs transition ${
      active ? "border-white/40 bg-white/10 text-[#F8FAFC]" : "border-white/12 text-[#9CA3AF] hover:border-white/25 hover:text-[#D1D5DB]"
    }`;

  return (
    <div className="mt-6 rounded-xl border border-white/12 bg-white/2 p-6">
      <label className="block text-sm font-medium text-[#F8FAFC]">3. Flow structure</label>
      <p className="mt-1 text-xs text-[#9CA3AF]">
        {flow.nodes.length} nodes · {flow.edges.length} edges
        {flow.extraction_method && (
          <> · parsed using {flow.extraction_method === "llm" ? "LLM-assisted extraction" : "deterministic extraction"}</>
        )}
      </p>
      {typeCounts.length > 0 && (
        <p className="mt-1 text-xs text-[#9CA3AF]">
          {typeCounts.map(([t, c]) => `${c} ${t}`).join(" · ")}
        </p>
      )}

      <button
        onClick={() => setOpen((o) => !o)}
        className="mt-3 inline-flex items-center gap-2 rounded-lg border border-white/20 bg-white/8 px-4 py-2 text-sm font-medium text-[#F8FAFC] transition hover:bg-white/15"
      >
        {open ? <ChevronDown className="h-4 w-4" strokeWidth={1.5} /> : <ChevronRight className="h-4 w-4" strokeWidth={1.5} />}
        {open ? "Hide node details" : "View node details"}
      </button>

      {open && (
        <div className="mt-4">
          {typeCounts.length > 1 && (
            <div className="flex flex-wrap gap-1.5">
              <button onClick={() => setTypeFilter(null)} className={chip(typeFilter === null)}>
                All ({flow.nodes.length})
              </button>
              {typeCounts.map(([t, c]) => (
                <button key={t} onClick={() => setTypeFilter(t)} className={chip(typeFilter === t)}>
                  {t} ({c})
                </button>
              ))}
            </div>
          )}

          <ul className="mt-3 divide-y divide-white/8 overflow-hidden rounded-lg border border-white/10">
            {shown.map((n) => {
              const expanded = expandedId === n.id;
              const lines = purposeLines(n.purpose);
              return (
                <li key={n.id} className="bg-black/20">
                  <button
                    onClick={() => setExpandedId((cur) => (cur === n.id ? null : n.id))}
                    className="flex w-full items-center gap-2 px-3 py-2 text-left transition hover:bg-white/5"
                  >
                    {expanded ? (
                      <ChevronDown className="h-3.5 w-3.5 shrink-0 text-[#9CA3AF]" strokeWidth={1.5} />
                    ) : (
                      <ChevronRight className="h-3.5 w-3.5 shrink-0 text-[#9CA3AF]" strokeWidth={1.5} />
                    )}
                    <span className="truncate text-sm text-[#F8FAFC]">{n.name}</span>
                    {n.name !== n.id && <span className="truncate text-xs text-[#9CA3AF]">{n.id}</span>}
                    {n.type && (
                      <span className="ml-auto shrink-0 rounded-full border border-white/15 px-2 py-0.5 text-[10px] uppercase tracking-wide text-[#9CA3AF]">
                        {n.type}
                      </span>
                    )}
                  </button>
                  {expanded && (
                    <div className="space-y-2 px-3 pb-3 pl-8">
                      {lines.length === 0 ? (
                        <p className="text-xs text-[#9CA3AF]">
                          {n.type === "branch" ? "Routes the call to the next step — the agent says nothing here." : "No details in the flow."}
                        </p>
                      ) : (
                        lines.map((l, i) => (
                          <div key={i}>
                            <p className="text-[11px] font-medium uppercase tracking-wide text-[#9CA3AF]">{l.label}</p>
                            <p className="text-sm text-[#D1D5DB]">{l.text}</p>
                          </div>
                        ))
                      )}
                      {n.expected_inputs.length > 0 && (
                        <div className="flex flex-wrap gap-1.5">
                          {n.expected_inputs.map((inp) => (
                            <span key={inp} className="rounded-md border border-white/10 bg-white/5 px-1.5 py-0.5 text-[11px] text-[#9CA3AF]">
                              {inp}
                            </span>
                          ))}
                        </div>
                      )}
                      {n.source_path && <p className="pt-1 text-xs text-slate-600">{n.source_path}</p>}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>

          {flow.edges.length === 0 ? (
            <p className="mt-4 text-sm text-[#9CA3AF]">This flow has no edges.</p>
          ) : (
            <>
              <button
                onClick={() => setEdgesOpen((e) => !e)}
                className="mt-4 flex items-center gap-1.5 text-sm font-medium text-[#F8FAFC] transition hover:text-white"
              >
                {edgesOpen ? (
                  <ChevronDown className="h-3.5 w-3.5 shrink-0 text-[#9CA3AF]" strokeWidth={1.5} />
                ) : (
                  <ChevronRight className="h-3.5 w-3.5 shrink-0 text-[#9CA3AF]" strokeWidth={1.5} />
                )}
                Edges ({flow.edges.length})
              </button>
              {edgesOpen && (
                <ul className="mt-2 space-y-1.5 pl-5">
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
            </>
          )}
        </div>
      )}
    </div>
  );
}

function placementText(s: FlowInterruptScenario, label: (id: string) => string): string {
  const p = s.placement;
  if (p.policy === "after_greeting") return "after the greeting (first caller turn)";
  if (p.policy === "after_field_confirmed") {
    return `after "${p.field}" is collected at ${label(p.field_step ?? "")} (first caller turn after it)`;
  }
  if (p.policy === "before_field_confirmed") {
    return `before "${p.field}" can be confirmed, at ${label(p.field_step ?? "")}`;
  }
  return "not placed";
}

// One interrupt scenario, read-only: the planned event and the path around it. The
// interrupt is drawn as an event between segments, never as an ordinary edge.
function InterruptScenarioCard({
  scenario: s,
  label,
  expanded,
  onToggle,
  saved,
  busy,
  disabled,
  confirmingDelete,
  onSave,
  onAskDelete,
  onCancelDelete,
  onDelete,
  hasScript,
  scriptPanel,
}: {
  scenario: FlowInterruptScenario;
  label: (id: string) => string;
  expanded: boolean;
  onToggle: () => void;
  saved: boolean;
  busy: boolean;
  disabled: boolean;
  confirmingDelete: boolean;
  onSave: () => void;
  onAskDelete: () => void;
  onCancelDelete: () => void;
  onDelete: () => void;
  hasScript: boolean;
  scriptPanel: React.ReactNode;
}) {
  const outcome = s.interrupt.outcome;
  const steps = (ids: string[]) => (
    <div className="flex flex-wrap items-center gap-x-1.5 gap-y-1 text-sm text-[#D1D5DB]">
      {ids.map((id, i) => (
        <span key={i} className="inline-flex items-center gap-1.5">
          {i > 0 && <ArrowRight className="h-3.5 w-3.5 shrink-0 text-[#9CA3AF]" strokeWidth={1.5} />}
          {label(id)}
        </span>
      ))}
    </div>
  );
  return (
    <div className="rounded-lg border border-white/12 bg-black/30">
      <button onClick={onToggle} aria-expanded={expanded} className="w-full rounded-lg p-4 text-left transition hover:bg-white/5">
        <div className="flex items-center justify-between gap-2">
          <div className="flex min-w-0 items-center gap-1.5">
            {expanded ? (
              <ChevronDown className="h-3.5 w-3.5 shrink-0 text-[#9CA3AF]" strokeWidth={1.5} />
            ) : (
              <ChevronRight className="h-3.5 w-3.5 shrink-0 text-[#9CA3AF]" strokeWidth={1.5} />
            )}
            <Zap className="h-3.5 w-3.5 shrink-0 text-amber-300" strokeWidth={1.5} />
            <h3 className="truncate font-medium text-[#F8FAFC]">{s.name}</h3>
          </div>
          <div className="flex shrink-0 items-center gap-1.5">
            {saved && (
              <span className="rounded-full border border-emerald-400/30 bg-emerald-400/10 px-2 py-0.5 text-[10px] font-medium text-emerald-300">
                Saved
              </span>
            )}
            <span className="rounded-full border border-amber-400/30 px-2 py-0.5 text-[10px] uppercase tracking-wide text-amber-300">
              {OUTCOME_LABELS[outcome] ?? outcome}
            </span>
          </div>
        </div>
        <p className="mt-1 pl-5 text-xs text-[#9CA3AF]">
          {s.plannable ? <>Injected at: {label(s.placement.step ?? "")}</> : <>Not plannable: {s.reason}</>}
        </p>
      </button>

      {expanded && (
        <div className="border-t border-white/10 px-4 py-4 pl-9">
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
            <dt className="text-[#9CA3AF]">Interrupt</dt>
            <dd className="font-mono text-[#D1D5DB]">{s.interrupt.id}</dd>
            <dt className="text-[#9CA3AF]">Outcome</dt>
            <dd className="text-[#D1D5DB]">{OUTCOME_LABELS[outcome] ?? outcome}</dd>
            {s.plannable && (
              <>
                <dt className="text-[#9CA3AF]">Injected at</dt>
                <dd className="text-[#D1D5DB]">{label(s.placement.step ?? "")}</dd>
              </>
            )}
            {outcome === "goto" && s.interrupt.target && (
              <>
                <dt className="text-[#9CA3AF]">Target</dt>
                <dd className="text-[#D1D5DB]">{label(s.interrupt.target)}</dd>
              </>
            )}
            {outcome === "end" && (
              <>
                <dt className="text-[#9CA3AF]">End status</dt>
                <dd className="text-[#D1D5DB]">{s.interrupt.end_status ?? "not declared"}</dd>
              </>
            )}
            {outcome === "resume" && (
              <>
                <dt className="text-[#9CA3AF]">Resume</dt>
                <dd className="text-[#D1D5DB]">returns to the interrupted step</dd>
              </>
            )}
            {s.interrupt.when && (
              <>
                <dt className="text-[#9CA3AF]">Condition</dt>
                <dd className="font-mono text-[#D1D5DB]">{s.interrupt.when}</dd>
              </>
            )}
            <dt className="text-[#9CA3AF]">Placement</dt>
            <dd className="text-[#D1D5DB]">{placementText(s, label)} — planning convention, not declared by the flow</dd>
          </dl>

          {s.plannable && (
            <div className="mt-4 space-y-2">
              {s.segments.map((g, i) =>
                g.kind === "interrupt" ? (
                  <div key={i} className="flex items-center gap-1.5 pl-2 text-sm text-amber-300">
                    <span className="text-[#9CA3AF]">⟶</span>
                    <Zap className="h-3.5 w-3.5 shrink-0" strokeWidth={1.5} />
                    {s.name}
                    {g.outcome === "end" && (
                      <span className="text-xs text-[#9CA3AF]">— call ends{g.end_status ? ` · ${g.end_status}` : ""}</span>
                    )}
                  </div>
                ) : (
                  <div key={i}>
                    {g.kind === "resumed" && (
                      <p className="mb-1 text-[10px] uppercase tracking-wide text-[#9CA3AF]">
                        {g.via ? `resumes after ${label(g.via)} at` : "resumes at"} the interrupted step
                      </p>
                    )}
                    {g.kind !== "prefix" && g.kind !== "resumed" && (
                      <span className="mr-1.5 text-[#9CA3AF]">⟶</span>
                    )}
                    {steps(g.steps)}
                  </div>
                )
              )}
            </div>
          )}

          {s.plannable && <div className="-mx-4 mt-4 -mb-4">{scriptPanel}</div>}

          {!hasScript && (
          <div className="mt-4 flex flex-wrap items-center gap-2">
            {s.plannable && !saved && (
              <button
                onClick={onSave}
                disabled={disabled}
                className="inline-flex items-center gap-2 rounded-lg border border-white/20 bg-white/8 px-4 py-2 text-sm font-medium text-[#F8FAFC] transition hover:bg-white/15 disabled:opacity-50"
              >
                {busy ? <Loader2 className="h-4 w-4 animate-spin" strokeWidth={1.5} /> : <Save className="h-4 w-4" strokeWidth={1.5} />}
                Save
              </button>
            )}
            {saved && (
              <button
                onClick={onAskDelete}
                disabled={disabled}
                className="inline-flex items-center gap-1.5 rounded-full border border-rose-400/25 px-3 py-1.5 text-xs font-medium text-rose-300 transition hover:border-rose-400/50 hover:bg-rose-400/10 disabled:opacity-50"
              >
                {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" strokeWidth={1.5} /> : <Trash2 className="h-3.5 w-3.5" strokeWidth={1.5} />}
                Delete
              </button>
            )}
            <span className="text-xs text-[#9CA3AF]">
              {saved ? "Saved without a script." : "Save the scenario on its own, or generate a script above."}
            </span>
          </div>
          )}
          {confirmingDelete && (
            <div className="mt-3 rounded-lg border border-rose-400/30 bg-rose-400/10 px-3 py-3 text-sm text-rose-200">
              <p>Delete this saved interrupt scenario? The flow and its other scenarios are not affected.</p>
              <div className="mt-2 flex gap-2">
                <button
                  onClick={onDelete}
                  className="rounded-full border border-rose-400/40 px-3 py-1 text-xs font-medium text-rose-200 transition hover:bg-rose-400/20"
                >
                  Delete
                </button>
                <button
                  onClick={onCancelDelete}
                  className="rounded-full border border-white/15 px-3 py-1 text-xs text-[#9CA3AF] transition hover:border-white/30 hover:text-white"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function FlowScriptStatus({ work }: { work: FlowScriptWork }) {
  if (work.saved && !work.dirty) {
    return (
      <span className="rounded-full border border-emerald-400/30 bg-emerald-400/10 px-2 py-0.5 text-[10px] font-medium text-emerald-300">
        Saved
      </span>
    );
  }
  if (work.saved) {
    return (
      <span className="rounded-full border border-amber-400/30 bg-amber-400/10 px-2 py-0.5 text-[10px] font-medium text-amber-300">
        Unsaved changes
      </span>
    );
  }
  return (
    <span className="rounded-full border border-white/15 px-2 py-0.5 text-[10px] font-medium text-[#9CA3AF]">
      Draft — not saved
    </span>
  );
}

function ScriptErrors({ title, errors }: { title: string; errors: string[] }) {
  return (
    <div className="mt-3 rounded-lg border border-rose-400/30 bg-rose-400/10 px-3 py-2 text-sm text-rose-300">
      <div className="flex items-center gap-2 font-medium">
        <AlertTriangle className="h-4 w-4 shrink-0" strokeWidth={1.5} />
        {title}
      </div>
      <ul className="mt-1.5 list-disc space-y-0.5 pl-6">
        {errors.map((e, i) => (
          <li key={i}>{e}</li>
        ))}
      </ul>
    </div>
  );
}

// The conversation script for ONE flow scenario, shown inside its expanded card.
// Read mode shows plain text; Edit swaps in textareas. Generate/Regenerate/Save/Delete
// only — there is deliberately no Run here.
function FlowScenarioScriptPanel({
  label,
  work,
  busy,
  errors,
  editing,
  confirmingDelete,
  onGenerate,
  onToggleEdit,
  onChangeGoal,
  onChangeTurn,
  onSave,
  onAskDelete,
  onCancelDelete,
  onDelete,
  run,
  onRun,
  onChangeExpectations,
  runnable = true,
}: {
  label: (nodeId: string) => string;
  work: FlowScriptWork | undefined;
  busy: FlowScriptBusy | undefined;
  errors: FlowScriptFailure | undefined;
  editing: boolean;
  confirmingDelete: boolean;
  onGenerate: () => void;
  onToggleEdit: () => void;
  onChangeGoal: (value: string) => void;
  onChangeTurn: (index: number, field: "expected_agent_behavior" | "caller_line", value: string) => void;
  onSave: () => void;
  onAskDelete: () => void;
  onCancelDelete: () => void;
  onDelete: () => void;
  run: FlowScenarioRun | undefined;
  onRun: () => void;
  onChangeExpectations?: (expectations: FlowScriptExpectations) => void;
  runnable?: boolean;
}) {
  const generating = busy === "generating";
  const pillButton =
    "inline-flex items-center gap-1.5 rounded-full border border-white/15 px-3 py-1.5 text-xs font-medium text-[#9CA3AF] transition hover:border-white/30 hover:text-white disabled:opacity-50";

  if (!work) {
    return (
      <div className="border-t border-white/10 px-4 py-4">
        <button
          onClick={onGenerate}
          disabled={!!busy}
          className="inline-flex items-center gap-2 rounded-lg border border-white/20 bg-white/8 px-4 py-2 text-sm font-medium text-[#F8FAFC] transition hover:bg-white/15 disabled:opacity-50"
        >
          {generating ? (
            <Loader2 className="h-4 w-4 animate-spin" strokeWidth={1.5} />
          ) : (
            <Wand2 className="h-4 w-4" strokeWidth={1.5} />
          )}
          {generating ? "Generating…" : "Generate Script"}
        </button>
        {errors && <ScriptErrors title={FAILURE_TITLES[errors.action]} errors={errors.errors} />}
      </div>
    );
  }

  return (
    <div className="border-t border-white/10 px-4 py-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs font-medium uppercase tracking-wide text-[#9CA3AF]">Test goal</p>
        <FlowScriptStatus work={work} />
      </div>
      {editing ? (
        <textarea
          autoFocus
          value={work.testGoal}
          onChange={(e) => onChangeGoal(e.target.value)}
          disabled={!!busy}
          rows={2}
          className="mt-1.5 w-full resize-y rounded-lg border border-white/15 bg-black/40 px-3 py-2 text-sm text-[#F8FAFC] outline-none focus:border-white/40"
        />
      ) : (
        <p className="mt-1.5 text-sm text-[#F8FAFC]">{work.testGoal}</p>
      )}

      <p className="mt-5 text-xs font-medium uppercase tracking-wide text-[#9CA3AF]">Conversation script</p>
      <div className="mt-2 flex items-start gap-2 rounded-lg border border-sky-400/25 bg-sky-400/10 px-3 py-2 text-xs text-sky-200">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={1.5} />
        Caller lines are the planned intent for this turn — at runtime they may be naturally
        rephrased to actually answer what the Voice Agent just said, without changing what they
        mean. The Voice Agent&apos;s real replies are captured live and checked against
        &quot;Expected Agent Behavior&quot;.
      </div>

      <div className="mt-3 space-y-3">
        {work.turns.map((turn, i) => {
          const kind = turn.type ?? "caller";
          const speaks = kind === "caller" || kind === "interrupt";
          return (
          <div
            key={i}
            className={`rounded-lg border p-4 ${kind === "interrupt" ? "border-amber-400/30 bg-amber-400/5" : "border-white/12 bg-black/30"}`}
          >
            <p className="text-xs font-medium uppercase tracking-wide text-[#9CA3AF]">
              Turn {i + 1}
              {kind === "interrupt" && (
                <span className="ml-1.5 inline-flex items-center gap-1 normal-case tracking-normal text-amber-300">
                  <Zap className="h-3.5 w-3.5" strokeWidth={1.5} /> Interrupt: {turn.interrupt_key}
                </span>
              )}
              {(kind === "listen" || kind === "readback") && (
                <span className="ml-1.5 font-mono normal-case tracking-normal text-sky-300">⟨{kind}⟩</span>
              )}
              {turn.step != null && (
                <span className="normal-case tracking-normal text-slate-500">
                  {" "}· step {turn.step}: {label(turn.node_id ?? "")}
                </span>
              )}
            </p>

            <p className="mt-3 flex items-center gap-1.5 text-xs font-medium text-amber-300">
              <ShieldCheck className="h-3.5 w-3.5" strokeWidth={1.5} />
              {speaks
                ? "Expected Agent Behavior (evaluation only — never spoken)"
                : kind === "readback"
                  ? "The agent reads the value back (evaluation only) — the next turn confirms it"
                  : "The agent speaks here and the caller does not reply (evaluation only)"}
            </p>
            {editing ? (
              <textarea
                value={turn.expected_agent_behavior}
                onChange={(e) => onChangeTurn(i, "expected_agent_behavior", e.target.value)}
                disabled={!!busy}
                rows={2}
                className="mt-1.5 w-full resize-y rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-sm text-[#F8FAFC] outline-none focus:border-amber-400/50"
              />
            ) : (
              <p className="mt-1 text-sm text-[#D1D5DB]">{turn.expected_agent_behavior}</p>
            )}

            {speaks && (<>
            <p className="mt-3 flex items-center gap-1.5 text-xs font-medium text-emerald-300">
              <Mic className="h-3.5 w-3.5" strokeWidth={1.5} />
              {kind === "interrupt" ? "Caller raises the interrupt (planned line — may be rephrased live)" : "Caller (planned line — may be rephrased live)"}
            </p>
            {editing ? (
              <textarea
                value={turn.caller_line ?? ""}
                onChange={(e) => onChangeTurn(i, "caller_line", e.target.value)}
                disabled={!!busy}
                rows={2}
                className="mt-1.5 w-full resize-y rounded-lg border border-emerald-400/20 bg-emerald-400/5 px-3 py-2 text-sm text-[#F8FAFC] outline-none focus:border-emerald-400/50"
              />
            ) : (
              <p className="mt-1 text-sm text-[#F8FAFC]">&ldquo;{turn.caller_line}&rdquo;</p>
            )}
            </>)}
          </div>
          );
        })}
      </div>

      {work.setup && (
        <div className="mt-5">
          <p className="text-xs font-medium uppercase tracking-wide text-[#9CA3AF]">Setup</p>
          <p className="mt-1 text-xs text-[#9CA3AF]">Record values used verbatim by the caller:</p>
          {Object.keys(work.setup.record).length > 0 ? (
            <dl className="mt-1 grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5 font-mono text-xs text-[#D1D5DB]">
              {Object.entries(work.setup.record).map(([k, v]) => (
                <React.Fragment key={k}><dt className="text-[#9CA3AF]">{k}</dt><dd>{v}</dd></React.Fragment>
              ))}
            </dl>
          ) : (
            <p className="mt-1 text-xs text-[#D1D5DB]">none supplied</p>
          )}
          {work.setup.call && Object.keys(work.setup.call).length > 0 && (
            <>
              <p className="mt-2 text-xs text-[#9CA3AF]">Voice agent call created with:</p>
              <dl className="mt-1 grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5 font-mono text-xs text-[#D1D5DB]">
                {Object.entries(work.setup.call).map(([k, v]) => (
                  <React.Fragment key={k}><dt className="text-[#9CA3AF]">{k}</dt><dd>{v}</dd></React.Fragment>
                ))}
              </dl>
            </>
          )}
          {work.setup.assumed_values && Object.keys(work.setup.assumed_values).length > 0 && (
            <div className="mt-2 rounded-lg border border-amber-400/30 bg-amber-400/10 px-3 py-2 text-xs text-amber-200">
              Assumed by the generator because no record value was supplied — review before running:
              <dl className="mt-1 grid grid-cols-[auto_1fr] gap-x-4 font-mono">
                {Object.entries(work.setup.assumed_values).map(([k, v]) => (
                  <React.Fragment key={k}><dt>{k}</dt><dd>{v}</dd></React.Fragment>
                ))}
              </dl>
            </div>
          )}
        </div>
      )}

      {work.expectations && (
        <div className="mt-5">
          <p className="text-xs font-medium uppercase tracking-wide text-[#9CA3AF]">Expected outcome</p>
          {editing && onChangeExpectations ? (
            <textarea
              value={work.expectations.outcome}
              onChange={(e) => onChangeExpectations({ ...work.expectations!, outcome: e.target.value })}
              disabled={!!busy}
              rows={2}
              className="mt-1.5 w-full resize-y rounded-lg border border-white/15 bg-black/40 px-3 py-2 text-sm text-[#F8FAFC] outline-none focus:border-white/40"
            />
          ) : (
            <p className="mt-1 text-sm text-[#D1D5DB]">{work.expectations.outcome}</p>
          )}
          {work.expectations.end_status && (
            <p className="mt-1 text-xs text-[#9CA3AF]">
              End status (declared by the flow): <span className="text-[#D1D5DB]">{work.expectations.end_status}</span>
            </p>
          )}
          <p className="mt-3 text-xs font-medium uppercase tracking-wide text-[#9CA3AF]">Bug guards</p>
          {editing && onChangeExpectations ? (
            <textarea
              value={work.expectations.bug_guards.join("\n")}
              onChange={(e) =>
                onChangeExpectations({
                  ...work.expectations!,
                  bug_guards: e.target.value.split("\n").map((g) => g.trim()).filter(Boolean),
                })
              }
              disabled={!!busy}
              rows={3}
              placeholder="One check per line"
              className="mt-1.5 w-full resize-y rounded-lg border border-white/15 bg-black/40 px-3 py-2 text-sm text-[#F8FAFC] outline-none focus:border-white/40"
            />
          ) : work.expectations.bug_guards.length > 0 ? (
            <ul className="mt-1 list-disc space-y-0.5 pl-5 text-sm text-[#D1D5DB]">
              {work.expectations.bug_guards.map((g, i) => <li key={i}>{g}</li>)}
            </ul>
          ) : (
            <p className="mt-1 text-xs text-[#9CA3AF]">none</p>
          )}
        </div>
      )}

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <button onClick={onToggleEdit} disabled={!!busy} className={pillButton}>
          {editing ? (
            <CheckCircle2 className="h-3.5 w-3.5" strokeWidth={1.5} />
          ) : (
            <Pencil className="h-3.5 w-3.5" strokeWidth={1.5} />
          )}
          {editing ? "Done Editing" : "Edit"}
        </button>
        <button onClick={onGenerate} disabled={!!busy} className={pillButton}>
          {generating ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" strokeWidth={1.5} />
          ) : (
            <RotateCcw className="h-3.5 w-3.5" strokeWidth={1.5} />
          )}
          {generating ? "Regenerating…" : "Regenerate"}
        </button>
        {work.saved && (
          <button
            onClick={onAskDelete}
            disabled={!!busy}
            className="inline-flex items-center gap-1.5 rounded-full border border-rose-400/25 px-3 py-1.5 text-xs font-medium text-rose-300 transition hover:border-rose-400/50 hover:bg-rose-400/10 disabled:opacity-50"
          >
            {busy === "deleting" ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" strokeWidth={1.5} />
            ) : (
              <Trash2 className="h-3.5 w-3.5" strokeWidth={1.5} />
            )}
            Delete
          </button>
        )}
        <div className="ml-auto flex flex-wrap items-center gap-2">
          {work.dirty && (
            <button
              onClick={onSave}
              disabled={!!busy}
              className="inline-flex items-center gap-2 rounded-lg border border-white/20 bg-white/8 px-4 py-2 text-sm font-medium text-[#F8FAFC] transition hover:bg-white/15 disabled:opacity-50"
            >
              {busy === "saving" ? (
                <Loader2 className="h-4 w-4 animate-spin" strokeWidth={1.5} />
              ) : (
                <Save className="h-4 w-4" strokeWidth={1.5} />
              )}
              {busy === "saving" ? "Saving…" : "Save"}
            </button>
          )}
          {work.saved && runnable && (
            <button
              onClick={onRun}
              disabled={!!busy || work.dirty}
              title={work.dirty ? "Save your changes before running" : undefined}
              className="inline-flex items-center gap-2 rounded-lg border border-emerald-400/40 bg-emerald-400/15 px-4 py-2 text-sm font-medium text-emerald-200 transition hover:bg-emerald-400/25 disabled:opacity-50"
            >
              {busy === "running" ? (
                <Loader2 className="h-4 w-4 animate-spin" strokeWidth={1.5} />
              ) : (
                <PlayCircle className="h-4 w-4" strokeWidth={1.5} />
              )}
              {busy === "running" ? "Running…" : "Run"}
            </button>
          )}
        </div>
      </div>
      <p className="mt-2 text-xs text-[#9CA3AF]">
        {!runnable
          ? "Runs aren't available for interrupt scenarios yet."
          : !work.saved
          ? "Save this script to run it."
          : work.dirty
            ? "Save your changes to run the updated script."
            : "Run plays the saved script against the Voice Agent through the existing voice pipeline."}
      </p>

      {confirmingDelete && (
        <div className="mt-3 rounded-lg border border-rose-400/30 bg-rose-400/10 px-3 py-3 text-sm text-rose-200">
          <p>
            {work.saved
              ? "Delete this saved scenario and its script? The flow and its nodes and edges are not affected — the path stays in the preview."
              : "Discard this generated draft? It hasn't been saved."}
          </p>
          <div className="mt-2 flex gap-2">
            <button
              onClick={onDelete}
              className="rounded-full border border-rose-400/40 px-3 py-1 text-xs font-medium text-rose-200 transition hover:bg-rose-400/20"
            >
              {work.saved ? "Delete" : "Discard Draft"}
            </button>
            <button
              onClick={onCancelDelete}
              className="rounded-full border border-white/15 px-3 py-1 text-xs text-[#9CA3AF] transition hover:border-white/30 hover:text-white"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {errors && <ScriptErrors title={FAILURE_TITLES[errors.action]} errors={errors.errors} />}

      {run && !run.report && (
        <div className="mt-4 flex items-center gap-2 text-sm text-[#9CA3AF]">
          <Loader2 className="h-4 w-4 animate-spin" strokeWidth={1.5} />
          {run.status === "queued" || run.status === "running"
            ? "Running against the Voice Agent through AgentShield's existing voice pipeline…"
            : "Starting…"}
        </div>
      )}
      {run?.report && <FlowRunResult report={run.report} />}
    </div>
  );
}

// The run's result block, reusing the dashboard's own
// transcript view.
function FlowRunResult({ report }: { report: Report }) {
  const conv = report.conversations[0];
  if (!conv) {
    return <p className="mt-4 text-sm text-[#9CA3AF]">The run finished without a conversation to show.</p>;
  }
  return (
    <div className="mt-6 rounded-lg border border-white/12 bg-black/30 p-4">
      <div className="flex items-center justify-between gap-2">
        <h4 className="font-medium text-[#F8FAFC]">Result</h4>
        <span
          className={`rounded-full border px-2.5 py-0.5 text-xs font-medium uppercase tracking-wide ${
            conv.verdict === "pass" ? "border-emerald-400/40 text-emerald-300" : "border-rose-400/40 text-rose-300"
          }`}
        >
          {conv.verdict ?? "unjudged"}
          {conv.severity ? ` · ${conv.severity}` : ""}
        </span>
      </div>
      {typeof report.reliability_score === "number" && (
        <p className="mt-1 text-xs text-[#9CA3AF]">Reliability score: {report.reliability_score}</p>
      )}
      {conv.explanation && (
        <p className="mt-3 text-sm text-[#D1D5DB]">
          <span className="font-medium text-[#F8FAFC]">Why: </span>
          {conv.explanation}
        </p>
      )}
      {conv.suggested_fix && (
        <p className="mt-2 text-sm text-[#D1D5DB]">
          <span className="font-medium text-[#F8FAFC]">Suggested fix: </span>
          {conv.suggested_fix}
        </p>
      )}
      <TranscriptDetails messages={conv.messages} label="Transcript" />
    </div>
  );
}
