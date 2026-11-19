"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { ArrowLeft, Check, ListChecks, Loader2, PlayCircle, ShieldCheck } from "lucide-react";
import { getInventory, type Inventory } from "../lib/api";

// One selectable customer-agent combination. The same agent under two customers is
// two separate rows, so the row id has to carry both.
type Combo = {
  id: string;
  customer: string;
  agentKey: string;
  agentName: string;
  simulated: boolean;
  firstOfCustomer: boolean;
  customerAgentId: number | null;
  agentId: number | null;
};

// Only agents of the modality being tested belong on this page — a chat agent's HTTP
// endpoint can't hold a voice conversation and vice versa, so cross-modality entries are
// filtered out here (before indexing for `firstOfCustomer`, so the customer-group divider
// still lands on the first row actually shown).
function toCombos(inventory: Inventory, modality: "chat" | "voice"): Combo[] {
  return inventory.customers.flatMap((c) => {
    const agents = c.agents.filter((a) => a.modality === modality);
    return agents.map((a, i) => ({
      id: `${c.name}::${a.key}`,
      customer: c.name,
      agentKey: a.key,
      agentName: a.name,
      simulated: !!a.simulated,
      firstOfCustomer: i === 0,
      customerAgentId: a.customer_agent_id,
      agentId: a.agent_id,
    }));
  });
}

export default function ExistingAgent() {
  const [combos, setCombos] = useState<Combo[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Carried in from /test via /start?modality=... — defaults to "chat" so a direct
  // link to this page (no modality chosen) behaves exactly as it always has.
  const [modality, setModality] = useState<"chat" | "voice">("chat");
  const router = useRouter();

  useEffect(() => {
    const q = new URLSearchParams(window.location.search).get("modality");
    const mod = q === "voice" ? "voice" : "chat";
    setModality(mod);
    getInventory()
      .then((inv) => setCombos(toCombos(inv, mod)))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, []);

  const toggle = (id: string) =>
    setSelected((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));

  // Hand every chosen combination to the dashboard, which loads (or generates) each
  // one's test cases and shows them in the same Review Test Cases UI — one tab per agent.
  const run = (chosen: Combo[]) => {
    const registered = chosen.filter((c) => c.customerAgentId !== null);
    if (!registered.length) {
      setError("None of those combinations are registered in the database yet.");
      return;
    }
    if (registered.length < chosen.length) {
      setError(
        `${chosen.length - registered.length} of the selected combinations aren't registered yet — running the rest.`
      );
    }
    const targets = registered.map((c) => ({
      ca: c.customerAgentId,
      agent: c.agentId,
      customer: c.customer,
      agentName: c.agentName,
    }));
    const first = registered[0];
    const params = new URLSearchParams({
      // `targets` drives the run; ca/agent stay for the first agent so a hand-written
      // single-agent link still works.
      targets: JSON.stringify(targets),
      ca: String(first.customerAgentId),
      agent: String(first.agentId),
      customer: first.customer,
      agentName: first.agentName,
      modality,
    });
    router.push(`/dashboard?${params.toString()}`);
  };

  const runSelected = () => run(combos.filter((c) => selected.includes(c.id)));

  const runAll = () => {
    setSelected(combos.map((c) => c.id));
    run(combos);
  };

  return (
    <main className="relative min-h-screen overflow-hidden bg-[#0B0B0F]">
      <header className="sticky top-0 z-50 border-b border-white/12 bg-[#0B0B0F]/70 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <Link href="/" className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/20 bg-white/4 text-slate-200">
              <ShieldCheck className="h-5 w-5" strokeWidth={1.5} />
            </div>
            <span className="font-logo text-lg font-extrabold tracking-tight text-[#F8FAFC]">AgentShield</span>
          </Link>
          <Link
            href={`/start?modality=${modality}`}
            className="flex items-center gap-2 rounded-full border border-white/12 bg-white/2 px-5 py-2.5 text-sm font-medium text-[#F8FAFC] backdrop-blur-md transition-all duration-300 hover:border-white/[0.16] hover:bg-white/4"
          >
            <ArrowLeft className="h-4 w-4" strokeWidth={1.5} />
            Back
          </Link>
        </div>
      </header>

      <section className="relative mx-auto max-w-4xl px-6 pb-24 pt-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        >
          <span className="inline-flex items-center gap-2 rounded-full border border-white/12 bg-white/2 px-4 py-1.5 text-xs font-medium text-[#9CA3AF] backdrop-blur-md">
            Testing: {modality === "voice" ? "Voice Agent" : "Chat Agent"}
          </span>
          <div className="mt-4 flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-lg border border-white/20 bg-white/4 text-slate-200">
              <ListChecks className="h-5 w-5" strokeWidth={1.5} />
            </div>
            <h1 className="font-heading text-2xl font-medium tracking-tight text-[#F8FAFC]">
              Existing Agent Testing
            </h1>
          </div>
          <p className="mt-3 text-sm text-[#9CA3AF]">
            Pick one or more customer-agent combinations to test. The same agent onboarded for two
            customers is tested separately for each.
          </p>

          {error && (
            <div className="mt-6 rounded-lg border border-[#F87171]/40 bg-[#F87171]/10 px-4 py-3 text-sm text-[#F87171]">
              {error}
            </div>
          )}

          <div className="mt-8 overflow-hidden rounded-xl border border-white/12 bg-white/2 backdrop-blur-md">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-white/12 text-xs font-medium text-[#9CA3AF]">
                  <th className="w-20 px-6 py-4">Select</th>
                  <th className="px-6 py-4">Customer</th>
                  <th className="px-6 py-4">Agent</th>
                </tr>
              </thead>
              <tbody>
                {loading && (
                  <tr>
                    <td colSpan={3} className="px-6 py-8 text-center text-[#9CA3AF]">
                      <Loader2 className="mx-auto h-5 w-5 animate-spin" strokeWidth={1.5} />
                    </td>
                  </tr>
                )}
                {!loading && combos.length === 0 && !error && (
                  <tr>
                    <td colSpan={3} className="px-6 py-8 text-center text-[#9CA3AF]">
                      No onboarded {modality} agents found. Try{" "}
                      <Link href={`/start?modality=${modality}`} className="underline hover:text-[#F8FAFC]">
                        connecting a new one
                      </Link>
                      .
                    </td>
                  </tr>
                )}
                {combos.map((combo) => (
                  <tr
                    key={combo.id}
                    onClick={() => toggle(combo.id)}
                    className={`cursor-pointer border-b border-white/[0.06] transition-colors duration-200 last:border-0 hover:bg-white/4 ${
                      combo.firstOfCustomer ? "border-t border-t-white/12" : ""
                    }`}
                  >
                    <td className="px-6 py-4">
                      <span
                        className={`flex h-5 w-5 items-center justify-center rounded border transition-colors duration-200 ${
                          selected.includes(combo.id)
                            ? "border-white/50 bg-white/20 text-[#F8FAFC]"
                            : "border-white/20 bg-white/4"
                        }`}
                      >
                        {selected.includes(combo.id) && <Check className="h-3.5 w-3.5" strokeWidth={2.5} />}
                      </span>
                      <input
                        type="checkbox"
                        className="sr-only"
                        checked={selected.includes(combo.id)}
                        onChange={() => toggle(combo.id)}
                        aria-label={`${combo.customer} — ${combo.agentName}`}
                      />
                    </td>
                    <td className="px-6 py-4 font-medium text-[#F8FAFC]">
                      {combo.customer}
                    </td>
                    <td className="px-6 py-4 text-[#9CA3AF]">
                      {combo.agentName}
                      {combo.simulated && (
                        <span className="ml-2 rounded-full border border-[#FBBF24]/30 bg-[#FBBF24]/10 px-2 py-0.5 text-[10px] font-medium text-[#FBBF24]">
                          Dry run
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-6 flex flex-col items-center justify-between gap-4 sm:flex-row">
            <span className="text-xs text-[#9CA3AF]">
              {selected.length} of {combos.length} selected
            </span>
            <div className="flex flex-col gap-3 sm:flex-row">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                transition={{ duration: 0.3 }}
                onClick={runAll}
                disabled={combos.length === 0}
                className="flex items-center justify-center gap-2 rounded-full border border-white/12 bg-white/2 px-6 py-3 text-sm font-medium text-[#F8FAFC] backdrop-blur-md transition-all duration-300 hover:border-white/[0.16] hover:bg-white/4 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Run All Agents
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                transition={{ duration: 0.3 }}
                onClick={runSelected}
                disabled={selected.length === 0}
                className="flex items-center justify-center gap-2 rounded-full border border-white/20 bg-white/4 px-6 py-3 text-sm font-medium text-[#F8FAFC] backdrop-blur-md transition-all duration-300 hover:border-white/40 hover:bg-white/[0.12] hover:shadow-[0_0_32px_rgba(255,255,255,0.2)] disabled:cursor-not-allowed disabled:border-white/10 disabled:bg-white/2 disabled:opacity-40 disabled:shadow-none"
              >
                <PlayCircle className="h-4 w-4" strokeWidth={1.5} />
                Run Selected Agents
              </motion.button>
            </div>
          </div>

          {selected.length > 1 && (
            <p className="mt-4 text-xs text-[#9CA3AF]">
              {selected.length} agents selected — they are tested in parallel, each with its own
              test cases and its own reliability score.
            </p>
          )}

        </motion.div>
      </section>
    </main>
  );
}
