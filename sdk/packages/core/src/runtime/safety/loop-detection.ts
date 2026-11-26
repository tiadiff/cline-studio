/**
 * Repeated tool-call loop detection.
 *
 * @see PLAN.md §3.1 — helpers moved from `packages/agents/src/context/loop-detection.ts`.
 * @see PLAN.md §3.2.3 — public surface of `LoopDetectionTracker`.
 *
 * The pure helpers (`createLoopDetectionState`, `resetLoopDetectionState`,
 * `toolCallSignature`, `checkRepeatedToolCall`) are ported verbatim. The
 * `LoopDetectionTracker` class is a thin wrapper that owns a
 * `LoopDetectionState` and exposes the `inspect()` / `reset()` surface that
 * `SessionRuntime` installs as a `beforeTool` hook per §3.2.3.
 */

import type { LoopDetectionConfig } from "@cline/shared";

// =============================================================================
// Pure helpers (verbatim port)
// =============================================================================

export interface LoopDetectionRecentCall {
	toolName: string;
	signature: string;
	normalizedSignature: string;
}

export interface LoopDetectionState {
	lastToolName: string;
	lastToolSignature: string;
	lastNormalizedSignature?: string;
	consecutiveIdenticalCount: number;
	recentCalls?: LoopDetectionRecentCall[];
}

export function createLoopDetectionState(): LoopDetectionState {
	return {
		lastToolName: "",
		lastToolSignature: "",
		lastNormalizedSignature: "",
		consecutiveIdenticalCount: 0,
		recentCalls: [],
	};
}

export function resetLoopDetectionState(state: LoopDetectionState): void {
	state.lastToolName = "";
	state.lastToolSignature = "";
	state.lastNormalizedSignature = "";
	state.consecutiveIdenticalCount = 0;
	state.recentCalls = [];
}

function sortKeys(value: unknown): unknown {
	if (value == null || typeof value !== "object") return value;
	if (Array.isArray(value)) return value.map(sortKeys);
	const sorted: Record<string, unknown> = {};
	for (const key of Object.keys(value as Record<string, unknown>).sort()) {
		sorted[key] = sortKeys((value as Record<string, unknown>)[key]);
	}
	return sorted;
}

export function toolCallSignature(input: unknown): string {
	if (input == null) return "null";
	if (typeof input === "string") return input;
	if (typeof input !== "object") return String(input);
	try {
		return JSON.stringify(sortKeys(input));
	} catch {
		return String(input);
	}
}

function normalizeShellCommandForLoopCheck(cmd: string): string {
	return cmd
		.trim()
		.replace(/\s+(?:2>&1|2>\s*\$null|>\s*\$null|\*>\s*\$null|2>\s*\/dev\/null|>\s*\/dev\/null)\s*$/gi, "")
		.replace(/\bchocolety\b/gi, "chocolatey")
		.replace(/\bchoco\.exe\b/gi, "chocolatey")
		.replace(/\bchoco\b/gi, "chocolatey")
		.replace(/\bwhere\.exe\b/gi, "where")
		.replace(/\s+/g, " ")
		.toLowerCase();
}

export function normalizedToolCallSignature(
	toolName: string,
	input: unknown,
	rawSignature?: string,
): string {
	const sig = rawSignature ?? toolCallSignature(input);
	if (
		toolName === "run_commands" &&
		input !== null &&
		typeof input === "object" &&
		Array.isArray((input as { commands?: unknown[] }).commands)
	) {
		const cmds = (input as { commands: unknown[] }).commands.map((c) => {
			if (typeof c === "string") {
				return normalizeShellCommandForLoopCheck(c);
			}
			if (c && typeof c === "object" && typeof (c as { command?: unknown }).command === "string") {
				const args = Array.isArray((c as { args?: unknown[] }).args)
					? (c as { args: unknown[] }).args.map(String).join(" ")
					: "";
				return normalizeShellCommandForLoopCheck(`${(c as { command: string }).command} ${args}`);
			}
			return String(c);
		});
		return JSON.stringify(cmds);
	}
	return sig;
}

export interface LoopCheckResult {
	softWarning: boolean;
	hardEscalation: boolean;
	effectiveCount?: number;
}

const RECENT_CALL_WINDOW = 8;

export function checkRepeatedToolCall(
	state: LoopDetectionState,
	toolName: string,
	signature: string,
	config: LoopDetectionConfig,
	normalizedSignature?: string,
): LoopCheckResult {
	const normSig = normalizedSignature ?? signature;
	const isConsecutiveMatch =
		toolName === state.lastToolName &&
		(signature === state.lastToolSignature ||
			(Boolean(state.lastNormalizedSignature) &&
				normSig === state.lastNormalizedSignature));

	if (isConsecutiveMatch) {
		state.consecutiveIdenticalCount++;
	} else {
		state.consecutiveIdenticalCount = 1;
	}
	state.lastToolName = toolName;
	state.lastToolSignature = signature;
	state.lastNormalizedSignature = normSig;

	if (!state.recentCalls) {
		state.recentCalls = [];
	}
	state.recentCalls.push({
		toolName,
		signature,
		normalizedSignature: normSig,
	});
	if (state.recentCalls.length > RECENT_CALL_WINDOW) {
		state.recentCalls.shift();
	}

	const windowMatches = state.recentCalls.filter(
		(c) =>
			c.toolName === toolName &&
			(c.signature === signature || c.normalizedSignature === normSig),
	).length;
	const effectiveCount = Math.max(
		state.consecutiveIdenticalCount,
		windowMatches,
	);

	return {
		softWarning:
			state.consecutiveIdenticalCount === config.softThreshold ||
			(effectiveCount === config.softThreshold &&
				state.consecutiveIdenticalCount < config.softThreshold),
		hardEscalation: effectiveCount >= config.hardThreshold,
		effectiveCount,
	};
}

// =============================================================================
// Class wrapper (new — per PLAN.md §3.2.3)
// =============================================================================

/**
 * Verdict returned by {@link LoopDetectionTracker.inspect}.
 *
 * - `"ok"`   — no repeated call detected.
 * - `"soft"` — soft-warning threshold reached; SessionRuntime may surface a
 *              recovery notice but should not block the call.
 * - `"hard"` — hard-escalation threshold reached; SessionRuntime should
 *              stop the run with the provided `message`.
 */
export interface LoopDetectionVerdict {
	kind: "ok" | "soft" | "hard";
	message?: string;
}

/** Minimal call shape the tracker needs; matches `AgentToolCallPart` subset. */
export interface LoopDetectionCall {
	name: string;
	input: unknown;
}

const DEFAULT_CONFIG: LoopDetectionConfig = {
	softThreshold: 3,
	hardThreshold: 5,
};

/**
 * Per-session repeated-tool-call detector.
 *
 * `SessionRuntime` owns the instance and installs a `beforeTool` hook
 * (see `AgentRuntimeHooks.beforeTool`) that calls `inspect()` to decide
 * whether to return `{ skip, stop, reason }`.
 */
export class LoopDetectionTracker {
	private readonly config: LoopDetectionConfig;
	private readonly state: LoopDetectionState = createLoopDetectionState();

	constructor(config?: Partial<LoopDetectionConfig>) {
		this.config = {
			softThreshold: config?.softThreshold ?? DEFAULT_CONFIG.softThreshold,
			hardThreshold: config?.hardThreshold ?? DEFAULT_CONFIG.hardThreshold,
		};
	}

	inspect(call: LoopDetectionCall): LoopDetectionVerdict {
		const signature = toolCallSignature(call.input);
		const normSig = normalizedToolCallSignature(
			call.name,
			call.input,
			signature,
		);
		const result = checkRepeatedToolCall(
			this.state,
			call.name,
			signature,
			this.config,
			normSig,
		);
		const count =
			result.effectiveCount ?? this.state.consecutiveIdenticalCount;
		if (result.hardEscalation) {
			return {
				kind: "hard",
				message: `Detected ${count} consecutive identical calls to \`${call.name}\`; stopping to avoid a loop.`,
			};
		}
		if (result.softWarning) {
			return {
				kind: "soft",
				message: `Detected ${count} consecutive identical calls to \`${call.name}\`; consider trying a different approach. Do NOT repeat this command or tool call again.`,
			};
		}
		return { kind: "ok" };
	}

	reset(): void {
		resetLoopDetectionState(this.state);
	}
}
