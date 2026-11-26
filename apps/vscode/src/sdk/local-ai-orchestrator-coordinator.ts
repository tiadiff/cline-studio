import { execFile } from "node:child_process"
import fs from "node:fs"
import path from "node:path"
import { promisify } from "node:util"
import { resolveSessionDataDir } from "@cline/shared/storage"
import type { ApiConfiguration } from "@shared/api"
import type {
	ClineAsk,
	ClineMessage,
	ClinePlanModeResponse,
	OrchestratorLastEvaluation,
} from "@shared/ExtensionMessage"
import type { Mode } from "@shared/storage/types"
import { Logger } from "@/shared/services/Logger"

const execFileAsync = promisify(execFile)

export type OrchestratorStatus = "idle" | "evaluating_prompt" | "evaluating_plan" | "evaluating_act"

export interface OrchestratorStateSnapshot {
	enabled: boolean
	modelId: string
	planModelId?: string
	actModelId?: string
	status: OrchestratorStatus
	phaseDetail?: string
	startedAt?: number
	revisionCount: number
	maxRevisions: number
	lastEvaluation?: OrchestratorLastEvaluation
}

export interface OrchestratorPromptEvaluation {
	isClear: boolean
	qualityScore: number
	objective: string
	technicalContext: string
	requirements: string[]
	risksAndEdgeCases: string[]
	acceptanceCriteria: string[]
	planningGuidance: string
}

export interface OrchestratorPlanDecision {
	valid: boolean
	score: number
	reasoning: string
	strengths: string[]
	missingPoints: string[]
	suggestedModifications: string[]
	isAcceptableFallback?: boolean
	requiresActExecution?: boolean
}

export interface OrchestratorActReview {
	completed: boolean
	score: number
	summary: string
	verifiedPoints: string[]
	issuesOrPending: string[]
	proposedNextAction?: string
}

type OrchestratorReasoningEffort = "low" | "medium" | "high"

interface LlmCallMetrics {
	content: string
	thinkingText: string
	reasoningEffort: OrchestratorReasoningEffort
	vramCheckMs: number
	vramUnloadBeforeMs: number
	vramUnloadAfterMs: number
	previouslyLoadedModels: string[]
	ttftMs: number
	generationMs: number
	totalLlmMs: number
	estimatedPromptTokens: number
	estimatedOutputTokens: number
	estimatedThinkingTokens: number
	tokensPerSecond: number
	finishReason?: string
}

export interface LocalAiOrchestratorOptions {
	getApiConfiguration: () => ApiConfiguration
	getCurrentMode: () => Mode
	setModeToPlan: () => Promise<void>
	getClineMessages: () => ClineMessage[]
	getSessionId: () => string
	isSessionRunning: () => boolean
	getWorkspaceRoot?: () => Promise<string>
	emitMessage: (message: ClineMessage) => void
	postStateToWebview: () => Promise<void>
	triggerActMode: () => Promise<void>
	sendPlanRevisionPrompt: (prompt: string) => Promise<void>
	onStateChanged?: (state: OrchestratorStateSnapshot) => void
	gatewayBaseUrl?: string
	maxPlanRevisions?: number
	initialEnabled?: boolean
}

export const ORCHESTRATOR_MODEL_PRIORITY: readonly string[] = [
	"TwIL-8B-Reasoning-96K-80toks",
	"Qwen3.6-27B-CoderX-48K-115toks",
	"Qwen3.6-35B-Coding-Avanzato-60toks",
	"Inkling-2-Mini-26B-32K-50toks",
	"Ternary-Bonsai-2-27B-48K-45toks",
	"Qwen3.6-27B-CoderX-32K-125toks",
	"Gemma4-12B-Agentic-47toks",
	"Nemotron-4B-Coding-Veloce-72toks",
	"Qwen3.8-27B-Ragionamento-6toks",
]

export const ORCHESTRATOR_REVISION_PREFIX = "[ORCHESTRATOR PLAN FEEDBACK]"

const USER_BLOCKING_ASKS: ReadonlySet<ClineAsk> = new Set<ClineAsk>([
	"tool",
	"command",
	"use_mcp_server",
	"browser_action_launch",
	"use_subagents",
	"followup",
	"api_req_failed",
	"mistake_limit_reached",
])

function formatClock(ts = Date.now()): string {
	const d = new Date(ts)
	return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}.${String(d.getMilliseconds()).padStart(3, "0")}`
}

function formatDuration(ms: number): string {
	return ms < 1000 ? `${Math.max(0, Math.round(ms))}ms` : `${(ms / 1000).toFixed(2)}s`
}

function formatEffortLabel(effort: OrchestratorReasoningEffort): string {
	return effort === "high" ? "High" : effort === "medium" ? "Medium" : "Low"
}

function buildScoreMeter(score: number): string {
	const clamped = Math.max(1, Math.min(10, Math.round(score)))
	const dot = clamped >= 8 ? "🟢" : clamped >= 6 ? "🟡" : "🔴"
	return `${dot.repeat(clamped)}${"⚪".repeat(10 - clamped)} **${clamped}/10**`
}

function buildSupervisorCallout(params: {
	verdictBadge: string
	score: number
	evaluationText: string
	extraSubtitle?: string
}): string {
	const { verdictBadge, score, evaluationText, extraSubtitle } = params
	const cleanEval = evaluationText
		.trim()
		.split("\n")
		.map((line) => `> ${line.trim()}`)
		.join("\n")

	return [
		`## 🛡️ Supervisor Evaluation`,
		`> ### ${verdictBadge}`,
		`> **Score:** ${buildScoreMeter(score)}${extraSubtitle ? ` • ${extraSubtitle}` : ""}`,
		`>`,
		cleanEval,
	].join("\n")
}

function buildLiveCardMarkdown(params: {
	title: string
	elapsedMs: number
	pipelineLine: string
	metaLine?: string
	liveActivity: string
	traceLogs: string[]
	liveThinking?: string
	liveOutput?: string
}): string {
	const { title, elapsedMs, pipelineLine, metaLine, liveActivity, traceLogs, liveThinking, liveOutput } = params
	const lines = [
		`## 🛡️ AI Supervisor — ${title} • \`${formatDuration(elapsedMs)}\``,
		`> ${liveActivity.replace(/\n/g, "\n> ")}`,
		"",
		pipelineLine,
		metaLine ?? "",
	].filter(Boolean)

	if (liveThinking) {
		const tailThink = liveThinking.length > 700 ? `...${liveThinking.slice(-700)}` : liveThinking
		lines.push("", `**Live Reasoning:**`, `> ${tailThink.replace(/\n/g, "\n> ")}`)
	}
	if (liveOutput) {
		const tailOut = liveOutput.length > 500 ? `...${liveOutput.slice(-500)}` : liveOutput
		lines.push("", `**Partial JSON Output:**`, `\`\`\`json\n${tailOut}\n\`\`\``)
	}
	lines.push("", `---`, `**Live Log:**`, ...traceLogs.slice(-5).map((l) => `- ${l}`))
	return lines.join("\n")
}

function formatTelemetryBar(
	orchestratorModel: string,
	totalMs: number,
	metrics: LlmCallMetrics,
	extraSegment?: string,
): string {
	return [
		`**Supervisor:** \`${orchestratorModel}\` *(Thinking: ${formatEffortLabel(metrics.reasoningEffort)})*`,
		`**Duration:** \`${formatDuration(totalMs)}\``,
		extraSegment ?? "",
		`**VRAM Swap:** \`${formatDuration(metrics.vramUnloadBeforeMs + metrics.vramUnloadAfterMs)}\``,
		`**TTFT:** \`${formatDuration(metrics.ttftMs)}\``,
		`**Gen:** \`${formatDuration(metrics.generationMs)}\` (@ **\`${metrics.tokensPerSecond.toFixed(1)} tok/s\`**)`,
	]
		.filter(Boolean)
		.join(" • ")
}

function formatThinkingPreview(thinkingText: string, estimatedTokens: number, maxChars = 1100): string {
	if (!thinkingText) {
		return ""
	}
	const trimmed = thinkingText.trim()
	const preview =
		trimmed.length > maxChars
			? `${trimmed.slice(0, Math.floor(maxChars * 0.45)).trim()}\n...\n${trimmed.slice(-Math.floor(maxChars * 0.45)).trim()}`
			: trimmed
	return `**Supervisor Internal Reasoning (~${estimatedTokens} tok):**\n\`\`\`text\n${preview}\n\`\`\``
}

function sendWindowsNotification(title: string, message: string): void {
	if (process.platform !== "win32") {
		return
	}
	try {
		const clean = (s: string, max: number) =>
			s
				.replace(/[`*_#>]/g, "")
				.replace(/\s+/g, " ")
				.trim()
				.slice(0, max)
				.replace(/&/g, "&amp;")
				.replace(/</g, "&lt;")
				.replace(/>/g, "&gt;")
				.replace(/"/g, "&quot;")
				.replace(/'/g, "&apos;")

		const psScript = [
			"[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null",
			"$doc = [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime]::new()",
			`$xml = '<toast><visual><binding template="ToastGeneric"><text>${clean(title, 80)}</text><text>${clean(message, 200)}</text></binding></visual></toast>'`,
			"$doc.LoadXml($xml)",
			"$toast = [Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType = WindowsRuntime]::new($doc)",
			"$appId = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe'",
			"[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($appId).Show($toast)",
		].join("; ")

		execFile(
			"powershell.exe",
			[
				"-NoProfile",
				"-NonInteractive",
				"-ExecutionPolicy",
				"Bypass",
				"-EncodedCommand",
				Buffer.from(psScript, "utf16le").toString("base64"),
			],
			{ timeout: 4000, windowsHide: true },
			() => { },
		)
	} catch (err) {
		Logger.warn("[LocalAiOrchestrator] Failed to show Windows notification:", err)
	}
}

function estimateTokens(text: string): number {
	return text ? Math.max(1, Math.ceil(text.length / 3.6)) : 0
}

function extractThinkingBlocks(raw: string): string {
	const matches: string[] = []
	for (const regex of [
		/<think>([\s\S]*?)(?:<\/think>|$)/gi,
		/<\|channel\|?>thought([\s\S]*?)(?:<\|channel\|?>|<\|end\|?>|$)/gi,
	]) {
		let m = regex.exec(raw)
		while (m !== null) {
			if (m[1]?.trim()) {
				matches.push(m[1].trim())
			}
			m = regex.exec(raw)
		}
	}
	return matches.join("\n").trim()
}

function stripThinkingBlocks(raw: string): string {
	return raw
		.replace(/<think>[\s\S]*?<\/think>/gi, "")
		.replace(/<think>[\s\S]*$/gi, "")
		.replace(/<\|channel\|?>thought[\s\S]*?(?:<\|channel\|?>|<\|end\|?>|$)/gi, "")
		.trim()
}

function tryParseJsonRecord(jsonSlice: string): Record<string, unknown> | undefined {
	for (const candidate of [jsonSlice, jsonSlice.replace(/,\s*([}\]])/g, "$1")]) {
		try {
			const parsed = JSON.parse(candidate)
			if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
				return parsed as Record<string, unknown>
			}
		} catch {
			// Ignore invalid JSON candidate
		}
	}
	return undefined
}

function extractBalancedJsonObjects(text: string): string[] {
	const results: string[] = []
	let startIdx = -1
	let depth = 0
	let inString = false
	let isEscaped = false

	for (let i = 0; i < text.length; i++) {
		const ch = text[i]
		if (inString) {
			if (isEscaped) {
				isEscaped = false
			} else if (ch === "\\") {
				isEscaped = true
			} else if (ch === '"') {
				inString = false
			}
			continue
		}
		if (ch === '"') {
			inString = true
			continue
		}
		if (ch === "{") {
			if (depth === 0) {
				startIdx = i
			}
			depth++
		} else if (ch === "}" && depth > 0) {
			depth--
			if (depth === 0 && startIdx !== -1) {
				results.push(text.slice(startIdx, i + 1))
				startIdx = -1
			}
		}
	}
	return results
}

function extractFirstJsonObject(raw: string): Record<string, unknown> | undefined {
	const cleaned = stripThinkingBlocks(raw)
	for (const source of [cleaned, raw]) {
		if (!source) {
			continue
		}
		const fencedRegex = /```(?:json)?\s*([\s\S]*?)```/gi
		let match = fencedRegex.exec(source)
		while (match !== null) {
			for (const slice of extractBalancedJsonObjects(match[1].trim())) {
				const parsed = tryParseJsonRecord(slice)
				if (parsed) {
					return parsed
				}
			}
			match = fencedRegex.exec(source)
		}
		for (const slice of extractBalancedJsonObjects(source)) {
			const parsed = tryParseJsonRecord(slice)
			if (parsed) {
				return parsed
			}
		}
		const firstBrace = source.indexOf("{")
		const lastBrace = source.lastIndexOf("}")
		if (firstBrace !== -1 && lastBrace > firstBrace) {
			const parsed = tryParseJsonRecord(source.slice(firstBrace, lastBrace + 1))
			if (parsed) {
				return parsed
			}
		}
	}
	return undefined
}

function synthesizeEvaluationFromReasoning(
	rawPrompt: string,
	thinkingText: string,
	contentText: string,
	workspaceSnapshot: string,
): OrchestratorPromptEvaluation {
	const combined = (contentText || thinkingText || "").trim()
	const sentences = combined
		.split(/[\n.]+/)
		.map((s) => s.replace(/^[-*•\d.)+\s]+/, "").trim())
		.filter((s) => s.length > 20 && s.toLowerCase() !== rawPrompt.toLowerCase())

	return {
		isClear: rawPrompt.split(/\s+/).length >= 5,
		qualityScore: rawPrompt.split(/\s+/).length >= 12 ? 8 : 6,
		objective:
			sentences[0] ||
			`Analyze the request in detail ("${rawPrompt}"), verify the actual state of the files in the workspace, and formulate a complete operational plan.`,
		technicalContext: workspaceSnapshot
			? `Workspace state detected by the Orchestrator:\n${workspaceSnapshot.slice(0, 600)}`
			: "Preliminary analysis performed on the user prompt.",
		requirements:
			sentences.slice(1, 5).length >= 2
				? sentences.slice(1, 5)
				: [
					`Analyze the workspace context to comprehensively address: "${rawPrompt}"`,
					"Inspect the actual state of files, dependencies, or Git repository before proposing changes",
					"Define concrete, ordered, and verifiable steps",
				],
		risksAndEdgeCases: [
			"Avoid unverified assumptions about file or Git state without prior inspection.",
		],
		acceptanceCriteria: [
			"The Plan must directly and specifically address the request based on real workspace data.",
			"Every step proposed in the Plan must indicate concrete files, commands, or verifications.",
		],
		planningGuidance:
			"Use read-only tools in Plan Mode if necessary to gather missing data, then present a structured, clear, and immediately actionable plan.",
	}
}

function getModelFamilyPrefix(modelId: string): string {
	const normalized = modelId.trim().toLowerCase()
	return normalized.startsWith("qwen3.6-27b-coderx") ? "qwen3.6-27b-coderx" : normalized
}

function readLocalAiDisabledModels(): Set<string> {
	const disabled = new Set<string>()
	const possiblePaths = [
		"C:\\LocalAI\\ChatStudio\\data\\model_visibility.json",
		path.join(process.cwd(), "ChatStudio", "data", "model_visibility.json"),
		path.join(process.cwd(), "..", "ChatStudio", "data", "model_visibility.json"),
	]
	for (const p of possiblePaths) {
		try {
			if (fs.existsSync(p)) {
				const parsed = JSON.parse(fs.readFileSync(p, "utf-8"))
				if (Array.isArray(parsed?.disabled)) {
					for (const id of parsed.disabled) {
						if (typeof id === "string" && id.trim()) {
							disabled.add(id.trim())
						}
					}
					break
				}
			}
		} catch {
			// Ignore filesystem errors
		}
	}
	return disabled
}

function unwrapRespondPayload(rawText: string): string {
	const trimmed = rawText.trim()
	if (!trimmed.startsWith("{")) {
		return trimmed
	}
	try {
		const parsed = JSON.parse(trimmed) as Partial<ClinePlanModeResponse>
		if (typeof parsed?.response === "string" && parsed.response.trim()) {
			return parsed.response.trim()
		}
	} catch {
		// Fall back to raw text
	}
	return trimmed
}

export interface PersistedOrchestratorCard {
	ts: number
	phase: "prompt" | "plan" | "act"
	text: string
}

export class LocalAiOrchestratorCoordinator {
	private enabled: boolean
	private customModelId?: string
	private status: OrchestratorStatus = "idle"
	private phaseDetail?: string
	private startedAt?: number
	private lastEvaluation?: OrchestratorLastEvaluation
	private readonly gatewayBaseUrl: string
	private readonly maxPlanRevisions: number

	private initialUserPrompt = ""
	private followupUserPrompts: string[] = []
	private lastPromptEvaluation?: OrchestratorPromptEvaluation
	private isCurrentTaskInformational = false
	private planRevisionCount = 0
	private lastEvaluatedPlanTs = 0
	private lastPlanTurnStartedAt = 0
	private isEvaluatingPlan = false
	private lastEvaluatedActTs = 0
	private isEvaluatingAct = false
	private activeAbortController?: AbortController
	private persistedCards = new Map<number, PersistedOrchestratorCard>()

	private cachedAvailableModels: string[] = []
	private cachedModelsFetchedAt = 0

	constructor(private readonly options: LocalAiOrchestratorOptions) {
		this.enabled = options.initialEnabled ?? false
		this.gatewayBaseUrl = (options.gatewayBaseUrl ?? "http://127.0.0.1:8080").replace(/\/+$/, "")
		this.maxPlanRevisions = options.maxPlanRevisions ?? 3
	}

	isEnabled(): boolean {
		return this.enabled
	}

	setEnabled(enabled: boolean): void {
		if (this.enabled === enabled) {
			return
		}
		this.enabled = enabled
		if (!enabled) {
			this.abortActiveEvaluation()
			this.setStatus("idle")
		}
		Logger.log(
			`[LocalAiOrchestrator] Orchestrator ${enabled ? "ENABLED" : "DISABLED"} (model=${this.resolveOrchestratorModelIdSync()})`,
		)
		this.notifyStateChanged()
	}

	setCustomModelId(modelId?: string): void {
		this.customModelId = modelId?.trim() || undefined
		this.notifyStateChanged()
	}

	getStateSnapshot(): OrchestratorStateSnapshot {
		const { planModelId, actModelId } = this.getPlanAndActModelIds()
		return {
			enabled: this.enabled,
			modelId: this.resolveOrchestratorModelIdSync(),
			planModelId,
			actModelId,
			status: this.status,
			phaseDetail: this.phaseDetail,
			startedAt: this.startedAt,
			revisionCount: this.planRevisionCount,
			maxRevisions: this.maxPlanRevisions,
			lastEvaluation: this.lastEvaluation,
		}
	}

	resetForNewTask(): void {
		this.abortActiveEvaluation()
		this.initialUserPrompt = ""
		this.followupUserPrompts = []
		this.lastPromptEvaluation = undefined
		this.lastEvaluation = undefined
		this.isCurrentTaskInformational = false
		this.planRevisionCount = 0
		this.lastEvaluatedPlanTs = 0
		this.lastPlanTurnStartedAt = 0
		this.isEvaluatingPlan = false
		this.lastEvaluatedActTs = 0
		this.isEvaluatingAct = false
		this.persistedCards.clear()
		this.setStatus("idle")
		this.notifyStateChanged()
	}

	abortActiveEvaluation(): void {
		if (this.activeAbortController) {
			try {
				this.activeAbortController.abort()
			} catch { }
			this.activeAbortController = undefined
		}
		this.isEvaluatingPlan = false
		this.isEvaluatingAct = false
		if (this.status !== "idle") {
			this.setStatus("idle")
		}
	}

	getPlanAndActModelIds(): { planModelId?: string; actModelId?: string } {
		const cfg = this.options.getApiConfiguration()
		const planModelId =
			(
				cfg.planModeOpenAiModelId ||
				cfg.planModeOllamaModelId ||
				cfg.planModeLmStudioModelId ||
				cfg.planModeApiModelId ||
				""
			).trim() || undefined
		const actModelId =
			(
				cfg.actModeOpenAiModelId ||
				cfg.actModeOllamaModelId ||
				cfg.actModeLmStudioModelId ||
				cfg.actModeApiModelId ||
				""
			).trim() || undefined
		return { planModelId, actModelId }
	}

	resolveOrchestratorModelIdSync(availablePool?: readonly string[]): string {
		const { planModelId, actModelId } = this.getPlanAndActModelIds()
		const excludedExact = new Set<string>(["qwen3-vl-8b-vision-caption-uncensored"])
		const excludedFamilies = new Set<string>()

		if (planModelId) {
			excludedExact.add(planModelId.toLowerCase())
			excludedFamilies.add(getModelFamilyPrefix(planModelId))
		}
		if (actModelId) {
			excludedExact.add(actModelId.toLowerCase())
			excludedFamilies.add(getModelFamilyPrefix(actModelId))
		}

		if (
			this.customModelId &&
			!excludedExact.has(this.customModelId.toLowerCase()) &&
			!excludedFamilies.has(getModelFamilyPrefix(this.customModelId))
		) {
			return this.customModelId
		}

		const disabledLower = new Set(Array.from(readLocalAiDisabledModels()).map((m) => m.toLowerCase()))
		const pool = availablePool && availablePool.length > 0 ? availablePool : this.cachedAvailableModels
		const poolLowerSet = pool.length > 0 ? new Set(pool.map((m) => m.toLowerCase())) : undefined

		for (const candidate of ORCHESTRATOR_MODEL_PRIORITY) {
			const lower = candidate.toLowerCase()
			if (excludedExact.has(lower) || excludedFamilies.has(getModelFamilyPrefix(candidate))) {
				continue
			}
			if (disabledLower.has(lower) || (poolLowerSet && !poolLowerSet.has(lower))) {
				continue
			}
			return candidate
		}

		for (const candidate of ORCHESTRATOR_MODEL_PRIORITY) {
			const lower = candidate.toLowerCase()
			if (!excludedExact.has(lower) && !disabledLower.has(lower)) {
				return candidate
			}
		}

		for (const candidate of ORCHESTRATOR_MODEL_PRIORITY) {
			if (!excludedExact.has(candidate.toLowerCase())) {
				return candidate
			}
		}

		return ORCHESTRATOR_MODEL_PRIORITY[0]
	}

	async resolveOrchestratorModelId(): Promise<string> {
		const liveModels = await this.fetchAvailableGatewayModels()
		return this.resolveOrchestratorModelIdSync(liveModels)
	}

	private recordPromptInHistory(trimmedPrompt: string): void {
		if (!this.initialUserPrompt) {
			this.initialUserPrompt = trimmedPrompt
			return
		}
		if (trimmedPrompt === this.initialUserPrompt) {
			return
		}
		if (this.followupUserPrompts[this.followupUserPrompts.length - 1] !== trimmedPrompt) {
			this.followupUserPrompts.push(trimmedPrompt)
		}
	}

	private hydratePromptHistoryFromMessages(messages: ClineMessage[]): void {
		if (this.initialUserPrompt) {
			return
		}
		for (const m of messages) {
			if (m.type === "say" && m.say === "task" && m.text?.trim()) {
				const t = m.text.trim()
				if (!this.isSyntheticOrInternalPrompt(t)) {
					this.initialUserPrompt = t
					this.isCurrentTaskInformational = this.isDirectSimpleQuestion(t)
				}
			} else if (m.type === "say" && m.say === "user_feedback" && m.text?.trim()) {
				const f = m.text.trim()
				if (!this.isSyntheticOrInternalPrompt(f)) {
					this.recordPromptInHistory(f)
					this.isCurrentTaskInformational = this.isDirectSimpleQuestion(f)
				}
			}
		}
	}

	private getEffectivePromptContextForAudit(): string {
		if (!this.initialUserPrompt && this.followupUserPrompts.length === 0) {
			return "(No prompt recorded)"
		}
		if (this.followupUserPrompts.length === 0) {
			return `USER REQUEST:\n"""\n${this.initialUserPrompt}\n"""`
		}
		const latest = this.followupUserPrompts[this.followupUserPrompts.length - 1]
		const earlier = this.followupUserPrompts.slice(0, -1)
		return [
			`INITIAL SESSION REQUEST:\n"""\n${this.initialUserPrompt}\n"""`,
			earlier.length > 0 ? `INTERMEDIATE INSTRUCTIONS:\n${earlier.map((p, i) => `${i + 1}. ${p}`).join("\n")}` : "",
			`LATEST ACTIVE USER REQUEST (PRIORITY):\n"""\n${latest}\n"""`,
		]
			.filter(Boolean)
			.join("\n\n")
	}

	private getActiveUserPrompt(): string {
		return this.followupUserPrompts.length > 0
			? this.followupUserPrompts[this.followupUserPrompts.length - 1]
			: this.initialUserPrompt
	}

	private async collectWorkspaceSnapshot(): Promise<string> {
		if (!this.options.getWorkspaceRoot) {
			return ""
		}
		try {
			const cwd = await this.options.getWorkspaceRoot()
			if (!cwd || !fs.existsSync(cwd)) {
				return ""
			}
			const parts: string[] = [`Workspace: ${path.basename(cwd)}`]
			try {
				const entries = fs.readdirSync(cwd, { withFileTypes: true }).slice(0, 15)
				parts.push(`Structure: ${entries.map((e) => (e.isDirectory() ? `${e.name}/` : e.name)).join(", ")}`)
			} catch { }
			try {
				const { stdout } = await execFileAsync("git", ["-C", cwd, "status", "--short", "--branch"], {
					timeout: 1000,
				})
				if (stdout.trim()) {
					parts.push(`Git Status:\n${stdout.trim().slice(0, 500)}`)
				}
			} catch { }
			return parts.join("\n")
		} catch {
			return ""
		}
	}

	private isDirectSimpleQuestion(prompt: string): boolean {
		const lower = prompt.toLowerCase().trim()
		if (prompt.split(/\s+/).length > 15) {
			return false
		}
		return [
			/^(do|should|can|shall)\s+(i|we)\s+(git\s+)?(push|commit|pull|merge|rebase|stash|build|test)\b/i,
			/^(devo|dovrei|posso|serve)\s+(fare\s+il\s+|fare\s+un\s+|fare\s+)?(push|commit|pull|git\s+push|git\s+commit)\b/i,
			/^(is|are|was|were|has|have|did|does)\b.*\?$/i,
			/^(serve|bisogna|c'è\s+bisogno)\b.*\?$/i,
			/\b(need to\s+(git\s+)?push|serve fare\s+(il\s+)?push)\b/i,
		].some((pattern) => pattern.test(lower))
	}

	async evaluateUserPrompt(rawPrompt: string, resolvedPrompt: string): Promise<string> {
		const trimmed = rawPrompt.trim()
		if (!this.enabled || !trimmed || this.isSyntheticOrInternalPrompt(trimmed)) {
			if (this.enabled && trimmed.startsWith(ORCHESTRATOR_REVISION_PREFIX)) {
				this.lastPlanTurnStartedAt = Date.now()
			}
			return resolvedPrompt
		}

		this.recordPromptInHistory(trimmed)
		this.planRevisionCount = 0

		if (this.isDirectSimpleQuestion(trimmed)) {
			this.isCurrentTaskInformational = true
			this.lastPromptEvaluation = undefined
			this.lastPlanTurnStartedAt = Date.now()
			const directHint = [
				"\n\n<orchestrator_evaluation>",
				"Objective: Answer the user's question directly and extremely concisely (e.g., Yes/No or a brief status).",
				"Directive: Do not generate a complex action plan or lengthy lists of individual files unless strictly necessary or requested by the user.",
				"</orchestrator_evaluation>",
			].join("\n")
			return `${resolvedPrompt}${directHint}`
		}

		this.isCurrentTaskInformational = false
		const effort: OrchestratorReasoningEffort = "low"
		const opStartMs = Date.now()
		const statusTs = opStartMs
		const traceLogs: string[] = []
		const pushLog = (step: string) =>
			traceLogs.push(`\`[${formatClock()}] (+${formatDuration(Date.now() - opStartMs)})\` ${step}`)

		this.startedAt = opStartMs
		this.setStatus("evaluating_prompt", "Inspecting workspace...")

		pushLog(`Intercepted user prompt (${trimmed.length} chars, ~${estimateTokens(trimmed)} tokens).`)
		const [orchestratorModel, workspaceSnapshot] = await Promise.all([
			this.resolveOrchestratorModelId(),
			this.collectWorkspaceSnapshot(),
		])
		const { planModelId, actModelId } = this.getPlanAndActModelIds()
		const pipelineLine = `**Pipeline:** Supervisor \`${orchestratorModel}\` *(Thinking: ${formatEffortLabel(effort)})* → Plan \`${planModelId ?? "default"}\` → Act \`${actModelId ?? "default"}\``

		if (workspaceSnapshot) {
			pushLog(`Acquired workspace and Git snapshot (${workspaceSnapshot.length} chars).`)
		}
		pushLog(`Selected supervisor: **\`${orchestratorModel}\`** *(Thinking: ${formatEffortLabel(effort)})*.`)

		const renderLive = (liveActivity: string, liveThinking?: string, liveOutput?: string) => {
			this.emitInfoMessage(
				statusTs,
				buildLiveCardMarkdown({
					title: "Prompt Analysis in Progress",
					elapsedMs: Date.now() - opStartMs,
					pipelineLine,
					liveActivity,
					traceLogs,
					liveThinking,
					liveOutput,
				}),
			)
		}

		renderLive("🔄 Initializing prompt and workspace context analysis...")

		try {
			const systemPrompt = [
				"You are the supervisory AI Orchestrator in Cline.",
				"Analyze the user's request and DIRECTLY generate an operational technical specification in JSON for Plan Mode.",
				"GUIDELINES:",
				"1. If the request is a simple question (e.g. 'do i need to push?', 'what does this function do?', 'can i commit?'), the objective must be to ANSWER DIRECTLY without turning it into complex plans.",
				"2. The agent starts in PLAN MODE (read-only exploration and planning). NEVER instruct Plan Mode to install packages, download files, or execute mutating actions directly; those belong in the plan for Act Mode.",
				"3. If the user mentions an example external tool like '(e.g., OWASP ZAP)', treat it as an example: instruct Plan Mode to check if it is installed with a single read-only check, and if not installed, plan a fallback using already-available local tools (e.g., Python scripts) instead of trying to install missing third-party software or package managers.",
				"4. Maximum technical conciseness, no text outside the JSON:",
				"{",
				'  "isClear": true,',
				'  "qualityScore": 8,',
				'  "objective": "Concise objective in 1 sentence",',
				'  "technicalContext": "Essential context in 1 sentence",',
				'  "requirements": ["Concise requirement"],',
				'  "risksAndEdgeCases": [],',
				'  "acceptanceCriteria": ["Provide a concise and direct answer or plan"],',
				'  "planningGuidance": "In Plan Mode, perform only read-only checks to verify available tools/files, then present a concrete plan (using Python/local fallbacks if external CLI tools are absent)."',
				"}",
			].join("\n")

			const userContent = [
				`USER REQUEST:\n"""\n${trimmed}\n"""`,
				workspaceSnapshot ? `\nCONTEXT:\n${workspaceSnapshot}` : "",
				this.followupUserPrompts.length > 0 && this.initialUserPrompt !== trimmed
					? `\nINITIAL SESSION CONTEXT:\n${this.initialUserPrompt}`
					: "",
			]
				.filter(Boolean)
				.join("\n")

			const metrics = await this.callOrchestratorLlmVerbose({
				modelId: orchestratorModel,
				systemPrompt,
				userPrompt: userContent,
				maxTokens: 2048,
				reasoningEffort: effort,
				phaseLabel: "Prompt",
				pushLog,
				onLiveUpdate: renderLive,
			})

			const parsed = extractFirstJsonObject(metrics.content) || extractFirstJsonObject(metrics.thinkingText)
			let evaluation: OrchestratorPromptEvaluation

			if (
				parsed &&
				typeof parsed.objective === "string" &&
				parsed.objective.trim().toLowerCase() !== trimmed.toLowerCase()
			) {
				const parsedReqs = Array.isArray(parsed.requirements)
					? parsed.requirements.filter(
						(r): r is string =>
							typeof r === "string" && r.trim().length > 0 && r.trim().toLowerCase() !== trimmed.toLowerCase(),
					)
					: []
				const parsedRisks = Array.isArray(parsed.risksAndEdgeCases)
					? parsed.risksAndEdgeCases.filter((r): r is string => typeof r === "string" && r.trim().length > 0)
					: []
				const parsedCriteria = Array.isArray(parsed.acceptanceCriteria)
					? parsed.acceptanceCriteria.filter((c): c is string => typeof c === "string" && c.trim().length > 0)
					: []

				evaluation = {
					isClear: typeof parsed.isClear === "boolean" ? parsed.isClear : true,
					qualityScore:
						typeof parsed.qualityScore === "number" && Number.isFinite(parsed.qualityScore)
							? Math.max(1, Math.min(10, Math.round(parsed.qualityScore)))
							: 8,
					objective: parsed.objective.trim(),
					technicalContext:
						typeof parsed.technicalContext === "string" && parsed.technicalContext.trim()
							? parsed.technicalContext.trim()
							: workspaceSnapshot
								? "Workspace and Git context acquired and integrated into the analysis."
								: "Architectural analysis based on the user prompt.",
					requirements:
						parsedReqs.length > 0
							? parsedReqs
							: synthesizeEvaluationFromReasoning(trimmed, metrics.thinkingText, metrics.content, workspaceSnapshot)
								.requirements,
					risksAndEdgeCases:
						parsedRisks.length > 0
							? parsedRisks
							: ["Verify the actual state of files and the repository before proposing changes."],
					acceptanceCriteria:
						parsedCriteria.length > 0
							? parsedCriteria
							: ["The Plan must cover all technical requirements extracted above."],
					planningGuidance:
						typeof parsed.planningGuidance === "string" && parsed.planningGuidance.trim()
							? parsed.planningGuidance.trim()
							: "Formulate a detailed technical plan, verified against actual workspace files and ready for execution.",
				}
			} else {
				pushLog(`ℹ️ Structured summary extracted from reasoning (${metrics.estimatedThinkingTokens} tok).`)
				evaluation = synthesizeEvaluationFromReasoning(trimmed, metrics.thinkingText, metrics.content, workspaceSnapshot)
			}

			this.lastPromptEvaluation = evaluation
			this.setLastEvaluation({
				phase: "prompt",
				verdict: "prompt_ready",
				score: evaluation.qualityScore,
				summary: evaluation.objective,
				timestamp: Date.now(),
			})

			const totalMs = Date.now() - opStartMs
			pushLog(
				`Analysis completed (**${evaluation.qualityScore}/10**, **${evaluation.requirements.length}** requirements). Handoff to **Plan Mode** (\`${planModelId ?? "default"}\`).`,
			)

			const callout = buildSupervisorCallout({
				verdictBadge: `🎯 TECHNICAL SPECIFICATION READY FOR PLAN MODE`,
				score: evaluation.qualityScore,
				extraSubtitle: `${evaluation.requirements.length} requirements extracted`,
				evaluationText: `**Objective:** ${evaluation.objective}\n**Directive for \`${planModelId ?? "default"}\`:** ${evaluation.planningGuidance}`,
			})

			this.emitInfoMessage(
				statusTs,
				[
					callout,
					`**⚡ Orchestrator Action:** Specification sent to **Plan Mode** (\`${planModelId ?? "default"}\`) — awaiting Plan generation...`,
					evaluation.requirements.length > 0
						? `**Mandatory Technical Requirements (${evaluation.requirements.length}):**\n${evaluation.requirements.map((r, i) => `${i + 1}. ${r}`).join("\n")}`
						: "",
					evaluation.risksAndEdgeCases.length > 0
						? `**Risks & Edge Cases:**\n${evaluation.risksAndEdgeCases.map((r) => `- ⚠️ ${r}`).join("\n")}`
						: "",
					evaluation.acceptanceCriteria.length > 0
						? `**Acceptance Criteria:**\n${evaluation.acceptanceCriteria.map((c) => `- [ ] ${c}`).join("\n")}`
						: "",
					`---`,
					formatTelemetryBar(orchestratorModel, totalMs, metrics),
					formatThinkingPreview(metrics.thinkingText, metrics.estimatedThinkingTokens),
					`**Operation Log (${traceLogs.length} events):**\n${traceLogs.map((l) => `- ${l}`).join("\n")}`,
				]
					.filter(Boolean)
					.join("\n\n"),
			)

			sendWindowsNotification(
				"AI Supervisor: Prompt Specification Ready",
				`Quality ${evaluation.qualityScore}/10 • ${evaluation.objective}`,
			)

			const orchestratorNotesBlock = [
				"\n\n<orchestrator_evaluation>",
				`Expanded Technical Objective: ${evaluation.objective}`,
				evaluation.planningGuidance ? `Directive: ${evaluation.planningGuidance}` : "",
				evaluation.requirements.length > 0
					? "Key Requirements:\n" + evaluation.requirements.map((r, idx) => `${idx + 1}. ${r}`).join("\n")
					: "",
				"PLAN MODE & ANTI-LOOP RULES: You are currently in PLAN MODE (read-only). Do NOT install packages, download executables, or modify files in Plan Mode. If an example CLI tool (e.g. zaproxy, chocolatey, docker) is not installed, do NOT retry installing it or guess download URLs — immediately design your plan around tools already available in the environment (such as Python `requests`/`ssl`) and present your plan.",
				"IMPORTANT: If the user asked a concise question (e.g. yes/no, git push status, clarification), answer directly and concisely without unsolicited file dumps or bloated step-by-step plans.",
				"</orchestrator_evaluation>",
			]
				.filter(Boolean)
				.join("\n")

			this.lastPlanTurnStartedAt = Date.now()
			return `${resolvedPrompt}${orchestratorNotesBlock}`
		} catch (error) {
			const totalMs = Date.now() - opStartMs
			const errMsg = error instanceof Error ? error.message : String(error)
			Logger.warn("[LocalAiOrchestrator] Prompt evaluation skipped/failed, proceeding with original prompt:", error)
			this.emitInfoMessage(
				statusTs,
				[
					`## 🛡️ Supervisor Evaluation`,
					`> ⚠️ **Preliminary analysis skipped** *(fast fallback in ${formatDuration(totalMs)}: ${errMsg})*`,
					`▶️ *Forwarding prompt directly to Plan Mode (\`${planModelId ?? "default"}\`)...*`,
				].join("\n\n"),
			)
			this.lastPlanTurnStartedAt = Date.now()
			return resolvedPrompt
		} finally {
			this.startedAt = undefined
			this.setStatus("idle")
		}
	}

	async handleTurnCompleted(): Promise<void> {
		this.persistCardsToDisk()
		if (!this.enabled) {
			return
		}
		const mode = this.options.getCurrentMode()
		if (mode === "plan") {
			await this.handlePlanTurnCompleted()
		} else if (mode === "act") {
			await this.handleActTurnCompleted()
		}
	}

	private isWaitingOnBlockingUserAsk(tailMessage?: ClineMessage): boolean {
		return Boolean(tailMessage?.type === "ask" && tailMessage.ask && USER_BLOCKING_ASKS.has(tailMessage.ask))
	}

	async handleActTurnCompleted(): Promise<void> {
		if (!this.enabled || this.isEvaluatingAct || this.options.getCurrentMode() !== "act" || this.options.isSessionRunning()) {
			return
		}

		const messages = this.options.getClineMessages()
		if (messages.length === 0 || this.isWaitingOnBlockingUserAsk(messages[messages.length - 1])) {
			return
		}

		const actTurn = this.findLatestCompletedTurn(messages, "act")
		if (!actTurn || !actTurn.text.trim() || actTurn.latestTs <= this.lastEvaluatedActTs) {
			return
		}

		this.lastEvaluatedActTs = actTurn.latestTs
		this.isEvaluatingAct = true
		const effort: OrchestratorReasoningEffort = "low"
		const opStartMs = Date.now()
		const statusTs = opStartMs
		this.startedAt = opStartMs
		this.setStatus("evaluating_act", "Verifying Act...")

		const actText = actTurn.text.trim()
		const traceLogs: string[] = []
		const pushLog = (step: string) =>
			traceLogs.push(`\`[${formatClock()}] (+${formatDuration(Date.now() - opStartMs)})\` ${step}`)

		const orchestratorModel = await this.resolveOrchestratorModelId()
		const { planModelId, actModelId } = this.getPlanAndActModelIds()
		this.hydratePromptHistoryFromMessages(messages)
		const pipelineLine = `**Pipeline:** Act \`${actModelId ?? "default"}\` → Supervisor \`${orchestratorModel}\` *(Thinking: ${formatEffortLabel(effort)})* → Plan \`${planModelId ?? "default"}\``

		pushLog(
			`Received **Act Mode** output from \`${actModelId ?? "default"}\` (${actText.length} chars, ~${estimateTokens(actText)} tok). Starting verification.`,
		)

		const renderLive = (liveActivity: string, liveThinking?: string, liveOutput?: string) => {
			this.emitInfoMessage(
				statusTs,
				buildLiveCardMarkdown({
					title: "Act Work Verification in Progress",
					elapsedMs: Date.now() - opStartMs,
					pipelineLine,
					liveActivity,
					traceLogs,
					liveThinking,
					liveOutput,
				}),
			)
		}

		renderLive("🔄 Verifying work performed in Act Mode and analyzing next steps...")

		try {
			const { review, metrics } = await this.evaluateActWithLlm(
				orchestratorModel,
				actText,
				effort,
				pushLog,
				renderLive,
			)
			if (!this.enabled) {
				return
			}

			this.setLastEvaluation({
				phase: "act",
				verdict: review.completed ? "act_completed" : "act_needs_work",
				score: review.score,
				summary: review.summary,
				timestamp: Date.now(),
			})

			const proposedAction =
				review.proposedNextAction?.trim() ||
				(review.completed
					? "Final integration check or end-to-end testing via Plan Mode."
					: "Complete the missing items by generating a new plan in Plan Mode.")

			pushLog(`ACT VERIFICATION COMPLETED: **${review.completed ? "COMPLETED" : "NEEDS REFINEMENT"}** (**${review.score}/10**).`)

			const callout = buildSupervisorCallout({
				verdictBadge: review.completed
					? `✅ ACT WORK VERIFIED AND COMPLETED`
					: `🛠️ PARTIAL ACT WORK — REQUIRES REFINEMENT`,
				score: review.score,
				extraSubtitle: `Act Model: \`${actModelId ?? "default"}\``,
				evaluationText: review.summary,
			})

			this.emitInfoMessage(
				statusTs,
				[
					callout,
					`💡 **Proposed Next Change (for Plan Mode \`${planModelId ?? "default"}\`):**\n${proposedAction}`,
					review.verifiedPoints.length > 0
						? `**Verified Points:**\n${review.verifiedPoints.map((v) => `- ✅ ${v}`).join("\n")}`
						: "",
					review.issuesOrPending.length > 0
						? `**Items to Refine / Pending:**\n${review.issuesOrPending.map((p) => `- ⚠️ ${p}`).join("\n")}`
						: "",
					`---`,
					formatTelemetryBar(orchestratorModel, Date.now() - opStartMs, metrics),
					formatThinkingPreview(metrics.thinkingText, metrics.estimatedThinkingTokens),
					`**Operation Log (${traceLogs.length} events):**\n${traceLogs.map((l) => `- ${l}`).join("\n")}`,
				]
					.filter(Boolean)
					.join("\n\n"),
			)

			sendWindowsNotification(`AI Supervisor: Act Verification (${review.score}/10)`, `${review.summary} → ${proposedAction}`)
		} catch (error) {
			if ((error as Error)?.name === "AbortError") {
				return
			}
			const errMsg = error instanceof Error ? error.message : String(error)
			Logger.error("[LocalAiOrchestrator] Act evaluation failed:", error)
			this.emitInfoMessage(
				statusTs,
				[
					`## 🛡️ Supervisor Evaluation`,
					`> ⚠️ **Error during Act Mode verification:** \`${errMsg}\``,
					`**Operation Log:**\n${traceLogs.map((l) => `- ${l}`).join("\n")}`,
				].join("\n\n"),
			)
		} finally {
			this.isEvaluatingAct = false
			this.startedAt = undefined
			this.setStatus("idle")
		}
	}

	private async evaluateActWithLlm(
		orchestratorModel: string,
		actText: string,
		effort: OrchestratorReasoningEffort,
		pushLog: (step: string) => void,
		onLiveUpdate: (activity: string, thinking?: string, partialContent?: string) => void,
	): Promise<{ review: OrchestratorActReview; metrics: LlmCallMetrics }> {
		const systemPrompt = [
			"You are the supervisory AI Orchestrator in Cline.",
			"Check whether the work performed in Act Mode meets the objectives of the user's request.",
			"Respond EXCLUSIVELY with valid JSON (no text outside the JSON):",
			"{",
			'  "completed": true,',
			'  "score": 9,',
			'  "summary": "Clear and direct assessment of the work performed",',
			'  "verifiedPoints": ["Verified point"],',
			'  "issuesOrPending": [],',
			'  "proposedNextAction": "Concrete proposal for the next step to generate in Plan Mode"',
			"}",
		].join("\n")

		const userContent = [
			this.getEffectivePromptContextForAudit(),
			`\nWORK EXECUTED BY ACT MODE:\n"""\n${actText.slice(0, 8000)}\n"""`,
		].join("\n")

		const metrics = await this.callOrchestratorLlmVerbose({
			modelId: orchestratorModel,
			systemPrompt,
			userPrompt: userContent,
			maxTokens: 2048,
			reasoningEffort: effort,
			phaseLabel: "Act",
			pushLog,
			onLiveUpdate,
		})

		const parsed = extractFirstJsonObject(metrics.content) || extractFirstJsonObject(metrics.thinkingText)
		if (!parsed) {
			const cleanedText = stripThinkingBlocks(metrics.content) || metrics.thinkingText
			return {
				review: {
					completed: true,
					score: 8,
					summary: cleanedText.slice(0, 500) || "Work successfully verified.",
					verifiedPoints: ["Work completed in Act Mode"],
					issuesOrPending: [],
					proposedNextAction: "Switch to Plan Mode to plan any refinements or new requirements.",
				},
				metrics,
			}
		}

		const completed = Boolean(parsed.completed)
		return {
			review: {
				completed,
				score:
					typeof parsed.score === "number" && Number.isFinite(parsed.score)
						? Math.max(1, Math.min(10, Math.round(parsed.score)))
						: completed
							? 8
							: 5,
				summary:
					typeof parsed.summary === "string" && parsed.summary.trim()
						? parsed.summary.trim()
						: completed
							? "The work in Act Mode is consistent with the user's request."
							: "The work in Act Mode requires additional steps.",
				verifiedPoints: Array.isArray(parsed.verifiedPoints)
					? parsed.verifiedPoints.filter((v): v is string => typeof v === "string" && v.trim().length > 0)
					: [],
				issuesOrPending: Array.isArray(parsed.issuesOrPending)
					? parsed.issuesOrPending.filter((i): i is string => typeof i === "string" && i.trim().length > 0)
					: [],
				proposedNextAction:
					typeof parsed.proposedNextAction === "string" && parsed.proposedNextAction.trim()
						? parsed.proposedNextAction.trim()
						: "Generate a plan in Plan Mode for the next steps or final verification.",
			},
			metrics,
		}
	}

	async handlePlanTurnCompleted(): Promise<void> {
		if (!this.enabled || this.isEvaluatingPlan || this.options.getCurrentMode() !== "plan" || this.options.isSessionRunning()) {
			return
		}

		const messages = this.options.getClineMessages()
		if (messages.length === 0 || this.isWaitingOnBlockingUserAsk(messages[messages.length - 1])) {
			return
		}

		const planTurn = this.findLatestCompletedTurn(messages, "plan")
		if (!planTurn || !planTurn.text.trim() || planTurn.latestTs <= this.lastEvaluatedPlanTs) {
			return
		}

		this.lastEvaluatedPlanTs = planTurn.latestTs
		this.isEvaluatingPlan = true
		const effort: OrchestratorReasoningEffort = "low"
		const opStartMs = Date.now()
		const statusTs = opStartMs
		this.startedAt = opStartMs
		this.setStatus("evaluating_plan", "Auditing Plan...")

		const planDurationMs =
			this.lastPlanTurnStartedAt > 0 ? Math.max(0, opStartMs - this.lastPlanTurnStartedAt) : undefined
		const planText = planTurn.text.trim()
		const planChars = planText.length
		const planTokens = estimateTokens(planText)

		const traceLogs: string[] = []
		const pushLog = (step: string) =>
			traceLogs.push(`\`[${formatClock()}] (+${formatDuration(Date.now() - opStartMs)})\` ${step}`)

		const orchestratorModel = await this.resolveOrchestratorModelId()
		const { planModelId, actModelId } = this.getPlanAndActModelIds()
		this.hydratePromptHistoryFromMessages(messages)

		const pipelineLine = `**Pipeline:** Plan \`${planModelId ?? "default"}\` → Supervisor \`${orchestratorModel}\` *(Thinking: ${formatEffortLabel(effort)})* → Act \`${actModelId ?? "default"}\``
		const metaLine = `**Plan Size:** \`${planChars} chars\` (\`~${planTokens} tok\`)${planDurationMs ? ` • **Plan Time:** \`${formatDuration(planDurationMs)}\`` : ""}`

		pushLog(`Received Plan from \`${planModelId ?? "default"}\` (${planChars} chars, ~${planTokens} tok). Starting audit.`)

		const renderLive = (liveActivity: string, liveThinking?: string, liveOutput?: string) => {
			this.emitInfoMessage(
				statusTs,
				buildLiveCardMarkdown({
					title: "Plan Audit in Progress",
					elapsedMs: Date.now() - opStartMs,
					pipelineLine,
					metaLine,
					liveActivity,
					traceLogs,
					liveThinking,
					liveOutput,
				}),
			)
		}

		renderLive("🔄 Evaluating Plan coherence, completeness, and safety...")

		try {
			const { decision, metrics } = await this.evaluatePlanWithLlm(
				orchestratorModel,
				planText,
				effort,
				pushLog,
				renderLive,
			)
			if (!this.enabled) {
				return
			}

			const telemetrySummary = formatTelemetryBar(
				orchestratorModel,
				Date.now() - opStartMs,
				metrics,
				planDurationMs ? `**Plan Time:** \`${formatDuration(planDurationMs)}\`` : undefined,
			)
			const strengthsBullets =
				decision.strengths.length > 0 ? decision.strengths.map((s) => `- ✅ ${s}`).join("\n") : ""

			if (decision.valid) {
				this.planRevisionCount = 0
				const shouldTriggerAct = !this.isCurrentTaskInformational && decision.requiresActExecution !== false

				if (!shouldTriggerAct) {
					this.setLastEvaluation({
						phase: "plan",
						verdict: "info_approved",
						score: decision.score,
						summary: decision.reasoning,
						timestamp: Date.now(),
					})

					pushLog(`VERDICT: INFORMATIONAL RESPONSE APPROVED (**${decision.score}/10**).`)
					const callout = buildSupervisorCallout({
						verdictBadge: `✅ VALID AND COMPLETE RESPONSE`,
						score: decision.score,
						extraSubtitle: `Informational question resolved in Plan Mode`,
						evaluationText: decision.reasoning,
					})

					this.emitInfoMessage(
						statusTs,
						[
							callout,
							`**⚡ Orchestrator Action:** Response confirmed in **Plan Mode** *(no changes required in Act Mode)*.`,
							strengthsBullets ? `**Verified Points:**\n${strengthsBullets}` : "",
							`---`,
							telemetrySummary,
							formatThinkingPreview(metrics.thinkingText, metrics.estimatedThinkingTokens),
							`**Operation Log (${traceLogs.length} events):**\n${traceLogs.map((l) => `- ${l}`).join("\n")}`,
						]
							.filter(Boolean)
							.join("\n\n"),
					)
					sendWindowsNotification(`AI Supervisor: Response Approved (${decision.score}/10)`, decision.reasoning)
					return
				}

				this.setLastEvaluation({
					phase: "plan",
					verdict: "approved",
					score: decision.score,
					summary: decision.reasoning,
					timestamp: Date.now(),
				})

				pushLog(`VERDICT: PLAN APPROVED (**${decision.score}/10**). Automatically starting **Act Mode**...`)
				const actSwitchStart = Date.now()
				this.setStatus("idle")
				this.startedAt = undefined

				const callout = buildSupervisorCallout({
					verdictBadge: `✅ PLAN APPROVED BY SUPERVISOR`,
					score: decision.score,
					extraSubtitle: `Authorized start of Act Mode (\`${actModelId ?? "default"}\`)`,
					evaluationText: decision.reasoning,
				})

				const buildApprovedCard = (actActionText: string) =>
					[
						callout,
						`**⚡ Orchestrator Action:** ${actActionText}`,
						strengthsBullets ? `**Requirements Verified in Plan:**\n${strengthsBullets}` : "",
						`---`,
						telemetrySummary,
						formatThinkingPreview(metrics.thinkingText, metrics.estimatedThinkingTokens),
						`**Operation Log (${traceLogs.length} events):**\n${traceLogs.map((l) => `- ${l}`).join("\n")}`,
					]
						.filter(Boolean)
						.join("\n\n")

				this.emitInfoMessage(
					statusTs,
					buildApprovedCard(`Automatically switching to **Act Mode** (\`${actModelId ?? "default"}\`)...`),
				)

				await this.options.triggerActMode()
				const actSwitchMs = Date.now() - actSwitchStart
				pushLog(`Switch to **Act Mode** completed in \`${formatDuration(actSwitchMs)}\`.`)
				this.emitInfoMessage(
					statusTs,
					buildApprovedCard(
						`Automatically triggered **Act Mode** in \`${formatDuration(actSwitchMs)}\` → Work started with \`${actModelId ?? "default"}\`.`,
					),
				)
				sendWindowsNotification(
					`AI Supervisor: Plan Approved (${decision.score}/10)`,
					`${decision.reasoning} → Started Act Mode (${actModelId ?? "default"})`,
				)
				return
			}

			// Plan rejected / needs modifications
			this.planRevisionCount += 1
			const modificationsBullets =
				decision.suggestedModifications.length > 0
					? decision.suggestedModifications.map((m, idx) => `${idx + 1}. ${m}`).join("\n")
					: decision.missingPoints.map((m, idx) => `${idx + 1}. ${m}`).join("\n") || `1. ${decision.reasoning}`
			const missingBullets =
				decision.missingPoints.length > 0 ? decision.missingPoints.map((p) => `- ❌ ${p}`).join("\n") : ""

			pushLog(`VERDICT: PLAN NEEDS REVISION (**${decision.score}/10** — Revision **${this.planRevisionCount}/${this.maxPlanRevisions}**).`)

			if (this.planRevisionCount <= this.maxPlanRevisions) {
				this.setLastEvaluation({
					phase: "plan",
					verdict: "revision_requested",
					score: decision.score,
					summary: decision.reasoning,
					timestamp: Date.now(),
				})

				const callout = buildSupervisorCallout({
					verdictBadge: `⚠️ PLAN NEEDS REVISION (Revision ${this.planRevisionCount}/${this.maxPlanRevisions})`,
					score: decision.score,
					extraSubtitle: `Requested regeneration from \`${planModelId ?? "default"}\``,
					evaluationText: decision.reasoning,
				})

				this.emitInfoMessage(
					statusTs,
					[
						callout,
						`**🔧 Changes Requested by Supervisor (automatically sent to Plan Mode):**\n${modificationsBullets}`,
						missingBullets ? `**Gaps or Risks Detected:**\n${missingBullets}` : "",
						`---`,
						telemetrySummary,
						formatThinkingPreview(metrics.thinkingText, metrics.estimatedThinkingTokens),
						`**Operation Log (${traceLogs.length} events):**\n${traceLogs.map((l) => `- ${l}`).join("\n")}`,
					]
						.filter(Boolean)
						.join("\n\n"),
				)
				sendWindowsNotification(
					`AI Supervisor: Plan Needs Revision (${this.planRevisionCount}/${this.maxPlanRevisions})`,
					`Score ${decision.score}/10: ${decision.reasoning}`,
				)

				const revisionPrompt = [
					`${ORCHESTRATOR_REVISION_PREFIX} The AI Orchestrator analyzed your plan against the user's request and requires the following changes before switching to Act Mode:`,
					"",
					`Supervisor Evaluation: ${decision.reasoning}`,
					"",
					"Mandatory changes to apply to the Plan:",
					modificationsBullets,
					"",
					`User request to satisfy in full:\n"""\n${this.getActiveUserPrompt()}\n"""`,
					"",
					"Use read-only tools if necessary to gather missing details, then immediately update and revise the plan in Plan Mode including all the changes listed above.",
				].join("\n")

				this.startedAt = undefined
				this.setStatus("idle")
				this.lastPlanTurnStartedAt = Date.now()
				await this.options.sendPlanRevisionPrompt(revisionPrompt)
				return
			}

			// Max revisions reached: evaluate fallback compromise
			const isAcceptable =
				decision.isAcceptableFallback ??
				(decision.score >= 6 ||
					(decision.score >= 5 &&
						decision.missingPoints.length <= 1 &&
						!decision.reasoning.toLowerCase().includes("critical") &&
						!decision.reasoning.toLowerCase().includes("unacceptable")))

			if (isAcceptable) {
				this.setLastEvaluation({
					phase: "plan",
					verdict: "accepted_fallback",
					score: decision.score,
					summary: decision.reasoning,
					timestamp: Date.now(),
				})

				const shouldTriggerAct = !this.isCurrentTaskInformational && decision.requiresActExecution !== false
				const callout = buildSupervisorCallout({
					verdictBadge: `🟡 PLAN ACCEPTED WITH RESERVATIONS (Limit ${this.maxPlanRevisions}/${this.maxPlanRevisions})`,
					score: decision.score,
					extraSubtitle: shouldTriggerAct ? `Authorized start of Act Mode` : `Response kept in Plan Mode`,
					evaluationText: decision.reasoning,
				})

				if (!shouldTriggerAct) {
					this.emitInfoMessage(
						statusTs,
						[
							callout,
							missingBullets ? `**Remaining Notes:**\n${missingBullets}` : "",
							`---`,
							telemetrySummary,
							`**Operation Log (${traceLogs.length} events):**\n${traceLogs.map((l) => `- ${l}`).join("\n")}`,
						]
							.filter(Boolean)
							.join("\n\n"),
					)
					return
				}

				this.setStatus("idle")
				this.startedAt = undefined
				this.emitInfoMessage(
					statusTs,
					[
						callout,
						`**⚡ Orchestrator Action:** Plan deemed sufficiently executable → Automatically switching to **Act Mode** (\`${actModelId ?? "default"}\`)...`,
						missingBullets ? `**Remaining Notes:**\n${missingBullets}` : "",
						`---`,
						telemetrySummary,
						`**Operation Log (${traceLogs.length} events):**\n${traceLogs.map((l) => `- ${l}`).join("\n")}`,
					]
						.filter(Boolean)
						.join("\n\n"),
				)
				sendWindowsNotification(
					`AI Supervisor: Plan Accepted with Reservations (${decision.score}/10)`,
					`Revision limit reached → Starting Act Mode`,
				)
				await this.options.triggerActMode()
				return
			}

			// Critical block: require manual user decision
			this.setLastEvaluation({
				phase: "plan",
				verdict: "blocked",
				score: decision.score,
				summary: decision.reasoning,
				timestamp: Date.now(),
			})

			const blockedCallout = buildSupervisorCallout({
				verdictBadge: `🛑 PLAN BLOCKED — USER INTERVENTION REQUIRED`,
				score: decision.score,
				extraSubtitle: `Limit of ${this.maxPlanRevisions}/${this.maxPlanRevisions} revisions reached`,
				evaluationText: decision.reasoning,
			})

			this.emitInfoMessage(
				statusTs,
				[
					blockedCallout,
					`**⚡ Orchestrator Action:** Precautionary block active *(Act Mode NOT started automatically)*. You can send corrective instructions in Plan Mode or manually start Act.`,
					missingBullets ? `**Critical Issues Detected:**\n${missingBullets}` : "",
					`**Required Changes:**\n${modificationsBullets}`,
					`---`,
					telemetrySummary,
					`**Operation Log (${traceLogs.length} events):**\n${traceLogs.map((l) => `- ${l}`).join("\n")}`,
				]
					.filter(Boolean)
					.join("\n\n"),
			)
			sendWindowsNotification(
				`AI Supervisor: User Intervention Required (${decision.score}/10)`,
				decision.reasoning,
			)
		} catch (error) {
			if ((error as Error)?.name === "AbortError") {
				return
			}
			const errMsg = error instanceof Error ? error.message : String(error)
			Logger.error("[LocalAiOrchestrator] Plan evaluation failed:", error)
			this.emitInfoMessage(
				statusTs,
				[
					`## 🛡️ Supervisor Evaluation`,
					`> ⚠️ **Error during Plan verification:** \`${errMsg}\``,
					`**Operation Log:**\n${traceLogs.map((l) => `- ${l}`).join("\n")}`,
				].join("\n\n"),
			)
		} finally {
			this.isEvaluatingPlan = false
			this.startedAt = undefined
			this.setStatus("idle")
		}
	}

	private async evaluatePlanWithLlm(
		orchestratorModel: string,
		planText: string,
		effort: OrchestratorReasoningEffort,
		pushLog: (step: string) => void,
		onLiveUpdate: (activity: string, thinking?: string, partialContent?: string) => void,
	): Promise<{ decision: OrchestratorPlanDecision; metrics: LlmCallMetrics }> {
		const systemPrompt = [
			"You are the supervisory AI Orchestrator in Cline.",
			"Verify whether the Plan adequately addresses the user's request.",
			"RULES:",
			'1. If the plan is reasonable, coherent, and executable (even if imperfect), set "valid": true.',
			'2. Reject the plan ("valid": false) ONLY if it is off-track, incomplete on essential requirements, or destructive.',
			'3. Set "requiresActExecution": false ONLY if the user request was a purely informational question and the Plan Mode response has already answered it completely without needing file edits or commands in Act Mode. Otherwise set "requiresActExecution": true.',
			"4. Respond EXCLUSIVELY with valid JSON (no text outside the JSON):",
			"{",
			'  "valid": true,',
			'  "requiresActExecution": true,',
			'  "isAcceptableFallback": true,',
			'  "score": 8,',
			'  "reasoning": "Clear and direct evaluation from the Supervisor",',
			'  "strengths": ["Covered point"],',
			'  "missingPoints": [],',
			'  "suggestedModifications": []',
			"}",
		].join("\n")

		const requirementsSection = this.lastPromptEvaluation
			? [
				"\nEXPECTED CRITERIA:",
				`Objective: ${this.lastPromptEvaluation.objective}`,
				...this.lastPromptEvaluation.requirements.map((r) => `- ${r}`),
			].join("\n")
			: ""

		const userContent = [
			this.getEffectivePromptContextForAudit(),
			requirementsSection,
			`\nPLAN PROPOSED BY PLAN MODE:\n"""\n${planText.slice(0, 8000)}\n"""`,
		]
			.filter(Boolean)
			.join("\n")

		const metrics = await this.callOrchestratorLlmVerbose({
			modelId: orchestratorModel,
			systemPrompt,
			userPrompt: userContent,
			maxTokens: 2048,
			reasoningEffort: effort,
			phaseLabel: "Plan",
			pushLog,
			onLiveUpdate,
		})

		const parsed = extractFirstJsonObject(metrics.content) || extractFirstJsonObject(metrics.thinkingText)
		if (!parsed) {
			const cleanedText = stripThinkingBlocks(metrics.content) || metrics.thinkingText
			const lower = cleanedText.toLowerCase()
			const looksApproved =
				lower.includes('"valid": true') ||
				lower.includes('"valid":true') ||
				(lower.includes("approved") && !lower.includes("not approved") && !lower.includes("invalid"))
			return {
				decision: {
					valid: looksApproved,
					score: looksApproved ? 8 : 5,
					reasoning: cleanedText.slice(0, 600) || "Evaluation completed.",
					strengths: [],
					missingPoints: [],
					suggestedModifications: looksApproved
						? []
						: ["Expand the plan by verifying the actual state of the files and covering all requested requirements."],
					requiresActExecution: !this.isCurrentTaskInformational,
				},
				metrics,
			}
		}

		const valid = Boolean(parsed.valid)
		const score =
			typeof parsed.score === "number" && Number.isFinite(parsed.score)
				? Math.max(1, Math.min(10, Math.round(parsed.score)))
				: valid
					? 8
					: 4
		const missingPoints = Array.isArray(parsed.missingPoints)
			? parsed.missingPoints.filter((p): p is string => typeof p === "string" && p.trim().length > 0)
			: []

		return {
			decision: {
				valid,
				score,
				reasoning:
					typeof parsed.reasoning === "string" && parsed.reasoning.trim()
						? parsed.reasoning.trim()
						: valid
							? "The plan satisfies all requirements requested by the user."
							: "The plan does not fully satisfy the requested requirements.",
				strengths: Array.isArray(parsed.strengths)
					? parsed.strengths.filter((s): s is string => typeof s === "string" && s.trim().length > 0)
					: [],
				missingPoints,
				suggestedModifications: Array.isArray(parsed.suggestedModifications)
					? parsed.suggestedModifications.filter((m): m is string => typeof m === "string" && m.trim().length > 0)
					: missingPoints,
				isAcceptableFallback:
					typeof parsed.isAcceptableFallback === "boolean" ? parsed.isAcceptableFallback : valid || score >= 6,
				requiresActExecution:
					typeof parsed.requiresActExecution === "boolean"
						? parsed.requiresActExecution
						: !this.isCurrentTaskInformational,
			},
			metrics,
		}
	}

	/**
	 * Collects completed assistant output for the latest Plan or Act turn, including both `say`
	 * (`plan_completion_result`, `completion_result`, `text`) and `ask` (`plan_mode_respond`, `act_mode_respond`, `completion_result`).
	 */
	private findLatestCompletedTurn(
		messages: ClineMessage[],
		targetMode: "plan" | "act",
	): { text: string; latestTs: number } | undefined {
		const collectedTexts: string[] = []
		let latestTs = 0

		for (let i = messages.length - 1; i >= 0; i--) {
			const msg = messages[i]
			const isRelevantSay =
				msg.type === "say" &&
				(msg.say === "completion_result" ||
					msg.say === "text" ||
					(targetMode === "plan" && msg.say === "plan_completion_result"))
			const isRelevantAsk =
				msg.type === "ask" &&
				(msg.ask === "completion_result" ||
					(targetMode === "plan" && msg.ask === "plan_mode_respond") ||
					(targetMode === "act" && msg.ask === "act_mode_respond"))

			if ((isRelevantSay || isRelevantAsk) && msg.partial) {
				return undefined
			}

			if (msg.type === "say" && (msg.say === "task" || msg.say === "user_feedback")) {
				break
			}

			if ((isRelevantSay || isRelevantAsk) && !msg.partial && msg.text?.trim()) {
				const extracted =
					isRelevantAsk && (msg.ask === "plan_mode_respond" || msg.ask === "act_mode_respond")
						? unwrapRespondPayload(msg.text)
						: msg.text.trim()
				if (extracted) {
					if (msg.ts > latestTs) {
						latestTs = msg.ts
					}
					collectedTexts.unshift(extracted)
				}
			}
		}

		if (collectedTexts.length === 0 || latestTs === 0) {
			return undefined
		}

		const deduplicated = collectedTexts.filter((block, idx) => idx === 0 || block !== collectedTexts[idx - 1])
		return { text: deduplicated.join("\n\n"), latestTs }
	}

	isSyntheticOrInternalPrompt(prompt: string): boolean {
		const trimmed = prompt.trim()
		return (
			trimmed.startsWith(ORCHESTRATOR_REVISION_PREFIX) ||
			trimmed.startsWith("[TASK RESUMPTION]") ||
			trimmed === "compact context and resume task" ||
			trimmed.startsWith("compact context and resume task.")
		)
	}

	private getOpenAiEndpointAndKey(): { baseUrl: string; apiKey: string } {
		const cfg = this.options.getApiConfiguration()
		const baseUrl = (cfg.openAiBaseUrl?.trim() || `${this.gatewayBaseUrl}/v1`).replace(/\/+$/, "")
		const apiKey = cfg.openAiApiKey?.trim() || "local"
		return { baseUrl, apiKey }
	}

	private async fetchAvailableGatewayModels(): Promise<string[]> {
		const now = Date.now()
		if (this.cachedAvailableModels.length > 0 && now - this.cachedModelsFetchedAt < 15_000) {
			return this.cachedAvailableModels
		}

		const { baseUrl, apiKey } = this.getOpenAiEndpointAndKey()
		try {
			const controller = new AbortController()
			const timeout = setTimeout(() => controller.abort(), 3000)
			const res = await fetch(`${baseUrl}/models`, {
				headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : undefined,
				signal: controller.signal,
			})
			clearTimeout(timeout)
			if (!res.ok) {
				return this.cachedAvailableModels
			}
			const data = (await res.json()) as { data?: Array<{ id?: string; disabled?: boolean }> }
			const disabledSet = readLocalAiDisabledModels()
			const models: string[] = []
			for (const item of data?.data ?? []) {
				const id = item?.id?.trim()
				if (id && item?.disabled !== true && !disabledSet.has(id)) {
					models.push(id)
				}
			}
			if (models.length > 0) {
				this.cachedAvailableModels = models
				this.cachedModelsFetchedAt = now
			}
			return models
		} catch {
			return this.cachedAvailableModels
		}
	}

	private async callOrchestratorLlmVerbose(params: {
		modelId: string
		systemPrompt: string
		userPrompt: string
		maxTokens: number
		reasoningEffort?: OrchestratorReasoningEffort
		phaseLabel: "Prompt" | "Plan" | "Act"
		pushLog: (step: string) => void
		onLiveUpdate: (activity: string, thinking?: string, partialContent?: string) => void
	}): Promise<LlmCallMetrics> {
		const {
			modelId,
			systemPrompt,
			userPrompt,
			maxTokens,
			reasoningEffort = "low",
			phaseLabel,
			pushLog,
			onLiveUpdate,
		} = params
		const { baseUrl, apiKey } = this.getOpenAiEndpointAndKey()
		const currentStatus: OrchestratorStatus =
			phaseLabel === "Prompt"
				? "evaluating_prompt"
				: phaseLabel === "Act"
					? "evaluating_act"
					: "evaluating_plan"
		const estimatedPromptTokens = estimateTokens(systemPrompt) + estimateTokens(userPrompt)

		const vramCheckStart = Date.now()
		this.setStatus(currentStatus, "Checking VRAM...")
		onLiveUpdate("🔎 Checking VRAM status on LocalAI Gateway (`/_localai/status`)...")
		const previouslyLoadedModels = await this.probeLoadedGatewayModels()
		const vramCheckMs = Date.now() - vramCheckStart

		let vramUnloadBeforeMs = 0
		const otherLoaded = previouslyLoadedModels.filter((m) => m.toLowerCase() !== modelId.toLowerCase())
		if (otherLoaded.length > 0) {
			pushLog(`Unloading \`${otherLoaded.join(", ")}\` from VRAM...`)
			this.setStatus(currentStatus, "Unloading VRAM...")
			onLiveUpdate(`🧹 Unloading \`${otherLoaded.join(", ")}\` from VRAM...`)
			const unloadStart = Date.now()
			await this.freeGatewayVram()
			vramUnloadBeforeMs = Date.now() - unloadStart
			pushLog(`VRAM freed in **${formatDuration(vramUnloadBeforeMs)}**.`)
		} else if (previouslyLoadedModels.some((m) => m.toLowerCase() === modelId.toLowerCase())) {
			pushLog(`Model \`${modelId}\` already warm in VRAM (${formatDuration(vramCheckMs)}).`)
		} else {
			pushLog(`VRAM ready (${formatDuration(vramCheckMs)}).`)
		}

		const llmStartMs = Date.now()
		let firstTokenAtMs = 0
		let reasoningAccumulated = ""
		let contentAccumulated = ""
		let finishReason: string | undefined
		let lastUiUpdateMs = 0

		pushLog(
			`Request to \`${modelId}\` (~${estimatedPromptTokens} input tok, \`reasoning_effort="${reasoningEffort}"\`, \`max_tokens=${maxTokens}\`).`,
		)
		this.setStatus(currentStatus, `Loading ${modelId.split("-")[0]}...`)
		onLiveUpdate(
			`🚀 Loading \`${modelId}\` and prefilling prompt (~${estimatedPromptTokens} tok, Thinking: ${formatEffortLabel(reasoningEffort)})...`,
		)

		const abortController = new AbortController()
		this.activeAbortController = abortController
		const timeout = setTimeout(() => abortController.abort(), 180_000)

		const getCombinedThinking = () =>
			[reasoningAccumulated.trim(), extractThinkingBlocks(contentAccumulated)].filter(Boolean).join("\n").trim()
		const getCleanContent = () => stripThinkingBlocks(contentAccumulated)

		let tickerActive = true
		let isProbingProgress = false
		const tickerTimer = setInterval(async () => {
			if (!tickerActive) {
				return
			}
			if (firstTokenAtMs === 0) {
				if (isProbingProgress) {
					return
				}
				isProbingProgress = true
				try {
					const progress = await this.probePromptProgress()
					if (!tickerActive || firstTokenAtMs > 0) {
						return
					}
					const curElapsedSec = ((Date.now() - llmStartMs) / 1000).toFixed(1)
					const progressLabel =
						typeof progress === "number" ? `Prompt prefill: **${progress}%**` : "Loading GGUF / GPU Prefill"
					this.setStatus(
						currentStatus,
						typeof progress === "number"
							? `${phaseLabel} ${progress}% (${curElapsedSec}s)`
							: `${phaseLabel} (${curElapsedSec}s)...`,
					)
					lastUiUpdateMs = Date.now()
					onLiveUpdate(`⏳ Waiting for \`${modelId}\` — ${progressLabel} • \`${curElapsedSec}s\``)
				} finally {
					isProbingProgress = false
				}
			} else if (Date.now() - lastUiUpdateMs >= 500) {
				lastUiUpdateMs = Date.now()
				const genElapsedSec = Math.max(0.1, (Date.now() - firstTokenAtMs) / 1000)
				const thinkToks = estimateTokens(getCombinedThinking())
				const outToks = estimateTokens(getCleanContent())
				const tps = ((thinkToks + outToks) / genElapsedSec).toFixed(1)
				this.setStatus(
					currentStatus,
					`${phaseLabel}: ${outToks > 0 ? `JSON ${outToks}` : `Think ${thinkToks}`} (${tps} t/s)`,
				)
				onLiveUpdate(
					`Streaming from \`${modelId}\` — Thinking: \`~${thinkToks} tok\` • JSON: \`~${outToks} tok\` (**\`${tps} tok/s\`**)`,
					getCombinedThinking(),
					getCleanContent(),
				)
			}
		}, 650)

		let vramUnloadAfterMs = 0
		let didUnloadAfter = false
		try {
			const res = await fetch(`${baseUrl}/chat/completions`, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
				},
				body: JSON.stringify({
					model: modelId,
					messages: [
						{ role: "system", content: systemPrompt },
						{ role: "user", content: userPrompt },
					],
					temperature: 0.3,
					max_tokens: maxTokens,
					reasoning_effort: reasoningEffort,
					stream: true,
				}),
				signal: abortController.signal,
			})

			if (!res.ok) {
				const errText = await res.text().catch(() => "")
				throw new Error(`Orchestrator HTTP ${res.status}: ${errText.slice(0, 200)}`)
			}

			const contentType = res.headers?.get?.("content-type") ?? ""
			const hasReadableStream =
				res.body && typeof (res.body as ReadableStream<Uint8Array>).getReader === "function"

			if (hasReadableStream && contentType.includes("text/event-stream")) {
				const reader = (res.body as ReadableStream<Uint8Array>).getReader()
				const decoder = new TextDecoder("utf-8")
				let sseBuffer = ""
				let loggedJsonTransition = false

				const processSseLine = (rawLine: string) => {
					const line = rawLine.trim()
					if (!line.startsWith("data:")) {
						return
					}
					const dataPayload = line.slice(5).trim()
					if (!dataPayload || dataPayload === "[DONE]") {
						return
					}
					try {
						const chunk = JSON.parse(dataPayload) as {
							choices?: Array<{
								finish_reason?: string | null
								delta?: { content?: string; reasoning_content?: string }
								message?: { content?: string; reasoning_content?: string }
							}>
						}
						const choice = chunk.choices?.[0]
						if (choice?.finish_reason) {
							finishReason = choice.finish_reason
						}
						const delta = choice?.delta ?? choice?.message
						const reasoningPiece = delta?.reasoning_content ?? ""
						const contentPiece = delta?.content ?? ""

						if ((reasoningPiece || contentPiece) && firstTokenAtMs === 0) {
							firstTokenAtMs = Date.now()
							pushLog(`⚡ First token from \`${modelId}\` (TTFT: **${formatDuration(firstTokenAtMs - llmStartMs)}**).`)
						}
						if (reasoningPiece) {
							reasoningAccumulated += reasoningPiece
						}
						if (contentPiece) {
							if (!loggedJsonTransition && reasoningAccumulated.length > 0 && contentAccumulated.length === 0) {
								loggedJsonTransition = true
								pushLog(`🧠 Thinking finished (~${estimateTokens(reasoningAccumulated)} tok). Emitting JSON...`)
							}
							contentAccumulated += contentPiece
						}

						const now = Date.now()
						if (now - lastUiUpdateMs >= 400) {
							lastUiUpdateMs = now
							const genSec = Math.max(0.1, (now - (firstTokenAtMs || llmStartMs)) / 1000)
							const thinkToks = estimateTokens(getCombinedThinking())
							const outToks = estimateTokens(getCleanContent())
							const tps = ((thinkToks + outToks) / genSec).toFixed(1)
							this.setStatus(
								currentStatus,
								`${phaseLabel}: ${outToks > 0 ? `JSON ${outToks}` : `Think ${thinkToks}`} (${tps} t/s)`,
							)
							onLiveUpdate(
								`Streaming from \`${modelId}\` — Thinking: \`~${thinkToks} tok\` • JSON: \`~${outToks} tok\` (**\`${tps} tok/s\`**)`,
								getCombinedThinking(),
								getCleanContent(),
							)
						}
					} catch {
						// Ignore non-JSON SSE lines
					}
				}

				while (true) {
					const { done, value } = await reader.read()
					if (done) {
						sseBuffer += decoder.decode()
						if (sseBuffer.trim()) {
							for (const remainingLine of sseBuffer.split("\n")) {
								processSseLine(remainingLine)
							}
						}
						break
					}
					sseBuffer += decoder.decode(value, { stream: true })
					const lines = sseBuffer.split("\n")
					sseBuffer = lines.pop() ?? ""
					for (const rawLine of lines) {
						processSseLine(rawLine)
					}
				}
			} else {
				const data = (await res.json()) as {
					choices?: Array<{
						finish_reason?: string | null
						message?: { content?: string; reasoning_content?: string }
					}>
				}
				firstTokenAtMs = Date.now()
				const choice = data?.choices?.[0]
				finishReason = choice?.finish_reason ?? undefined
				reasoningAccumulated = choice?.message?.reasoning_content ?? ""
				contentAccumulated = choice?.message?.content ?? ""
				pushLog(`Response received from \`${modelId}\` in **${formatDuration(firstTokenAtMs - llmStartMs)}**.`)
			}

			const llmEndMs = Date.now()
			const totalLlmMs = Math.max(1, llmEndMs - llmStartMs)
			const ttftMs = firstTokenAtMs > 0 ? Math.max(1, firstTokenAtMs - llmStartMs) : totalLlmMs
			const generationMs = firstTokenAtMs > 0 ? Math.max(1, llmEndMs - firstTokenAtMs) : totalLlmMs
			const finalThinking = getCombinedThinking()
			const finalCleanContent = getCleanContent()
			const estimatedThinkingTokens = finalThinking ? estimateTokens(finalThinking) : 0
			const estimatedOutputTokens = finalCleanContent ? estimateTokens(finalCleanContent) : 0
			const tokensPerSecond =
				Math.max(1, estimatedThinkingTokens + estimatedOutputTokens) / Math.max(0.05, generationMs / 1000)

			if (hasReadableStream && contentType.includes("text/event-stream")) {
				pushLog(
					`Generation finished in **${formatDuration(generationMs)}** (~${estimatedThinkingTokens} think + ~${estimatedOutputTokens} JSON @ **${tokensPerSecond.toFixed(1)} tok/s**).`,
				)
			}

			tickerActive = false
			clearInterval(tickerTimer)
			this.setStatus(currentStatus, "Releasing VRAM...")
			onLiveUpdate(`🧹 Releasing VRAM for \`${modelId}\`...`, finalThinking, finalCleanContent)
			const unloadAfterStart = Date.now()
			await this.freeGatewayVram()
			didUnloadAfter = true
			vramUnloadAfterMs = Date.now() - unloadAfterStart
			pushLog(`VRAM released in **${formatDuration(vramUnloadAfterMs)}**.`)

			return {
				content: finalCleanContent,
				thinkingText: finalThinking,
				reasoningEffort,
				vramCheckMs,
				vramUnloadBeforeMs,
				vramUnloadAfterMs,
				previouslyLoadedModels,
				ttftMs,
				generationMs,
				totalLlmMs,
				estimatedPromptTokens,
				estimatedOutputTokens,
				estimatedThinkingTokens,
				tokensPerSecond,
				finishReason,
			}
		} finally {
			tickerActive = false
			clearInterval(tickerTimer)
			clearTimeout(timeout)
			if (this.activeAbortController === abortController) {
				this.activeAbortController = undefined
			}
			if (!didUnloadAfter) {
				await this.freeGatewayVram()
			}
		}
	}

	private async probeLoadedGatewayModels(): Promise<string[]> {
		try {
			const controller = new AbortController()
			const timeout = setTimeout(() => controller.abort(), 2000)
			const statusRes = await fetch(`${this.gatewayBaseUrl}/_localai/status`, { signal: controller.signal })
			clearTimeout(timeout)
			if (statusRes.ok) {
				const statusData = (await statusRes.json()) as { llm?: { loadedModels?: string[] } }
				return statusData?.llm?.loadedModels ?? []
			}
		} catch { }
		return []
	}

	private async probePromptProgress(): Promise<number | null> {
		try {
			const controller = new AbortController()
			const timeout = setTimeout(() => controller.abort(), 500)
			const res = await fetch(`${this.gatewayBaseUrl}/_localai/prompt_progress`, { signal: controller.signal })
			clearTimeout(timeout)
			if (res.ok) {
				const data = (await res.json()) as { progress?: number | null }
				if (typeof data?.progress === "number") {
					return data.progress
				}
			}
		} catch { }
		return null
	}

	private async freeGatewayVram(): Promise<void> {
		try {
			const controller = new AbortController()
			const timeout = setTimeout(() => controller.abort(), 4000)
			await fetch(`${this.gatewayBaseUrl}/_localai/free-vram`, { method: "POST", signal: controller.signal })
			clearTimeout(timeout)
		} catch { }
	}

	private emitInfoMessage(ts: number, text: string): void {
		const existing = this.persistedCards.get(ts)
		const phase: PersistedOrchestratorCard["phase"] =
			existing?.phase ??
			(this.status === "evaluating_prompt"
				? "prompt"
				: this.status === "evaluating_act" || this.isEvaluatingAct
					? "act"
					: "plan")
		this.persistedCards.set(ts, { ts, phase, text })
		this.persistCardsToDisk()
		this.options.emitMessage({ ts, type: "say", say: "info", text, partial: false })
		void this.options.postStateToWebview().catch(() => { })
	}

	private persistCardsToDisk(): void {
		try {
			const sessionId = this.options.getSessionId?.()?.trim()
			if (!sessionId || this.persistedCards.size === 0) {
				return
			}
			const sessionDir = path.join(resolveSessionDataDir(), sessionId)
			if (!fs.existsSync(sessionDir)) {
				return
			}
			const filePath = path.join(sessionDir, `${sessionId}.orchestrator.json`)
			const cards = Array.from(this.persistedCards.values()).sort((a, b) => a.ts - b.ts)
			fs.writeFileSync(filePath, JSON.stringify({ version: 1, sessionId, cards }, null, 2), "utf8")
		} catch (err) {
			Logger.warn("[LocalAiOrchestrator] Failed to persist orchestrator cards:", err)
		}
	}

	private setLastEvaluation(evaluation: OrchestratorLastEvaluation): void {
		this.lastEvaluation = evaluation
		this.notifyStateChanged()
	}

	private setStatus(status: OrchestratorStatus, phaseDetail?: string): void {
		if (this.status === status && this.phaseDetail === phaseDetail) {
			return
		}
		this.status = status
		this.phaseDetail = status === "idle" ? undefined : phaseDetail
		this.notifyStateChanged()
	}

	private notifyStateChanged(): void {
		try {
			this.options.onStateChanged?.(this.getStateSnapshot())
		} catch (err) {
			Logger.warn("[LocalAiOrchestrator] onStateChanged callback error:", err)
		}
	}
}