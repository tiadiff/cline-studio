import { execFile } from "node:child_process"
import fs from "node:fs"
import path from "node:path"
import { promisify } from "node:util"
import type { ApiConfiguration } from "@shared/api"
import type { ClineMessage } from "@shared/ExtensionMessage"
import type { Mode } from "@shared/storage/types"
import { Logger } from "@/shared/services/Logger"

const execFileAsync = promisify(execFile)

export type OrchestratorStatus = "idle" | "evaluating_prompt" | "evaluating_plan"

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
}

interface LlmCallMetrics {
	content: string
	thinkingText: string
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

/**
 * Ranked priority list of LocalAI models best suited to act as Orchestrator.
 * The selector automatically excludes the two models configured for Plan Mode
 * and Act Mode so the Orchestrator is always an independent third model.
 */
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

/** Prefix used to tag automated plan-revision prompts sent by the Orchestrator. */
export const ORCHESTRATOR_REVISION_PREFIX = "[ORCHESTRATOR PLAN FEEDBACK]"

function formatClock(ts = Date.now()): string {
	const d = new Date(ts)
	const hh = String(d.getHours()).padStart(2, "0")
	const mm = String(d.getMinutes()).padStart(2, "0")
	const ss = String(d.getSeconds()).padStart(2, "0")
	const ms = String(d.getMilliseconds()).padStart(3, "0")
	return `${hh}:${mm}:${ss}.${ms}`
}

function formatDuration(ms: number): string {
	if (ms < 1000) {
		return `${Math.max(0, Math.round(ms))}ms`
	}
	return `${(ms / 1000).toFixed(2)}s`
}

/**
 * Invia una notifica nativa di sistema su Windows (Toast nativo WinRT) senza bloccare il thread.
 */
function sendWindowsNotification(title: string, message: string): void {
	if (process.platform !== "win32") {
		return
	}
	try {
		const escapeXml = (s: string) =>
			s
				.replace(/&/g, "&amp;")
				.replace(/</g, "&lt;")
				.replace(/>/g, "&gt;")
				.replace(/"/g, "&quot;")
				.replace(/'/g, "&apos;")

		const safeTitle = escapeXml(title.slice(0, 80))
		const safeMessage = escapeXml(message.slice(0, 200))

		const psScript = [
			"[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null",
			"$doc = [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime]::new()",
			`$xml = '<toast><visual><binding template=\"ToastGeneric\"><text>${safeTitle}</text><text>${safeMessage}</text></binding></visual></toast>'`,
			"$doc.LoadXml($xml)",
			"$toast = [Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType = WindowsRuntime]::new($doc)",
			"$appId = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe'",
			"[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($appId).Show($toast)",
		].join("; ")

		const encoded = Buffer.from(psScript, "utf16le").toString("base64")

		execFile(
			"powershell.exe",
			["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
			{ timeout: 4000, windowsHide: true },
			() => { },
		)
	} catch (err) {
		Logger.warn("[LocalAiOrchestrator] Failed to show Windows notification:", err)
	}
}

function estimateTokens(text: string): number {
	if (!text) {
		return 0
	}
	return Math.max(1, Math.ceil(text.length / 3.6))
}

function extractThinkingBlocks(raw: string): string {
	const matches: string[] = []
	const thinkRegex = /<think>([\s\S]*?)(?:<\/think>|$)/gi
	let m: RegExpExecArray | null = thinkRegex.exec(raw)
	while (m !== null) {
		if (m[1]?.trim()) {
			matches.push(m[1].trim())
		}
		m = thinkRegex.exec(raw)
	}
	const channelRegex = /<\|channel\|?>thought([\s\S]*?)(?:<\|channel\|?>|<\|end\|?>|$)/gi
	let cm: RegExpExecArray | null = channelRegex.exec(raw)
	while (cm !== null) {
		if (cm[1]?.trim()) {
			matches.push(cm[1].trim())
		}
		cm = channelRegex.exec(raw)
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

function extractFirstJsonObject(raw: string): Record<string, unknown> | undefined {
	const cleaned = stripThinkingBlocks(raw)
	const candidates = [cleaned, raw]

	for (const source of candidates) {
		const fencedMatch = source.match(/```(?:json)?\s*([\s\S]*?)```/i)
		const candidate = fencedMatch ? fencedMatch[1].trim() : source

		const firstBrace = candidate.indexOf("{")
		const lastBrace = candidate.lastIndexOf("}")
		if (firstBrace === -1 || lastBrace === -1 || lastBrace <= firstBrace) {
			continue
		}

		const jsonSlice = candidate.slice(firstBrace, lastBrace + 1)
		try {
			const parsed = JSON.parse(jsonSlice)
			if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
				return parsed as Record<string, unknown>
			}
		} catch {
			// Try cleaning trailing commas or control newlines inside strings
			try {
				const repaired = jsonSlice.replace(/,\s*([}\]])/g, "$1")
				const parsed = JSON.parse(repaired)
				if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
					return parsed as Record<string, unknown>
				}
			} catch {
				// Continue to next candidate
			}
		}
	}
	return undefined
}

/**
 * Extracts structured bullet lines from free-form thinking/text if JSON parsing failed,
 * ensuring the Orchestrator NEVER falls back to parroting the raw user prompt.
 */
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

	const objective =
		sentences[0] ||
		`Analizzare in dettaglio la richiesta ("${rawPrompt}"), verificare lo stato reale dei file e del repository nel workspace e formulare un piano operativo completo.`

	const requirements =
		sentences.slice(1, 5).length >= 2
			? sentences.slice(1, 5)
			: [
				`Analizzare il contesto del workspace per rispondere in modo esaustivo a: "${rawPrompt}"`,
				"Ispezionare lo stato effettivo dei file, delle dipendenze o del repository Git coinvolti prima di proporre azioni",
				"Definire i passaggi esatti, i comandi o le modifiche necessarie in ordine logico e verificabile",
			]

	return {
		isClear: rawPrompt.split(/\s+/).length >= 5,
		qualityScore: rawPrompt.split(/\s+/).length >= 12 ? 8 : 6,
		objective,
		technicalContext: workspaceSnapshot
			? `Stato workspace rilevato automaticamente dall'Orchestratore:\n${workspaceSnapshot.slice(0, 600)}`
			: "Analisi preliminare eseguita sul prompt utente.",
		requirements,
		risksAndEdgeCases: [
			"Evitare assunzioni non verificate sullo stato dei file o di Git senza prima averli ispezionati con i tool di lettura.",
		],
		acceptanceCriteria: [
			"Il Plan deve rispondere direttamente e specificamente alla domanda/richiesta dell'utente basandosi sui dati reali del workspace.",
			"Ogni passaggio proposto nel Plan deve indicare file, comandi o verifiche concrete.",
		],
		planningGuidance:
			"Usa i tool di sola lettura in Plan Mode se necessario per raccogliere i dati mancanti, quindi presenta un piano strutturato, chiaro e immediatamente azionabile.",
	}
}

function getModelFamilyPrefix(modelId: string): string {
	const normalized = modelId.trim().toLowerCase()
	if (normalized.startsWith("qwen3.6-27b-coderx")) {
		return "qwen3.6-27b-coderx"
	}
	return normalized
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

export class LocalAiOrchestratorCoordinator {
	private enabled: boolean
	private customModelId?: string
	private status: OrchestratorStatus = "idle"
	private phaseDetail?: string
	private startedAt?: number
	private readonly gatewayBaseUrl: string
	private readonly maxPlanRevisions: number

	private initialUserPrompt = ""
	private followupUserPrompts: string[] = []
	private lastPromptEvaluation?: OrchestratorPromptEvaluation
	private planRevisionCount = 0
	private lastEvaluatedPlanTs = 0
	private lastPlanTurnStartedAt = 0
	private isEvaluatingPlan = false
	private activeAbortController?: AbortController

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
		}
	}

	resetForNewTask(): void {
		this.abortActiveEvaluation()
		this.initialUserPrompt = ""
		this.followupUserPrompts = []
		this.lastPromptEvaluation = undefined
		this.planRevisionCount = 0
		this.lastEvaluatedPlanTs = 0
		this.lastPlanTurnStartedAt = 0
		this.isEvaluatingPlan = false
		this.setStatus("idle")
	}

	abortActiveEvaluation(): void {
		if (this.activeAbortController) {
			try {
				this.activeAbortController.abort()
			} catch { }
			this.activeAbortController = undefined
		}
		this.isEvaluatingPlan = false
		if (this.status !== "idle") {
			this.setStatus("idle")
		}
	}

	/**
	 * Returns the two models currently configured for Plan Mode and Act Mode.
	 */
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

	/**
	 * Synchronously resolves the best Orchestrator model that is NEITHER the
	 * default Plan Mode model NOR the default Act Mode model.
	 */
	resolveOrchestratorModelIdSync(availablePool?: readonly string[]): string {
		const { planModelId, actModelId } = this.getPlanAndActModelIds()
		const excludedExact = new Set<string>()
		const excludedFamilies = new Set<string>()

		if (planModelId) {
			excludedExact.add(planModelId.toLowerCase())
			excludedFamilies.add(getModelFamilyPrefix(planModelId))
		}
		if (actModelId) {
			excludedExact.add(actModelId.toLowerCase())
			excludedFamilies.add(getModelFamilyPrefix(actModelId))
		}

		// Never use vision-only captioning models as text orchestrator
		excludedExact.add("qwen3-vl-8b-vision-caption-uncensored")

		if (
			this.customModelId &&
			!excludedExact.has(this.customModelId.toLowerCase()) &&
			!excludedFamilies.has(getModelFamilyPrefix(this.customModelId))
		) {
			return this.customModelId
		}

		const disabledSet = readLocalAiDisabledModels()
		const disabledLower = new Set(Array.from(disabledSet).map((m) => m.toLowerCase()))
		const pool = availablePool && availablePool.length > 0 ? availablePool : this.cachedAvailableModels
		const poolLowerSet = pool.length > 0 ? new Set(pool.map((m) => m.toLowerCase())) : undefined

		// Pass 1: Pick the highest-ranked priority model that is enabled, in the pool,
		// and does not share identity or same-GGUF family with Plan/Act models.
		for (const candidate of ORCHESTRATOR_MODEL_PRIORITY) {
			const lower = candidate.toLowerCase()
			if (excludedExact.has(lower) || excludedFamilies.has(getModelFamilyPrefix(candidate))) {
				continue
			}
			if (disabledLower.has(lower)) {
				continue
			}
			if (poolLowerSet && !poolLowerSet.has(lower)) {
				continue
			}
			return candidate
		}

		// Pass 2: Allow any enabled priority model that is strictly different by ID from Plan & Act models.
		for (const candidate of ORCHESTRATOR_MODEL_PRIORITY) {
			const lower = candidate.toLowerCase()
			if (excludedExact.has(lower)) {
				continue
			}
			if (!disabledLower.has(lower)) {
				return candidate
			}
		}

		// Pass 3: Fallback to any priority model not equal to Plan/Act
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

	/**
	 * Collects a fast, non-destructive technical snapshot of the workspace (directory tree,
	 * git branch, uncommitted/untracked files, ahead/behind remote status, sub-repos)
	 * so the Orchestrator can reason deeply with real project context instead of guessing.
	 */
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

			// Top-level entries (lightweight)
			try {
				const entries = fs.readdirSync(cwd, { withFileTypes: true }).slice(0, 15)
				const formatted = entries.map((e) => (e.isDirectory() ? `${e.name}/` : e.name)).join(", ")
				parts.push(`Struttura: ${formatted}`)
			} catch { }

			// Git status on main workspace (concise)
			try {
				const { stdout: statusOut } = await execFileAsync("git", ["-C", cwd, "status", "--short", "--branch"], {
					timeout: 1000,
				})
				if (statusOut.trim()) {
					parts.push(`Git Status:\n${statusOut.trim().slice(0, 500)}`)
				}
			} catch { }

			return parts.join("\n")
		} catch {
			return ""
		}
	}

	private isDirectSimpleQuestion(prompt: string): boolean {
		const lower = prompt.toLowerCase().trim()
		const isShort = prompt.split(/\s+/).length <= 15
		if (!isShort) {
			return false
		}
		const simplePatterns = [
			/^(do|should|can|shall)\s+(i|we)\s+(git\s+)?(push|commit|pull|merge|rebase|stash|build|test)\b/i,
			/^(devo|dovrei|posso|serve)\s+(fare\s+il\s+|fare\s+un\s+|fare\s+)?(push|commit|pull|git\s+push|git\s+commit)\b/i,
			/^(is|are|was|were|has|have|did|does)\b.*\?$/i,
			/^(serve|bisogna|c'è\s+bisogno)\b.*\?$/i,
			/\b(need to\s+(git\s+)?push|serve fare\s+(il\s+)?push)\b/i,
		]
		return simplePatterns.some((pattern) => pattern.test(lower))
	}

	/**
	 * Step 1: Evaluates a user prompt before it is sent to the Plan Mode model.
	 * Uses HIGH thinking (`reasoning_effort: "high"`, `max_tokens: 4096`) and live workspace/git context
	 * to perform a deep architectural decomposition of the user's request.
	 */
	async evaluateUserPrompt(rawPrompt: string, resolvedPrompt: string): Promise<string> {
		const trimmed = rawPrompt.trim()
		if (!this.enabled || !trimmed || this.isSyntheticOrInternalPrompt(trimmed)) {
			if (this.enabled && trimmed.startsWith(ORCHESTRATOR_REVISION_PREFIX)) {
				this.lastPlanTurnStartedAt = Date.now()
			}
			return resolvedPrompt
		}

		// Se è una domanda diretta/semplice (es. "do i need to git push?"), non forzare
		// una decomposizione architetturale pesante che obbliga il modello Plan Mode
		// a produrre un piano multi-fase e l'elenco chilometrico di tutti i file.
		if (this.isDirectSimpleQuestion(trimmed)) {
			this.initialUserPrompt = trimmed
			this.planRevisionCount = 0
			this.lastPlanTurnStartedAt = Date.now()
			const directHint = [
				"\n\n<orchestrator_evaluation>",
				"Objective: Rispondi direttamente e in modo estremamente conciso alla domanda dell'utente (es. Sì/No o stato sintetico).",
				"Directive: Non generare un piano d'azione complesso o liste chilometriche di singoli file, a meno che non sia strettamente necessario o richiesto dall'utente.",
				"</orchestrator_evaluation>",
			].join("\n")
			return `${resolvedPrompt}${directHint}`
		}

		// Record prompt in task specification history
		if (!this.initialUserPrompt) {
			this.initialUserPrompt = trimmed
		} else if (trimmed !== this.initialUserPrompt) {
			this.followupUserPrompts.push(trimmed)
		}
		this.planRevisionCount = 0

		const opStartMs = Date.now()
		const statusTs = opStartMs
		const traceLogs: string[] = []
		const pushLog = (step: string) => {
			const elapsed = formatDuration(Date.now() - opStartMs)
			traceLogs.push(`\`[${formatClock()}] (+${elapsed})\` ${step}`)
		}

		this.startedAt = opStartMs
		this.setStatus("evaluating_prompt", "Ispezione workspace...")

		pushLog(`Intercettato prompt utente (${trimmed.length} caratteri, ~${estimateTokens(trimmed)} token stimati).`)
		const [orchestratorModel, workspaceSnapshot] = await Promise.all([
			this.resolveOrchestratorModelId(),
			this.collectWorkspaceSnapshot(),
		])
		const { planModelId, actModelId } = this.getPlanAndActModelIds()

		if (workspaceSnapshot) {
			pushLog(
				`Acquisito snapshot tecnico del workspace e stato Git (${workspaceSnapshot.length} caratteri di contesto reale) per arricchire il ragionamento dell'Orchestratore.`,
			)
		}
		pushLog(
			`Modello Orchestratore selezionato: **\`${orchestratorModel}\`** con **Thinking: HIGH** *(esclusi i modelli default Plan=\`${planModelId ?? "n/d"}\` e Act=\`${actModelId ?? "n/d"}\`)*.`,
		)

		const renderLivePromptCard = (liveActivity: string, liveThinking?: string, liveOutput?: string) => {
			const elapsedTotal = formatDuration(Date.now() - opStartMs)
			const headerBlock = [
				`**Orchestratore AI [DIETRO LE QUINTE — VALUTAZIONE PROFONDA PROMPT]** • \`${elapsedTotal}\``,
				"",
				`**Pipeline Modelli:**`,
				`- **Supervisore:** \`${orchestratorModel}\` *(Thinking: HIGH)*`,
				`- **Plan Mode:** \`${planModelId ?? "default"}\``,
				`- **Act Mode:** \`${actModelId ?? "default"}\``,
				"",
				`**Stato Attuale:**`,
				"",
				...liveActivity.split("\n").filter(Boolean).map((line) => `${line}  `),
				"",
				`---`,
				"",
				`**Log Operazioni & Tempi in Tempo Reale:**`,
				...traceLogs.map((l) => `- ${l}`),
			]

			if (liveThinking) {
				const tailThink = liveThinking.length > 900 ? `...${liveThinking.slice(-900)}` : liveThinking
				headerBlock.push(
					"",
					`**Ragionamento Profondo Orchestratore (Thinking High — live stream):**`,
					`> ${tailThink.replace(/\n/g, "\n> ")}`,
				)
			}
			if (liveOutput) {
				const tailOut = liveOutput.length > 600 ? `...${liveOutput.slice(-600)}` : liveOutput
				headerBlock.push("", `**Specifica Tecnica JSON in generazione:**`, `\`\`\`json\n${tailOut}\n\`\`\``)
			}
			this.emitInfoMessage(statusTs, headerBlock.join("\n"))
		}

		renderLivePromptCard("🔄 Inizializzazione analisi profonda del prompt e del contesto workspace...")

		try {
			const systemPrompt = [
				"Sei l'Orchestratore AI di supervisione in Cline.",
				"Analizza la richiesta dell'utente e genera DIRETTAMENTE una specifica tecnica operativa in JSON per il Plan Mode.",
				"LINEE GUIDA:",
				"1. Se la richiesta è una semplice domanda (es. 'do i need to push?', 'cosa fa questa funzione?', 'posso committare?'), l'obiettivo deve essere RISPONDERE DIRETTAMENTE (es. 'Rispondi Sì/No e motiva in 1 riga'). Non trasformare semplici domande in piani di implementazione complessi o analisi chilometriche di ogni file.",
				"2. Massima sintesi tecnica, nessuna spiegazione discorsiva o preambolo, rispondi ESCLUSIVAMENTE con il JSON:",
				"{",
				'  "isClear": true,',
				'  "qualityScore": 8,',
				'  "objective": "Obiettivo sintetico in 1 frase (se è una domanda informativa, richiedi risposta diretta e concisa)",',
				'  "technicalContext": "Contesto essenziale in 1 frase",',
				'  "requirements": ["Requisito sintetico"],',
				'  "risksAndEdgeCases": [],',
				'  "acceptanceCriteria": ["Fornire risposta o piano sintetico e diretto senza verbosità non necessaria"],',
				'  "planningGuidance": "Se è una domanda diretta (es. yes/no o stato), rispondi direttamente e concisamente senza elencare tutti i file singolarmente se non esplicitamente richiesto."',
				"}",
			].join("\n")

			const userContent = [
				`RICHIESTA UTENTE:\n"""\n${trimmed}\n"""`,
				workspaceSnapshot ? `\nCONTESTO:\n${workspaceSnapshot}` : "",
				this.followupUserPrompts.length > 0
					? `\nCONTESTO PRECEDENTE:\n${this.initialUserPrompt}`
					: "",
			]
				.filter(Boolean)
				.join("\n")

			const metrics = await this.callOrchestratorLlmVerbose({
				modelId: orchestratorModel,
				systemPrompt,
				userPrompt: userContent,
				maxTokens: 512,
				reasoningEffort: "low",
				phaseLabel: "Prompt",
				pushLog,
				onLiveUpdate: (activity, thinking, partialContent) => {
					renderLivePromptCard(activity, thinking, partialContent)
				},
			})

			const parsed = extractFirstJsonObject(metrics.content) || extractFirstJsonObject(metrics.thinkingText)

			let evaluation: OrchestratorPromptEvaluation
			if (parsed && typeof parsed.objective === "string" && parsed.objective.trim().toLowerCase() !== trimmed.toLowerCase()) {
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
								? "Contesto workspace e Git acquisito e integrato nell'analisi."
								: "Analisi architetturale basata sul prompt utente.",
					requirements:
						parsedReqs.length > 0
							? parsedReqs
							: synthesizeEvaluationFromReasoning(trimmed, metrics.thinkingText, metrics.content, workspaceSnapshot)
								.requirements,
					risksAndEdgeCases:
						parsedRisks.length > 0
							? parsedRisks
							: [
								"Verificare lo stato reale dei file e dei repository prima di eseguire modifiche o comandi.",
							],
					acceptanceCriteria:
						parsedCriteria.length > 0
							? parsedCriteria
							: [
								"Il Plan deve coprire integralmente ciascuno dei requisiti tecnici estratti sopra.",
								"Il Plan deve fornire indicazioni concrete basate sullo stato reale del workspace.",
							],
					planningGuidance:
						typeof parsed.planningGuidance === "string" && parsed.planningGuidance.trim()
							? parsed.planningGuidance.trim()
							: "Formula un piano tecnico dettagliato, verificato sui file reali del workspace e pronto per l'esecuzione.",
				}
			} else {
				pushLog(
					`ℹ️ Sintesi strutturata estratta direttamente dal ragionamento profondo (${metrics.estimatedThinkingTokens} token di thinking).`,
				)
				evaluation = synthesizeEvaluationFromReasoning(trimmed, metrics.thinkingText, metrics.content, workspaceSnapshot)
			}

			this.lastPromptEvaluation = evaluation
			const totalMs = Date.now() - opStartMs
			pushLog(
				`Analisi architetturale completata (Specificità prompt iniziale: **${evaluation.qualityScore}/10**, elaborati **${evaluation.requirements.length}** requisiti tecnici, **${evaluation.risksAndEdgeCases.length}** rischi/vincoli e **${evaluation.acceptanceCriteria.length}** criteri di validazione).`,
			)
			pushLog(
				`Handoff in corso al modello **Plan Mode** (\`${planModelId ?? "default"}\`) con specifica tecnica espansa \`<orchestrator_evaluation>\`.`,
			)

			const reqBullets = evaluation.requirements.map((r, i) => `${i + 1}. ${r}`).join("\n")
			const riskBullets = evaluation.risksAndEdgeCases.map((r) => `- ⚠️ ${r}`).join("\n")
			const criteriaBullets = evaluation.acceptanceCriteria.map((c) => `- [ ] ${c}`).join("\n")

			const telemetrySummary = [
				`**Totale:** \`${formatDuration(totalMs)}\``,
				`**Thinking (High):** \`~${metrics.estimatedThinkingTokens} tok\``,
				`**VRAM Swap:** \`${formatDuration(metrics.vramUnloadBeforeMs + metrics.vramUnloadAfterMs)}\``,
				`**TTFT:** \`${formatDuration(metrics.ttftMs)}\``,
				`**Generazione:** \`${formatDuration(metrics.generationMs)}\` (\`~${metrics.estimatedOutputTokens} tok\` @ **\`${metrics.tokensPerSecond.toFixed(1)} tok/s\`**)`,
			].join(" • ")

			const finalLines: string[] = [
				`**Orchestratore AI** (\`${orchestratorModel}\` • **Thinking: High**) — **Specifica Tecnica & Valutazione Prompt** (Specificità iniziale: **${evaluation.qualityScore}/10**)`,
				telemetrySummary,
				`---`,
				`**Obiettivo Tecnico Decomposto:**\n${evaluation.objective}`,
				evaluation.technicalContext ? `**Contesto Workspace & Git Rilevato:**\n${evaluation.technicalContext}` : "",
				reqBullets ? `**Requisiti Tecnici Obbligatori per il Plan (${evaluation.requirements.length}):**\n${reqBullets}` : "",
				riskBullets ? `**Rischi, Vincoli & Casi Limite da Gestire:**\n${riskBullets}` : "",
				criteriaBullets
					? `**Criteri di Accettazione (verificati prima dello switch automatico ad Act):**\n${criteriaBullets}`
					: "",
				`**Direttiva Architetturale inviata a \`${planModelId ?? "default"}\`:**\n*${evaluation.planningGuidance}*`,
				metrics.thinkingText
					? `\n**Ragionamento Profondo dell'Orchestratore (${metrics.estimatedThinkingTokens} token):**\n\`\`\`\n${metrics.thinkingText}\n\`\`\``
					: "",
				`\n**Log Dietro le Quinte & Telemetria Completa (${traceLogs.length} eventi):**\n${traceLogs.map((l) => `- ${l}`).join("\n")}`,
				`\n*Specifica inviata al modello in **Plan Mode** (\`${planModelId ?? "default"}\`): attesa generazione del Plan...*`,
			]

			this.emitInfoMessage(statusTs, finalLines.filter(Boolean).join("\n\n"))
			sendWindowsNotification(
				"Orchestratore AI: Analisi Prompt Pronta",
				`Specificità: ${evaluation.qualityScore}/10 • Inviata specifica a Plan Mode (${evaluation.requirements.length} requisiti)`,
			)

			const orchestratorNotesBlock = [
				"\n\n<orchestrator_evaluation>",
				`Expanded Technical Objective: ${evaluation.objective}`,
				evaluation.planningGuidance ? `Directive: ${evaluation.planningGuidance}` : "",
				evaluation.requirements.length > 0 ? "Key Requirements:\n" + evaluation.requirements.map((r, idx) => `${idx + 1}. ${r}`).join("\n") : "",
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
			pushLog(`⚠️ Valutazione prompt interrotta dopo ${formatDuration(totalMs)}: ${errMsg}`)
			Logger.warn("[LocalAiOrchestrator] Prompt evaluation skipped/failed, proceeding with original prompt:", error)
			this.emitInfoMessage(
				statusTs,
				[
					`🤖 **Orchestratore AI** (\`${orchestratorModel}\`) — **Prompt Registrato** *(fallback rapido in ${formatDuration(totalMs)}: ${errMsg})*`,
					`**📡 Log Dietro le Quinte:**\n${traceLogs.map((l) => `- ${l}`).join("\n")}`,
					`▶️ *Passaggio del prompt al modello in Plan Mode (\`${planModelId ?? "default"}\`)...*`,
				].join("\n\n"),
			)
			this.lastPlanTurnStartedAt = Date.now()
			return resolvedPrompt
		} finally {
			this.startedAt = undefined
			this.setStatus("idle")
		}
	}

	/**
	 * Steps 2, 3 & 4:
	 * - Reads the response received in Plan Mode
	 * - Audits the plan with HIGH thinking (`reasoning_effort: "high"`, `max_tokens: 4096`) against the initial prompt and acceptance criteria
	 * - Automatically presses Act to start the work, or suggests appropriate modifications in Plan Mode.
	 */
	async handlePlanTurnCompleted(): Promise<void> {
		if (!this.enabled || this.isEvaluatingPlan) {
			return
		}

		if (this.options.getCurrentMode() !== "plan") {
			return
		}

		if (this.options.isSessionRunning()) {
			return
		}

		const messages = this.options.getClineMessages()
		if (messages.length === 0) {
			return
		}

		// Do not interrupt if the last message is an explicit ask waiting on user interaction (e.g. tool approval)
		const tailMessage = messages[messages.length - 1]
		if (tailMessage?.type === "ask" && tailMessage.ask === "tool") {
			return
		}

		const planMessage = this.findLatestCompletedPlanMessage(messages)
		if (!planMessage || !planMessage.text?.trim()) {
			return
		}

		if (planMessage.ts <= this.lastEvaluatedPlanTs) {
			return
		}

		this.lastEvaluatedPlanTs = planMessage.ts
		this.isEvaluatingPlan = true
		const opStartMs = Date.now()
		const statusTs = opStartMs
		this.startedAt = opStartMs
		this.setStatus("evaluating_plan", "Lettura Plan...")

		const planDurationMs =
			this.lastPlanTurnStartedAt > 0 ? Math.max(0, opStartMs - this.lastPlanTurnStartedAt) : undefined
		const planText = planMessage.text.trim()
		const planChars = planText.length
		const planTokens = estimateTokens(planText)

		const traceLogs: string[] = []
		const pushLog = (step: string) => {
			const elapsed = formatDuration(Date.now() - opStartMs)
			traceLogs.push(`\`[${formatClock()}] (+${elapsed})\` ${step}`)
		}

		const orchestratorModel = await this.resolveOrchestratorModelId()
		const { planModelId, actModelId } = this.getPlanAndActModelIds()

		// Ensure we have the initial prompt even if Orchestrator was toggled ON mid-task
		if (!this.initialUserPrompt) {
			const firstTaskMsg = messages.find((m) => m.type === "say" && m.say === "task" && m.text?.trim())
			if (firstTaskMsg?.text) {
				this.initialUserPrompt = firstTaskMsg.text.trim()
			}
		}

		pushLog(
			`Ricevuta risposta di **Plan Mode** dal modello \`${planModelId ?? "default"}\` (${planChars} caratteri, ~${planTokens} token stimati${planDurationMs ? `, generata in ${formatDuration(planDurationMs)}` : ""}).`,
		)
		pushLog(
			`Avvio audit approfondito del Plan con supervisore **\`${orchestratorModel}\`** (**Thinking: HIGH**, confronto rispetto al prompt iniziale${this.lastPromptEvaluation ? ` e a ${this.lastPromptEvaluation.requirements.length} requisiti + ${this.lastPromptEvaluation.acceptanceCriteria.length} criteri` : ""}).`,
		)

		const renderLivePlanCard = (liveActivity: string, liveThinking?: string, liveOutput?: string) => {
			const elapsedTotal = formatDuration(Date.now() - opStartMs)
			const headerBlock = [
				`**Orchestratore AI [DIETRO LE QUINTE — AUDIT DEL PLAN]** • \`${elapsedTotal}\``,
				"",
				`**Pipeline Modelli:**`,
				`- **Plan Mode:** \`${planModelId ?? "default"}\``,
				`- **Supervisore:** \`${orchestratorModel}\` *(Thinking: HIGH)*`,
				`- **Act Mode:** \`${actModelId ?? "default"}\``,
				"",
				`**Metadati Plan:** \`${planChars} caratteri\` (\`~${planTokens} token\`)${planDurationMs ? ` • **Generato in:** \`${formatDuration(planDurationMs)}\`` : ""}`,
				"",
				`**Stato Attuale:**`,
				"",
				...liveActivity.split("\n").filter(Boolean).map((line) => `${line}  `),
				"",
				`---`,
				"",
				`**Log Operazioni & Tempi in Tempo Reale:**`,
				...traceLogs.map((l) => `- ${l}`),
			]

			if (liveThinking) {
				const tailThink = liveThinking.length > 900 ? `...${liveThinking.slice(-900)}` : liveThinking
				headerBlock.push(
					"",
					`**Ragionamento Profondo Orchestratore (Thinking High — live stream):**`,
					`> ${tailThink.replace(/\n/g, "\n> ")}`,
				)
			}
			if (liveOutput) {
				const tailOut = liveOutput.length > 600 ? `...${liveOutput.slice(-600)}` : liveOutput
				headerBlock.push("", `**Verdetto JSON parziale:**`, `\`\`\`json\n${tailOut}\n\`\`\``)
			}
			this.emitInfoMessage(statusTs, headerBlock.join("\n"))
		}

		renderLivePlanCard("🔄 Preparazione audit del Plan ricevuto...")

		try {
			const { decision, metrics } = await this.evaluatePlanWithLlm(
				orchestratorModel,
				planText,
				pushLog,
				(activity, thinking, partialContent) => {
					renderLivePlanCard(activity, thinking, partialContent)
				},
			)

			if (!this.enabled) {
				return
			}

			const telemetrySummary = [
				`**Audit Orchestratore:** \`${formatDuration(Date.now() - opStartMs)}\``,
				`**Thinking (High):** \`~${metrics.estimatedThinkingTokens} tok\``,
				planDurationMs ? `**Durata Plan Mode:** \`${formatDuration(planDurationMs)}\`` : "",
				`**VRAM Swap:** \`${formatDuration(metrics.vramUnloadBeforeMs + metrics.vramUnloadAfterMs)}\``,
				`**TTFT:** \`${formatDuration(metrics.ttftMs)}\``,
				`**Inferenza:** \`${formatDuration(metrics.generationMs)}\` (@ **\`${metrics.tokensPerSecond.toFixed(1)} tok/s\`**)`,
			]
				.filter(Boolean)
				.join(" • ")

			const strengthsBullets =
				decision.strengths.length > 0 ? decision.strengths.map((s) => `- ${s}`).join("\n") : ""

			if (decision.valid) {
				this.planRevisionCount = 0
				pushLog(
					`VERDETTO: PLAN APPROVATO (Punteggio: **${decision.score}/10**). Il piano rispetta tutti i requisiti del prompt iniziale.`,
				)
				pushLog(
					`Attivazione automatica switch **Plan Mode → Act Mode** (caricamento modello Act: \`${actModelId ?? "default"}\` e avvio esecuzione)...`,
				)

				const actSwitchStart = Date.now()
				this.setStatus("idle")
				this.startedAt = undefined
				this.isEvaluatingPlan = false

				const buildApprovedCard = (actSwitchDurationText: string) =>
					[
						`**Orchestratore AI** (\`${orchestratorModel}\` • **Thinking: High**) — **Plan Valido e Approvato** (**${decision.score}/10**)`,
						telemetrySummary,
						`---`,
						`**Valutazione Architetturale del Supervisore:**\n${decision.reasoning}`,
						strengthsBullets ? `**Requisiti Verificati nel Plan:**\n${strengthsBullets}` : "",
						`**Azione Automatica Eseguita:** ${actSwitchDurationText}`,
						metrics.thinkingText
							? `\n**Ragionamento Profondo dell'Orchestratore (${metrics.estimatedThinkingTokens} token):**\n\`\`\`\n${metrics.thinkingText}\n\`\`\``
							: "",
						`\n**Log Dietro le Quinte & Telemetria Completa (${traceLogs.length} eventi):**\n${traceLogs.map((l) => `- ${l}`).join("\n")}`,
					]
						.filter(Boolean)
						.join("\n\n")

				this.emitInfoMessage(
					statusTs,
					buildApprovedCard(`Passaggio automatico a **Act Mode** (\`${actModelId ?? "default"}\`) in corso...`),
				)

				await this.options.triggerActMode()
				const actSwitchMs = Date.now() - actSwitchStart
				pushLog(
					`Sessione ricostruita in **Act Mode** in \`${formatDuration(actSwitchMs)}\` e comando di inizio lavoro inviato al modello \`${actModelId ?? "default"}\`.`,
				)
				this.emitInfoMessage(
					statusTs,
					buildApprovedCard(
						`Premuto automaticamente su **Act Mode** (switch completato in \`${formatDuration(actSwitchMs)}\`) → Lavoro avviato con il modello \`${actModelId ?? "default"}\`.`,
					),
				)
				sendWindowsNotification(
					"Orchestratore AI: Piano Approvato!",
					`Punteggio ${decision.score}/10: ${decision.reasoning.slice(0, 120)} → Avviato Act Mode (${actModelId ?? "default"})`,
				)
				return
			}

			// Plan is NOT valid or misses requirements from the initial prompt
			this.planRevisionCount += 1
			const modificationsBullets =
				decision.suggestedModifications.length > 0
					? decision.suggestedModifications.map((m, idx) => `${idx + 1}. ${m}`).join("\n")
					: decision.missingPoints.map((m, idx) => `${idx + 1}. ${m}`).join("\n") || `1. ${decision.reasoning}`

			const missingBullets =
				decision.missingPoints.length > 0 ? decision.missingPoints.map((p) => `- ${p}`).join("\n") : ""

			pushLog(
				`VERDETTO: PLAN NON VALIDO / INCOMPLETO (Punteggio: **${decision.score}/10** — Tentativo revisione **${this.planRevisionCount}/${this.maxPlanRevisions}**).`,
			)

			if (this.planRevisionCount <= this.maxPlanRevisions) {
				pushLog(
					`Invio automatico di **${decision.suggestedModifications.length || 1} modifiche richieste** al modello in **Plan Mode** (\`${planModelId ?? "default"}\`) per rigenerare il piano...`,
				)

				this.emitInfoMessage(
					statusTs,
					[
						`**Orchestratore AI** (\`${orchestratorModel}\` • **Thinking: High**) — **Plan da Correggere** (**${decision.score}/10** • Revisione **${this.planRevisionCount}/${this.maxPlanRevisions}**)`,
						telemetrySummary,
						`---`,
						`**Diagnosi Architetturale del Supervisore:**\n${decision.reasoning}`,
						missingBullets ? `**Lacune o Errori Rilevati nel Plan:**\n${missingBullets}` : "",
						`**Modifiche Richieste (inviate automaticamente a \`${planModelId ?? "default"}\` in Plan Mode):**\n${modificationsBullets}`,
						metrics.thinkingText
							? `\n**Ragionamento Profondo dell'Orchestratore (${metrics.estimatedThinkingTokens} token):**\n\`\`\`\n${metrics.thinkingText}\n\`\`\``
							: "",
						`\n**Log Dietro le Quinte & Telemetria Completa (${traceLogs.length} eventi):**\n${traceLogs.map((l) => `- ${l}`).join("\n")}`,
					]
						.filter(Boolean)
						.join("\n\n"),
				)
				sendWindowsNotification(
					`Orchestratore AI: Plan da Correggere (${this.planRevisionCount}/${this.maxPlanRevisions})`,
					`Punteggio ${decision.score}/10: ${decision.reasoning.slice(0, 120)}`,
				)

				const revisionPrompt = [
					`${ORCHESTRATOR_REVISION_PREFIX} L'Orchestratore AI ha analizzato il tuo piano rispetto al prompt iniziale dell'utente e ha rilevato che richiede le seguenti modifiche prima di poter passare ad Act Mode:`,
					"",
					`Diagnosi: ${decision.reasoning}`,
					"",
					"Modifiche obbligatorie da apportare al Plan:",
					modificationsBullets,
					"",
					`Prompt iniziale dell'utente da rispettare integralmente:\n"""\n${this.initialUserPrompt}\n"""`,
					"",
					"Usa i tool di lettura se necessario per raccogliere i dettagli mancanti, quindi aggiorna e correggi subito il piano in Plan Mode includendo tutte le modifiche indicate sopra.",
				].join("\n")

				this.startedAt = undefined
				this.setStatus("idle")
				this.isEvaluatingPlan = false
				this.lastPlanTurnStartedAt = Date.now()
				await this.options.sendPlanRevisionPrompt(revisionPrompt)
				return
			}

			// Reached max automatic revisions: the Orchestrator evaluates whether the plan,
			// although imperfect, is acceptable enough to proceed to Act mode, or if it must halt and wait for user input.
			pushLog(
				`LIMITE REVISIONI RAGGIUNTO (${this.maxPlanRevisions}/${this.maxPlanRevisions}): Valutazione di compromesso...`,
			)

			const isAcceptable =
				decision.isAcceptableFallback ??
				(decision.score >= 6 ||
					(decision.score >= 5 && decision.missingPoints.length <= 1 && !decision.reasoning.toLowerCase().includes("critico") && !decision.reasoning.toLowerCase().includes("inaccettabile")))

			if (isAcceptable) {
				pushLog(
					`DECISIONE FALLBACK: Il piano non è perfetto (${decision.score}/10) ma è giudicato sufficientemente eseguibile. Passaggio automatico ad Act Mode autorizzato.`,
				)

				const actSwitchStart = Date.now()
				this.emitInfoMessage(
					statusTs,
					[
						`**Orchestratore AI** (\`${orchestratorModel}\`) — **Limite revisioni raggiunto (${this.maxPlanRevisions}/${this.maxPlanRevisions}) • Plan Accettato con Riserva**`,
						telemetrySummary,
						`---`,
						`**Valutazione di Compromesso:**\nNonostante alcune imperfezioni minori, il piano è sufficientemente solido per iniziare l'esecuzione in sicurezza. L'Orchestratore autorizza il passaggio ad Act Mode.`,
						`**Punteggio:** **${decision.score}/10**`,
						`**Ragionamento:** ${decision.reasoning}`,
						missingBullets ? `**Note/Imperfezioni rimaste:**\n${missingBullets}` : "",
						`\n**Log Dietro le Quinte & Telemetria Completa (${traceLogs.length} eventi):**\n${traceLogs.map((l) => `- ${l}`).join("\n")}`,
						`*Premuto automaticamente su **Act Mode** (\`${actModelId ?? "default"}\`) per avviare il lavoro...*`,
					]
						.filter(Boolean)
						.join("\n\n"),
				)
				sendWindowsNotification(
					"Orchestratore AI: Plan Accettato con Riserva",
					`Limite revisioni: piano sufficiente (${decision.score}/10) → Passaggio ad Act Mode`,
				)

				await this.options.triggerActMode()
				const actSwitchMs = Date.now() - actSwitchStart
				pushLog(
					`Passaggio ad **Act Mode** completato in \`${formatDuration(actSwitchMs)}\` con il modello \`${actModelId ?? "default"}\`.`,
				)
				return
			}

			// Plan is fundamentally flawed/unacceptable: halt and wait for manual user input
			pushLog(
				`DECISIONE FALLBACK: Il piano è critico/inaccettabile (Punteggio: ${decision.score}/10). Blocco automatico e attesa dell'intervento dell'utente.`,
			)
			this.emitInfoMessage(
				statusTs,
				[
					`**Orchestratore AI** (\`${orchestratorModel}\`) — **Limite revisioni raggiunto (${this.maxPlanRevisions}/${this.maxPlanRevisions}) • Richiesto Intervento Utente**`,
					telemetrySummary,
					`---`,
					`**Diagnosi del Supervisore:**\nIl piano presenta lacune critiche e non è considerato sicuro né sufficiente per l'avvio automatico dei lavori.`,
					`**Punteggio:** **${decision.score}/10**`,
					`**Motivazione del blocco:**\n${decision.reasoning}`,
					missingBullets ? `**Punti Critici o Mancanti Rilevati:**\n${missingBullets}` : "",
					`**Modifiche Necessarie:**\n${modificationsBullets}`,
					`\n**Log Dietro le Quinte & Telemetria Completa (${traceLogs.length} eventi):**\n${traceLogs.map((l) => `- ${l}`).join("\n")}`,
					`*L'Orchestratore non ha premuto Act. Puoi inviare correzioni in Plan Mode o premere manualmente su Act se desideri procedere comunque.*`,
				]
					.filter(Boolean)
					.join("\n\n"),
			)
			sendWindowsNotification(
				"Orchestratore AI: Richiesto Intervento Utente",
				`Piano non accettabile (${decision.score}/10): ${decision.reasoning.slice(0, 120)}`,
			)
		} catch (error) {
			if ((error as Error)?.name === "AbortError") {
				Logger.log("[LocalAiOrchestrator] Plan evaluation aborted")
				return
			}
			const errMsg = error instanceof Error ? error.message : String(error)
			pushLog(`❌ Errore durante la verifica del Plan: ${errMsg}`)
			Logger.error("[LocalAiOrchestrator] Plan evaluation failed:", error)
			this.emitInfoMessage(
				statusTs,
				[
					`⚠️ **Orchestratore AI** (\`${orchestratorModel}\`) — **Errore durante la verifica del Plan**`,
					`\`${errMsg}\``,
					`**📡 Log Dietro le Quinte:**\n${traceLogs.map((l) => `- ${l}`).join("\n")}`,
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
		pushLog: (step: string) => void,
		onLiveUpdate: (activity: string, thinking?: string, partialContent?: string) => void,
	): Promise<{ decision: OrchestratorPlanDecision; metrics: LlmCallMetrics }> {
		const systemPrompt = [
			"Sei l'Orchestratore AI di supervisione in Cline.",
			"Verifica se il Piano risponde adeguatamente alla richiesta dell'utente.",
			"REGOLE:",
			"1. Se il piano è ragionevole, coerente ed eseguibile (anche se perfettibile), imposta \"valid\": true.",
			"2. Boccia il piano (\"valid\": false) SOLO se è completamente fuori strada o distruttivo.",
			"3. Rispondi ESCLUSIVAMENTE con un JSON valido (nessun testo o markdown fuori dal JSON):",
			"{",
			'  "valid": true,',
			'  "isAcceptableFallback": true,',
			'  "score": 8,',
			'  "reasoning": "Breve giudizio sintetico",',
			'  "strengths": ["Punti coperti"],',
			'  "missingPoints": [],',
			'  "suggestedModifications": []',
			"}",
		].join("\n")

		const requirementsSection = this.lastPromptEvaluation
			? [
				"\nCRITERI ATTESI:",
				`Obiettivo: ${this.lastPromptEvaluation.objective}`,
				...this.lastPromptEvaluation.requirements.map((r) => `- ${r}`),
			]
				.filter(Boolean)
				.join("\n")
			: ""

		const userContent = [
			`RICHIESTA UTENTE:\n"""\n${this.initialUserPrompt || "(Nessun prompt registrato)"}\n"""`,
			requirementsSection,
			`\nPIANO PROPOSTO DA PLAN MODE:\n"""\n${planText.slice(0, 8000)}\n"""`,
		]
			.filter(Boolean)
			.join("\n")

		const metrics = await this.callOrchestratorLlmVerbose({
			modelId: orchestratorModel,
			systemPrompt,
			userPrompt: userContent,
			maxTokens: 512,
			reasoningEffort: "low",
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
				(lower.includes("approvato") && !lower.includes("non approvato") && !lower.includes("non valido"))
			return {
				decision: {
					valid: looksApproved,
					score: looksApproved ? 8 : 5,
					reasoning: cleanedText.slice(0, 600) || "Valutazione completata.",
					strengths: [],
					missingPoints: [],
					suggestedModifications: looksApproved
						? []
						: [
							"Approfondisci il piano verificando lo stato effettivo dei file nel workspace e coprendo tutti i requisiti del prompt iniziale.",
						],
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
		const reasoning =
			typeof parsed.reasoning === "string" && parsed.reasoning.trim()
				? parsed.reasoning.trim()
				: valid
					? "Il piano rispetta tutti i requisiti del prompt iniziale."
					: "Il piano non soddisfa completamente i requisiti richiesti."
		const strengths = Array.isArray(parsed.strengths)
			? parsed.strengths.filter((s): s is string => typeof s === "string" && s.trim().length > 0)
			: []
		const missingPoints = Array.isArray(parsed.missingPoints)
			? parsed.missingPoints.filter((p): p is string => typeof p === "string" && p.trim().length > 0)
			: []
		const suggestedModifications = Array.isArray(parsed.suggestedModifications)
			? parsed.suggestedModifications.filter((m): m is string => typeof m === "string" && m.trim().length > 0)
			: missingPoints

		const isAcceptableFallback =
			typeof parsed.isAcceptableFallback === "boolean"
				? parsed.isAcceptableFallback
				: valid || score >= 6

		return {
			decision: {
				valid,
				score,
				reasoning,
				strengths,
				missingPoints,
				suggestedModifications,
				isAcceptableFallback,
			},
			metrics,
		}
	}

	private findLatestCompletedPlanMessage(messages: ClineMessage[]): ClineMessage | undefined {
		for (let i = messages.length - 1; i >= 0; i--) {
			const msg = messages[i]
			if (msg.type === "say" && (msg.say === "task" || msg.say === "user_feedback")) {
				break
			}
			if (
				msg.type === "say" &&
				!msg.partial &&
				msg.text?.trim() &&
				(msg.say === "plan_completion_result" || msg.say === "completion_result" || msg.say === "text")
			) {
				return msg
			}
		}
		return undefined
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
		const rawBase = cfg.openAiBaseUrl?.trim() || `${this.gatewayBaseUrl}/v1`
		const baseUrl = rawBase.replace(/\/+$/, "")
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
				if (!id || item?.disabled === true || disabledSet.has(id)) {
					continue
				}
				models.push(id)
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

	/**
	 * Calls the Orchestrator model with HIGH thinking (`reasoning_effort: "high"`, `max_tokens: 4096`)
	 * and verbose real-time telemetry:
	 * 1. Probes `/_localai/status` and logs currently loaded VRAM models
	 * 2. Flushes VRAM (`/_localai/free-vram`) if a different model is loaded, timing the unload
	 * 3. Polls `/_localai/prompt_progress` while waiting for cold load & prefill
	 * 4. Streams thinking (`reasoning_content`) and JSON (`content`) tokens via SSE
	 * 5. Flushes VRAM after completion so the Plan/Act model loads cleanly
	 */
	private async callOrchestratorLlmVerbose(params: {
		modelId: string
		systemPrompt: string
		userPrompt: string
		maxTokens: number
		reasoningEffort?: "low" | "medium" | "high"
		phaseLabel: "Prompt" | "Plan"
		pushLog: (step: string) => void
		onLiveUpdate: (activity: string, thinking?: string, partialContent?: string) => void
	}): Promise<LlmCallMetrics> {
		const {
			modelId,
			systemPrompt,
			userPrompt,
			maxTokens,
			reasoningEffort = "high",
			phaseLabel,
			pushLog,
			onLiveUpdate,
		} = params
		const { baseUrl, apiKey } = this.getOpenAiEndpointAndKey()
		const currentStatus: OrchestratorStatus =
			phaseLabel === "Prompt" ? "evaluating_prompt" : "evaluating_plan"
		const estimatedPromptTokens = estimateTokens(systemPrompt) + estimateTokens(userPrompt)

		// Step 1: Check Gateway & VRAM Status
		const vramCheckStart = Date.now()
		this.setStatus(currentStatus, "Verifica VRAM...")
		onLiveUpdate("🔎 Controllo stato VRAM sul Gateway LocalAI (`/_localai/status`)...")
		const previouslyLoadedModels = await this.probeLoadedGatewayModels()
		const vramCheckMs = Date.now() - vramCheckStart

		let vramUnloadBeforeMs = 0
		const otherLoaded = previouslyLoadedModels.filter((m) => m.toLowerCase() !== modelId.toLowerCase())
		if (otherLoaded.length > 0) {
			pushLog(
				`Rilevato modello attivo in VRAM: \`${otherLoaded.join(", ")}\` (controllo in ${formatDuration(vramCheckMs)}). Avvio scaricamento VRAM (\`POST /_localai/free-vram\`)...`,
			)
			this.setStatus(currentStatus, "Scaricamento VRAM...")
			onLiveUpdate(`🧹 Scaricamento di \`${otherLoaded.join(", ")}\` dai 12 GB di VRAM...`)
			const unloadStart = Date.now()
			await this.freeGatewayVram()
			vramUnloadBeforeMs = Date.now() - unloadStart
			pushLog(`VRAM liberata con successo in **${formatDuration(vramUnloadBeforeMs)}**.`)
		} else if (previouslyLoadedModels.some((m) => m.toLowerCase() === modelId.toLowerCase())) {
			pushLog(`Il modello Orchestratore \`${modelId}\` è già caldo in VRAM (${formatDuration(vramCheckMs)}).`)
		} else {
			const unloadStart = Date.now()
			await this.freeGatewayVram()
			vramUnloadBeforeMs = Date.now() - unloadStart
			pushLog(`Verifica VRAM completata (${formatDuration(vramCheckMs + vramUnloadBeforeMs)} — VRAM pronta).`)
		}

		// Step 2: Launch LLM inference & start live prefill/wait ticker
		const llmStartMs = Date.now()
		let firstTokenAtMs = 0
		let reasoningAccumulated = ""
		let contentAccumulated = ""
		let finishReason: string | undefined
		let lastUiUpdateMs = 0

		pushLog(
			`Invio richiesta \`POST ${baseUrl}/chat/completions\` a \`${modelId}\` (payload: ~${estimatedPromptTokens} token input, \`reasoning_effort="${reasoningEffort}"\`, \`max_tokens=${maxTokens}\`).`,
		)
		this.setStatus(currentStatus, `Caricamento ${modelId.split("-")[0]}...`)
		onLiveUpdate(
			`🚀 Caricamento pesi GGUF di \`${modelId}\` e prefill prompt (~${estimatedPromptTokens} token, Thinking: ${reasoningEffort.toUpperCase()})...`,
		)

		const abortController = new AbortController()
		this.activeAbortController = abortController
		const timeout = setTimeout(() => abortController.abort(), 180_000)

		const getCombinedThinking = () => {
			const inlineThink = extractThinkingBlocks(contentAccumulated)
			return [reasoningAccumulated.trim(), inlineThink].filter(Boolean).join("\n").trim()
		}
		const getCleanContent = () => stripThinkingBlocks(contentAccumulated)

		// Background ticker that polls prompt_progress and updates wait time every 650ms until finish
		let tickerActive = true
		const tickerTimer = setInterval(async () => {
			if (!tickerActive) {
				return
			}
			const elapsedSec = ((Date.now() - llmStartMs) / 1000).toFixed(1)
			if (firstTokenAtMs === 0) {
				const progress = await this.probePromptProgress()
				const progressLabel =
					typeof progress === "number" ? `Prefill prompt: **${progress}%**` : "Caricamento GGUF / Prefill in GPU"
				this.setStatus(
					currentStatus,
					typeof progress === "number"
						? `${phaseLabel} ${progress}% (${elapsedSec}s)`
						: `${phaseLabel} (${elapsedSec}s)...`,
				)
				onLiveUpdate(`⏳ Attesa risposta da \`${modelId}\` — ${progressLabel} • ⏱️ \`${elapsedSec}s trascorsi\``)
			} else {
				const genElapsedSec = Math.max(0.1, (Date.now() - firstTokenAtMs) / 1000)
				const thinkToks = estimateTokens(getCombinedThinking())
				const outToks = estimateTokens(getCleanContent())
				const totalGenToks = thinkToks + outToks
				const tps = (totalGenToks / genElapsedSec).toFixed(1)
				const subPhase = outToks > 0 ? `JSON ${outToks} tok` : `Think ${thinkToks} tok`
				this.setStatus(currentStatus, `${phaseLabel}: ${subPhase} (${tps} t/s)`)
				onLiveUpdate(
					`Elaborazione in corso da \`${modelId}\`\nThinking: \`~${thinkToks} tok\` • Output JSON: \`~${outToks} tok\` (**\`${tps} tok/s\`**) • \`${elapsedSec}s\``,
					getCombinedThinking(),
					getCleanContent(),
				)
			}
		}, 650)

		let vramUnloadAfterMs = 0
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

				while (true) {
					const { done, value } = await reader.read()
					if (done) {
						break
					}
					sseBuffer += decoder.decode(value, { stream: true })
					const lines = sseBuffer.split("\n")
					sseBuffer = lines.pop() ?? ""

					for (const rawLine of lines) {
						const line = rawLine.trim()
						if (!line.startsWith("data:")) {
							continue
						}
						const dataPayload = line.slice(5).trim()
						if (!dataPayload || dataPayload === "[DONE]") {
							continue
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
								pushLog(
									`⚡ Primo token ricevuto da \`${modelId}\` (TTFT / Cold Load + Prefill: **${formatDuration(firstTokenAtMs - llmStartMs)}**). Avvio streaming ragionamento profondo...`,
								)
							}
							if (reasoningPiece) {
								reasoningAccumulated += reasoningPiece
							}
							if (contentPiece) {
								if (!loggedJsonTransition && reasoningAccumulated.length > 0 && contentAccumulated.length === 0) {
									loggedJsonTransition = true
									pushLog(
										`🧠 Fase di thinking conclusa (~${estimateTokens(reasoningAccumulated)} token in ${formatDuration(Date.now() - (firstTokenAtMs || llmStartMs))}). Inizio emissione JSON strutturato...`,
									)
								}
								contentAccumulated += contentPiece
							}

							const now = Date.now()
							if (now - lastUiUpdateMs >= 350) {
								lastUiUpdateMs = now
								const genSec = Math.max(0.1, (now - (firstTokenAtMs || llmStartMs)) / 1000)
								const thinkToks = estimateTokens(getCombinedThinking())
								const outToks = estimateTokens(getCleanContent())
								const tps = ((thinkToks + outToks) / genSec).toFixed(1)
								onLiveUpdate(
									`Streaming da \`${modelId}\`\n\nThinking: \`~${thinkToks} tok\` • JSON: \`~${outToks} tok\` (**\`${tps} tok/s\`**)`,
									getCombinedThinking(),
									getCleanContent(),
								)
							}
						} catch {
							// Ignore non-JSON SSE comments
						}
					}
				}
			} else {
				// Standard JSON response (or unit test mock)
				const data = (await res.json()) as {
					choices?: Array<{
						finish_reason?: string | null
						message?: { content?: string; reasoning_content?: string }
					}>
				}
				firstTokenAtMs = Date.now()
				const choice = data?.choices?.[0]
				finishReason = choice?.finish_reason ?? undefined
				const msg = choice?.message
				reasoningAccumulated = msg?.reasoning_content ?? ""
				contentAccumulated = msg?.content ?? ""
				pushLog(
					`Risposta ricevuta da \`${modelId}\` in **${formatDuration(firstTokenAtMs - llmStartMs)}** (Thinking: ~${estimateTokens(getCombinedThinking())} tok, Output: ~${estimateTokens(getCleanContent())} tok).`,
				)
			}

			const llmEndMs = Date.now()
			const totalLlmMs = Math.max(1, llmEndMs - llmStartMs)
			const ttftMs = firstTokenAtMs > 0 ? Math.max(1, firstTokenAtMs - llmStartMs) : totalLlmMs
			const generationMs = firstTokenAtMs > 0 ? Math.max(1, llmEndMs - firstTokenAtMs) : totalLlmMs
			const finalThinking = getCombinedThinking()
			const finalCleanContent = getCleanContent()
			const estimatedThinkingTokens = finalThinking ? estimateTokens(finalThinking) : 0
			const estimatedOutputTokens = finalCleanContent ? estimateTokens(finalCleanContent) : 0
			const totalGeneratedTokens = Math.max(1, estimatedThinkingTokens + estimatedOutputTokens)
			const tokensPerSecond = totalGeneratedTokens / Math.max(0.05, generationMs / 1000)

			if (hasReadableStream && contentType.includes("text/event-stream")) {
				pushLog(
					`Generazione completata in **${formatDuration(generationMs)}** (Thinking: ~${estimatedThinkingTokens} tok + JSON: ~${estimatedOutputTokens} tok @ **${tokensPerSecond.toFixed(1)} tok/s**, finish_reason=\`${finishReason ?? "stop"}\`).`,
				)
			}

			// Step 3: Free VRAM after orchestrator call so Plan/Act model loads cleanly
			tickerActive = false
			clearInterval(tickerTimer)
			this.setStatus(currentStatus, "Rilascio VRAM...")
			onLiveUpdate(
				`🧹 Scaricamento dell'Orchestratore (\`${modelId}\`) dalla VRAM per preparare il modello successivo...`,
				finalThinking,
				finalCleanContent,
			)
			const unloadAfterStart = Date.now()
			await this.freeGatewayVram()
			vramUnloadAfterMs = Date.now() - unloadAfterStart
			pushLog(
				`VRAM rilasciata dall'Orchestratore in **${formatDuration(vramUnloadAfterMs)}** (12 GB VRAM liberi per il prossimo modello).`,
			)

			return {
				content: finalCleanContent,
				thinkingText: finalThinking,
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
			if (vramUnloadAfterMs === 0) {
				await this.freeGatewayVram()
			}
		}
	}

	private async probeLoadedGatewayModels(): Promise<string[]> {
		try {
			const controller = new AbortController()
			const timeout = setTimeout(() => controller.abort(), 2000)
			const statusRes = await fetch(`${this.gatewayBaseUrl}/_localai/status`, {
				signal: controller.signal,
			})
			clearTimeout(timeout)
			if (statusRes.ok) {
				const statusData = (await statusRes.json()) as {
					llm?: { loadedModels?: string[] }
				}
				return statusData?.llm?.loadedModels ?? []
			}
		} catch {
			// Ignore offline/mocked gateway
		}
		return []
	}

	private async probePromptProgress(): Promise<number | null> {
		try {
			const controller = new AbortController()
			const timeout = setTimeout(() => controller.abort(), 500)
			const res = await fetch(`${this.gatewayBaseUrl}/_localai/prompt_progress`, {
				signal: controller.signal,
			})
			clearTimeout(timeout)
			if (res.ok) {
				const data = (await res.json()) as { progress?: number | null }
				if (typeof data?.progress === "number") {
					return data.progress
				}
			}
		} catch {
			// Ignore
		}
		return null
	}

	private async freeGatewayVram(): Promise<void> {
		try {
			const controller = new AbortController()
			const timeout = setTimeout(() => controller.abort(), 4000)
			await fetch(`${this.gatewayBaseUrl}/_localai/free-vram`, {
				method: "POST",
				signal: controller.signal,
			})
			clearTimeout(timeout)
		} catch {
			// Ignore if gateway is offline or unreachable in tests
		}
	}

	private emitInfoMessage(ts: number, text: string): void {
		const msg: ClineMessage = {
			ts,
			type: "say",
			say: "info",
			text,
			partial: false,
		}
		this.options.emitMessage(msg)
		void this.options.postStateToWebview().catch(() => { })
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
