import { beforeEach, describe, expect, it, vi } from "vitest"
import type { ClineMessage } from "@shared/ExtensionMessage"
import {
	LocalAiOrchestratorCoordinator,
	ORCHESTRATOR_REVISION_PREFIX,
} from "./local-ai-orchestrator-coordinator"

describe("LocalAiOrchestratorCoordinator", () => {
	let emittedMessages: ClineMessage[]
	let triggerActMode: () => Promise<void>
	let sendPlanRevisionPrompt: (prompt: string) => Promise<void>
	let postStateToWebview: () => Promise<void>
	let triggerActModeMock: ReturnType<typeof vi.fn>
	let sendPlanRevisionPromptMock: ReturnType<typeof vi.fn>
	let postStateToWebviewMock: ReturnType<typeof vi.fn>
	let currentMode: "plan" | "act"
	let clineMessages: ClineMessage[]
	let planModelId: string
	let actModelId: string

	beforeEach(() => {
		emittedMessages = []
		triggerActModeMock = vi.fn().mockResolvedValue(undefined)
		sendPlanRevisionPromptMock = vi.fn().mockResolvedValue(undefined)
		postStateToWebviewMock = vi.fn().mockResolvedValue(undefined)
		triggerActMode = () => triggerActModeMock()
		sendPlanRevisionPrompt = (prompt: string) => sendPlanRevisionPromptMock(prompt)
		postStateToWebview = () => postStateToWebviewMock()
		currentMode = "plan"
		clineMessages = []
		planModelId = "Qwen3.6-27B-CoderX-48K-115toks"
		actModelId = "Qwen3.6-27B-CoderX-32K-125toks"
	})

	function createCoordinator(initialEnabled = true) {
		return new LocalAiOrchestratorCoordinator({
			initialEnabled,
			getApiConfiguration: () => ({
				openAiBaseUrl: "http://127.0.0.1:8080/v1",
				openAiApiKey: "local",
				planModeOpenAiModelId: planModelId,
				actModeOpenAiModelId: actModelId,
			}),
			getCurrentMode: () => currentMode,
			setModeToPlan: async () => {
				currentMode = "plan"
			},
			getClineMessages: () => clineMessages,
			getSessionId: () => "test-session",
			isSessionRunning: () => false,
			emitMessage: (msg) => {
				emittedMessages.push(msg)
			},
			postStateToWebview,
			triggerActMode,
			sendPlanRevisionPrompt,
		})
	}

	it("selects the best orchestrator model that is neither the Plan Mode nor the Act Mode default model", () => {
		const coordinator = createCoordinator()
		// Plan is Qwen3.6-27B-CoderX-48K-115toks, Act is Qwen3.6-27B-CoderX-32K-125toks
		const chosen = coordinator.resolveOrchestratorModelIdSync([
			"Gemma4-12B-Agentic-47toks",
			"Qwen3.6-35B-Coding-Avanzato-60toks",
			"Inkling-2-Mini-26B-32K-50toks",
			"Qwen3.6-27B-CoderX-48K-115toks",
			"Qwen3.6-27B-CoderX-32K-125toks",
		])
		expect(chosen).toBe("Qwen3.6-35B-Coding-Avanzato-60toks")
		expect(chosen).not.toBe(planModelId)
		expect(chosen).not.toBe(actModelId)

		// If the user switches Plan Mode to Qwen3.6-35B-Coding-Avanzato-60toks,
		// the orchestrator automatically shifts to Inkling-2-Mini-26B-32K-50toks!
		planModelId = "Qwen3.6-35B-Coding-Avanzato-60toks"
		actModelId = "Qwen3.6-27B-CoderX-48K-115toks"
		const chosen2 = coordinator.resolveOrchestratorModelIdSync([
			"Gemma4-12B-Agentic-47toks",
			"Qwen3.6-35B-Coding-Avanzato-60toks",
			"Inkling-2-Mini-26B-32K-50toks",
			"Qwen3.6-27B-CoderX-48K-115toks",
			"TwIL-8B-Reasoning-96K-80toks",
		])
		expect(chosen2).toBe("TwIL-8B-Reasoning-96K-80toks")
		expect(chosen2).not.toBe(planModelId)
		expect(chosen2).not.toBe(actModelId)
	})

	it("evaluates user prompts and appends structured acceptance criteria when enabled", async () => {
		global.fetch = vi.fn().mockImplementation(async (url: string) => {
			if (String(url).endsWith("/models")) {
				return {
					ok: true,
					json: async () => ({
						data: [
							{ id: "Qwen3.6-35B-Coding-Avanzato-60toks" },
							{ id: "Qwen3.6-27B-CoderX-48K-115toks" },
						],
					}),
				}
			}
			if (String(url).endsWith("/chat/completions")) {
				return {
					ok: true,
					json: async () => ({
						choices: [
							{
								message: {
									content: JSON.stringify({
										isClear: true,
										qualityScore: 9,
										objective: "Creare un endpoint REST di autenticazione JWT",
										requirements: ["Endpoint POST /api/login", "Validazione token JWT"],
										acceptanceCriteria: ["Il piano deve includere middleware e gestione errori 401"],
										planningGuidance: "Definisci rotte, controller e test unitari.",
									}),
								},
							},
						],
					}),
				}
			}
			return { ok: true, json: async () => ({ ok: true }) }
		}) as any

		const coordinator = createCoordinator(true)
		const result = await coordinator.evaluateUserPrompt(
			"Crea un endpoint di login con JWT",
			"Crea un endpoint di login con JWT",
		)

		expect(result).toContain("<orchestrator_evaluation>")
		expect(result).toContain("Creare un endpoint REST di autenticazione JWT")
		expect(result).toContain("Endpoint POST /api/login")
		expect(emittedMessages.length).toBeGreaterThanOrEqual(2)
		expect(emittedMessages[emittedMessages.length - 1].text).toContain("TECHNICAL SPECIFICATION READY FOR PLAN MODE")
	})

	it("reads Plan Mode response, validates it against initial prompt, and automatically triggers Act Mode when valid", async () => {
		global.fetch = vi.fn().mockImplementation(async (url: string) => {
			if (String(url).endsWith("/chat/completions")) {
				return {
					ok: true,
					json: async () => ({
						choices: [
							{
								message: {
									content: JSON.stringify({
										valid: true,
										score: 9,
										reasoning: "Il piano copre integralmente l'endpoint JWT, il middleware e i test richiesti.",
										missingPoints: [],
										suggestedModifications: [],
									}),
								},
							},
						],
					}),
				}
			}
			return { ok: true, json: async () => ({ data: [{ id: "Qwen3.6-35B-Coding-Avanzato-60toks" }] }) }
		}) as any

		const coordinator = createCoordinator(true)
		clineMessages = [
			{ ts: 100, type: "say", say: "task", text: "Crea un endpoint di login con JWT", partial: false },
			{
				ts: 200,
				type: "say",
				say: "plan_completion_result",
				text: "1. Creare POST /api/login\n2. Aggiungere middleware JWT\n3. Aggiungere test",
				partial: false,
			},
		]

		await coordinator.handlePlanTurnCompleted()

		expect(triggerActModeMock).toHaveBeenCalledTimes(1)
		expect(sendPlanRevisionPromptMock).not.toHaveBeenCalled()
		expect(emittedMessages[emittedMessages.length - 1].text).toContain("PLAN APPROVED BY SUPERVISOR")
	})

	it("suggests appropriate modifications and stays in Plan Mode when the received Plan is incomplete or invalid", async () => {
		global.fetch = vi.fn().mockImplementation(async (url: string) => {
			if (String(url).endsWith("/chat/completions")) {
				return {
					ok: true,
					json: async () => ({
						choices: [
							{
								message: {
									content: JSON.stringify({
										valid: false,
										score: 4,
										reasoning: "Il piano non include la validazione del token JWT e la gestione degli errori 401.",
										missingPoints: ["Manca middleware di verifica JWT"],
										suggestedModifications: [
											"Aggiungere il middleware di verifica del token JWT",
											"Specificare la gestione delle risposte HTTP 401 Unauthorized",
										],
									}),
								},
							},
						],
					}),
				}
			}
			return { ok: true, json: async () => ({ data: [{ id: "Qwen3.6-35B-Coding-Avanzato-60toks" }] }) }
		}) as any

		const coordinator = createCoordinator(true)
		clineMessages = [
			{ ts: 100, type: "say", say: "task", text: "Crea un endpoint di login con JWT e middleware 401", partial: false },
			{
				ts: 201,
				type: "say",
				say: "plan_completion_result",
				text: "1. Creare solo un file vuoto login.ts",
				partial: false,
			},
		]

		await coordinator.handlePlanTurnCompleted()

		expect(triggerActModeMock).not.toHaveBeenCalled()
		expect(sendPlanRevisionPromptMock).toHaveBeenCalledTimes(1)
		const sentRevision = sendPlanRevisionPromptMock.mock.calls[0][0] as string
		expect(sentRevision).toContain(ORCHESTRATOR_REVISION_PREFIX)
		expect(sentRevision).toContain("Aggiungere il middleware di verifica del token JWT")
		expect(emittedMessages[emittedMessages.length - 1].text).toContain("PLAN NEEDS REVISION")
	})

	it("when max revisions limit is reached, triggers Act mode if the imperfect plan is acceptable as a fallback", async () => {
		global.fetch = vi.fn().mockImplementation(async (url: string) => {
			if (String(url).endsWith("/chat/completions")) {
				return {
					ok: true,
					json: async () => ({
						choices: [
							{
								message: {
									content: JSON.stringify({
										valid: false,
										score: 6,
										isAcceptableFallback: true,
										reasoning: "Il piano ha qualche imprecisione di naming, ma i passaggi essenziali ci sono.",
										missingPoints: ["Dettaglio naming endpoint"],
										suggestedModifications: ["Aggiusta il nome della route"],
									}),
								},
							},
						],
					}),
				}
			}
			return { ok: true, json: async () => ({ data: [{ id: "Qwen3.6-35B-Coding-Avanzato-60toks" }] }) }
		}) as any

		const coordinator = createCoordinator(true)
		// maxPlanRevisions is 3 by default; simulate having already performed 3 revisions
		;(coordinator as any).planRevisionCount = 3
		;(coordinator as any).initialUserPrompt = "Crea un endpoint di login"

		clineMessages = [
			{ ts: 100, type: "say", say: "task", text: "Crea un endpoint di login", partial: false },
			{
				ts: 202,
				type: "say",
				say: "plan_completion_result",
				text: "1. Creare login router con auth handler",
				partial: false,
			},
		]

		await coordinator.handlePlanTurnCompleted()

		// Since planRevisionCount became 3 (> maxPlanRevisions 2) and isAcceptableFallback is true,
		// it should trigger Act mode!
		expect(triggerActModeMock).toHaveBeenCalledTimes(1)
		expect(sendPlanRevisionPromptMock).not.toHaveBeenCalled()
		expect(emittedMessages[emittedMessages.length - 1].text).toContain("PLAN ACCEPTED WITH RESERVATIONS")
	})

	it("when max revisions limit is reached, halts and waits for user if the plan is critically flawed", async () => {
		global.fetch = vi.fn().mockImplementation(async (url: string) => {
			if (String(url).endsWith("/chat/completions")) {
				return {
					ok: true,
					json: async () => ({
						choices: [
							{
								message: {
									content: JSON.stringify({
										valid: false,
										score: 2,
										isAcceptableFallback: false,
										reasoning: "Il piano è completamente errato ed esegue un comando distruttivo rm -rf.",
										missingPoints: ["Piano inaccettabile"],
										suggestedModifications: ["Riscrivi da zero"],
									}),
								},
							},
						],
					}),
				}
			}
			return { ok: true, json: async () => ({ data: [{ id: "Qwen3.6-35B-Coding-Avanzato-60toks" }] }) }
		}) as any

		const coordinator = createCoordinator(true)
		;(coordinator as any).planRevisionCount = 3
		;(coordinator as any).initialUserPrompt = "Crea un endpoint di login"

		clineMessages = [
			{ ts: 100, type: "say", say: "task", text: "Crea un endpoint di login", partial: false },
			{
				ts: 203,
				type: "say",
				say: "plan_completion_result",
				text: "1. Cancellare tutto con rm -rf",
				partial: false,
			},
		]

		await coordinator.handlePlanTurnCompleted()

		// Must NOT trigger Act mode and must halt for user
		expect(triggerActModeMock).not.toHaveBeenCalled()
		expect(sendPlanRevisionPromptMock).not.toHaveBeenCalled()
		expect(emittedMessages[emittedMessages.length - 1].text).toContain("PLAN BLOCKED — USER INTERVENTION REQUIRED")
	})

	it("checks completed Act work and proposes next modification for Plan mode", async () => {
		global.fetch = vi.fn().mockImplementation(async (url: string) => {
			if (String(url).endsWith("/chat/completions")) {
				return {
					ok: true,
					json: async () => ({
						choices: [
							{
								message: {
									content: JSON.stringify({
										completed: true,
										score: 9,
										summary: "Endpoint di login creato e test di autenticazione completati con successo.",
										verifiedPoints: ["Creato auth router", "Aggiunto JWT middleware"],
										issuesOrPending: [],
										proposedNextAction: "Aggiungere endpoint di refresh token e logout.",
									}),
								},
							},
						],
					}),
				}
			}
			return { ok: true, json: async () => ({ data: [{ id: "Qwen3.6-35B-Coding-Avanzato-60toks" }] }) }
		}) as any

		const coordinator = createCoordinator(true)
		currentMode = "act"
		clineMessages = [
			{ ts: 100, type: "say", say: "task", text: "Crea un endpoint di login", partial: false },
			{
				ts: 300,
				type: "say",
				say: "completion_result",
				text: "Ho implementato l'endpoint di login con successo.",
				partial: false,
			},
		]

		await coordinator.handleTurnCompleted()

		expect(emittedMessages[emittedMessages.length - 1].text).toContain("ACT WORK VERIFIED AND COMPLETED")
		expect(emittedMessages[emittedMessages.length - 1].text).toContain("Aggiungere endpoint di refresh token e logout")
	})
})
