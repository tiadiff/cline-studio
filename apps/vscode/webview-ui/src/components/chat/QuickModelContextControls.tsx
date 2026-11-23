import { StringRequest } from "@shared/proto/cline/common"
import { OpenAiModelsRequest } from "@shared/proto/cline/models"
import {
	isOpenaiReasoningEffort,
	type Mode,
	OPENAI_REASONING_EFFORT_OPTIONS,
	type OpenaiReasoningEffort,
} from "@shared/storage/types"
import { BotIcon, RefreshCwIcon, SlidersHorizontalIcon } from "lucide-react"
import type React from "react"
import { useCallback, useEffect, useMemo, useState } from "react"
import { getModeSpecificFields } from "@/components/settings/utils/providerUtils"
import {
	DEFAULT_REASONING_EFFORT_OPTION,
	type ReasoningEffortSelection,
	reasoningEffortPatch,
} from "@/components/settings/utils/reasoningEffort"
import { useApiConfigurationHandlers } from "@/components/settings/utils/useApiConfigurationHandlers"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { useExtensionState } from "@/context/ExtensionStateContext"
import { useNormalizedApiConfiguration } from "@/hooks/useNormalizedApiConfiguration"
import { useOrchestratorState } from "@/hooks/useOrchestratorState"
import { fromProtobufProviderModelOverrides, useProviderConfig } from "@/hooks/useProviderConfig"
import { useProviderModels } from "@/hooks/useProviderModels"
import { cn } from "@/lib/utils"
import { ModelsServiceClient } from "@/services/grpc-client"

const CONTEXT_WINDOW_PRESETS = [
	{ value: 4096, label: "4k" },
	{ value: 8192, label: "8k" },
	{ value: 16384, label: "16k" },
	{ value: 32768, label: "32k" },
	{ value: 43008, label: "42k" },
	{ value: 49152, label: "48k" },
	{ value: 65536, label: "64k" },
	{ value: 131072, label: "128k" },
	{ value: 200000, label: "200k" },
	{ value: 262144, label: "256k" },
	{ value: 524288, label: "512k" },
	{ value: 1000000, label: "1M" },
]

const REASONING_EFFORT_LABELS: Record<OpenaiReasoningEffort, string> = {
	none: "None",
	low: "Low",
	medium: "Medium",
	high: "High",
	xhigh: "Max",
}

function formatCtxLabel(val: number): string {
	if (val >= 1_000_000) {
		return `${(val / 1_000_000).toFixed(val % 1_000_000 === 0 ? 0 : 1)}M`
	}
	if (val >= 1000) {
		return `${Math.round(val / 1024)}k`
	}
	return String(val)
}

/**
 * Infer context window from model ID naming convention (e.g. "Inkling-2-Mini-26B-32K-50toks" -> 32768,
 * "Ternary-Bonsai-2-27B-48K-45toks" -> 49152).
 */
function inferContextWindowFromModelId(modelId: string): number | undefined {
	const match = modelId.match(/[-_](\d+)K(?:[-_]|$)/i)
	if (match) {
		const kNum = Number.parseInt(match[1], 10)
		if (Number.isFinite(kNum) && kNum > 0) {
			return kNum * 1024
		}
	}
	return undefined
}

interface QuickModelContextControlsProps {
	mode: Mode
	cloudActOnly: boolean
	onOpenApiSettings: () => void
	section?: "model" | "controls" | "all"
}

export const QuickModelContextControls: React.FC<QuickModelContextControlsProps> = ({
	mode,
	cloudActOnly,
	onOpenApiSettings,
	section = "all",
}) => {
	const { apiConfiguration, planActSeparateModelsSetting, turnState, clineMessages = [] } = useExtensionState()
	const { selectedProvider, selectedModelId, selectedModelInfo } = useNormalizedApiConfiguration(mode)
	const { handleFieldChange, handleModeFieldChange } = useApiConfigurationHandlers()
	const { config, write, commitSelection, reasoningEffort } = useProviderConfig(selectedProvider)
	const { models: catalogModels, refresh: refreshCatalog } = useProviderModels(selectedProvider)
	const { orchestratorState, toggleOrchestrator } = useOrchestratorState()

	const [dynamicModels, setDynamicModels] = useState<string[]>([])
	const [isRefreshing, setIsRefreshing] = useState(false)

	// Controllo se il modello sta generando o pensando: in tal caso disabilitiamo il selettore
	const lastMessage = clineMessages[clineMessages.length - 1]
	const isStreamingLegacy =
		lastMessage?.partial === true || (lastMessage?.type === "say" && lastMessage.say === "api_req_started")
	const isGeneratingOrThinking = turnState?.phase === "streaming" || isStreamingLegacy

	const modeFields = getModeSpecificFields(apiConfiguration, mode)
	const activeReasoningEffort: ReasoningEffortSelection =
		reasoningEffort !== null && reasoningEffort !== undefined
			? reasoningEffort
			: isOpenaiReasoningEffort(modeFields.reasoningEffort)
				? modeFields.reasoningEffort
				: DEFAULT_REASONING_EFFORT_OPTION

	const isOpenAiCompat = selectedProvider === "openai" || (selectedProvider as string) === "openai-compatible"
	const isOllama = selectedProvider === "ollama"
	const isLmStudio = selectedProvider === "lmstudio"
	const supportsDynamicFetch = isOpenAiCompat || isOllama || isLmStudio

	const fetchDynamicModels = useCallback(async () => {
		if (cloudActOnly) {
			return
		}
		setIsRefreshing(true)
		try {
			if (isOpenAiCompat) {
				const baseUrl = config?.baseUrl || apiConfiguration?.openAiBaseUrl || "http://127.0.0.1:8080/v1"
				const apiKey = apiConfiguration?.openAiApiKey || ""
				if (baseUrl) {
					const res = await ModelsServiceClient.refreshOpenAiModels(
						OpenAiModelsRequest.create({ baseUrl, apiKey, providerId: selectedProvider }),
					)
					setDynamicModels(res.values ?? [])
				}
			} else if (isOllama) {
				const baseUrl = config?.baseUrl || apiConfiguration?.ollamaBaseUrl || ""
				const res = await ModelsServiceClient.getOllamaModels(StringRequest.create({ value: baseUrl }))
				setDynamicModels(res.values ?? [])
			} else if (isLmStudio) {
				const baseUrl = config?.baseUrl || apiConfiguration?.lmStudioBaseUrl || ""
				const res = await ModelsServiceClient.getLmStudioModels(StringRequest.create({ value: baseUrl }))
				setDynamicModels(res.values ?? [])
			} else {
				await refreshCatalog()
			}
		} catch (err) {
			console.error("Failed to fetch models for quick selector:", err)
		} finally {
			setIsRefreshing(false)
		}
	}, [
		cloudActOnly,
		isOpenAiCompat,
		isOllama,
		isLmStudio,
		config?.baseUrl,
		apiConfiguration?.openAiBaseUrl,
		apiConfiguration?.openAiApiKey,
		apiConfiguration?.ollamaBaseUrl,
		apiConfiguration?.lmStudioBaseUrl,
		refreshCatalog,
	])

	useEffect(() => {
		setDynamicModels([])
		if (supportsDynamicFetch) {
			void fetchDynamicModels()
		}
	}, [selectedProvider, config?.baseUrl, supportsDynamicFetch, fetchDynamicModels])

	const modelOptions = useMemo(() => {
		const set = new Set<string>()
		if (isOpenAiCompat) {
			// Per OpenAI / OpenAI-compatible (gateway locale), mostriamo SOLO i modelli
			// ritornati dall'endpoint dinamico (che filtra i modelli disabilitati e gpt-4o),
			// più il modello attualmente selezionato se presente (escluso gpt-4o).
			if (selectedModelId && selectedModelId !== "gpt-4o") {
				set.add(selectedModelId)
			}
			for (const id of dynamicModels) {
				if (id?.trim() && id.trim() !== "gpt-4o") {
					set.add(id.trim())
				}
			}
			// Non fare MAI fallback a catalogModels se contiene gpt-4o o se siamo su provider locale
			if (dynamicModels.length === 0) {
				for (const id of Object.keys(catalogModels)) {
					if (id?.trim() && id.trim() !== "gpt-4o") {
						set.add(id.trim())
					}
				}
			}
		} else {
			if (selectedModelId) {
				set.add(selectedModelId)
			}
			for (const id of dynamicModels) {
				if (id?.trim()) {
					set.add(id.trim())
				}
			}
			for (const id of Object.keys(catalogModels)) {
				if (id?.trim()) {
					set.add(id.trim())
				}
			}
		}
		return Array.from(set)
	}, [isOpenAiCompat, selectedModelId, dynamicModels, catalogModels])

	const committedSelection = mode === "plan" ? config?.planSelection : config?.actSelection
	const currentOverrides = useMemo(
		() => fromProtobufProviderModelOverrides(committedSelection?.overrides) ?? {},
		[committedSelection?.overrides],
	)

	const activeContextWindow = useMemo(() => {
		if (selectedModelId) {
			const inferred = inferContextWindowFromModelId(selectedModelId)
			if (inferred) {
				return inferred
			}
		}
		if (currentOverrides.contextWindow && currentOverrides.contextWindow > 0) {
			return currentOverrides.contextWindow
		}
		if (isOllama) {
			const ollamaCtx = config?.contextWindow || Number.parseInt(apiConfiguration?.ollamaApiOptionsCtxNum || "", 10)
			if (Number.isFinite(ollamaCtx) && ollamaCtx > 0) {
				return ollamaCtx
			}
		}
		if (selectedModelInfo?.contextWindow && selectedModelInfo.contextWindow > 0) {
			return selectedModelInfo.contextWindow
		}
		return 131072
	}, [
		selectedModelId,
		currentOverrides.contextWindow,
		isOllama,
		config?.contextWindow,
		apiConfiguration?.ollamaApiOptionsCtxNum,
		selectedModelInfo?.contextWindow,
	])

	const contextOptions = useMemo(() => {
		const hasActive = CONTEXT_WINDOW_PRESETS.some((p) => p.value === activeContextWindow)
		if (hasActive) {
			return CONTEXT_WINDOW_PRESETS
		}
		return [
			...CONTEXT_WINDOW_PRESETS,
			{ value: activeContextWindow, label: formatCtxLabel(activeContextWindow) },
		].sort((a, b) => a.value - b.value)
	}, [activeContextWindow])

	const commitForTargetModes = useCallback(
		async (modelId: string, overrides?: typeof currentOverrides) => {
			const modes: Mode[] = planActSeparateModelsSetting ? [mode] : ["plan", "act"]
			for (const targetMode of modes) {
				await commitSelection(targetMode, {
					providerId: selectedProvider,
					modelId,
					...(overrides !== undefined ? { overrides } : {}),
				})
			}
		},
		[commitSelection, mode, planActSeparateModelsSetting, selectedProvider],
	)

	const handleModelSelect = useCallback(
		async (e: React.ChangeEvent<HTMLSelectElement>) => {
			const nextModelId = e.target.value.trim()
			if (!nextModelId || nextModelId === selectedModelId) {
				return
			}
			try {
				if (isOpenAiCompat) {
					await handleModeFieldChange(
						{ plan: "planModeOpenAiModelId", act: "actModeOpenAiModelId" },
						nextModelId,
						mode,
					)
				} else if (isOllama) {
					await handleModeFieldChange(
						{ plan: "planModeOllamaModelId", act: "actModeOllamaModelId" },
						nextModelId,
						mode,
					)
				} else if (isLmStudio) {
					await handleModeFieldChange(
						{ plan: "planModeLmStudioModelId", act: "actModeLmStudioModelId" },
						nextModelId,
						mode,
					)
				} else {
					await handleModeFieldChange(
						{ plan: "planModeApiModelId", act: "actModeApiModelId" },
						nextModelId,
						mode,
					)
				}

				const inferredCtx = inferContextWindowFromModelId(nextModelId)
				const catalogCtx = catalogModels[nextModelId]?.contextWindow
				const targetCtx = inferredCtx ?? catalogCtx ?? currentOverrides.contextWindow

				const nextOverrides = {
					...currentOverrides,
					contextWindow: targetCtx,
				}

				await commitForTargetModes(nextModelId, nextOverrides)
			} catch (err) {
				console.error("Failed to switch model:", err)
			}
		},
		[
			catalogModels,
			commitForTargetModes,
			currentOverrides,
			handleModeFieldChange,
			isLmStudio,
			isOllama,
			isOpenAiCompat,
			mode,
			selectedModelId,
		],
	)

	const handleContextWindowSelect = useCallback(
		async (e: React.ChangeEvent<HTMLSelectElement>) => {
			const nextCtx = Number.parseInt(e.target.value, 10)
			if (!Number.isFinite(nextCtx) || nextCtx <= 0) {
				return
			}
			try {
				if (isOllama) {
					await write({ contextWindow: nextCtx })
					await handleFieldChange("ollamaApiOptionsCtxNum", String(nextCtx))
				}
				if (isLmStudio) {
					await handleFieldChange("lmStudioMaxTokens", String(nextCtx))
				}
				if (selectedModelId) {
					await commitForTargetModes(selectedModelId, {
						...currentOverrides,
						contextWindow: nextCtx,
					})
				}
			} catch (err) {
				console.error("Failed to update context window:", err)
			}
		},
		[commitForTargetModes, currentOverrides, handleFieldChange, isLmStudio, isOllama, selectedModelId, write],
	)

	const handleReasoningEffortSelect = useCallback(
		async (e: React.ChangeEvent<HTMLSelectElement>) => {
			const nextEffort = e.target.value as ReasoningEffortSelection
			try {
				if (isOpenaiReasoningEffort(nextEffort)) {
					await handleModeFieldChange(
						{ plan: "planModeReasoningEffort", act: "actModeReasoningEffort" },
						nextEffort,
						mode,
					)
				}
				await write(reasoningEffortPatch(nextEffort))
			} catch (err) {
				console.error("Failed to update reasoning effort:", err)
			}
		},
		[handleModeFieldChange, mode, write],
	)

	if (cloudActOnly) {
		return null
	}

	const showModelSection = section === "all" || section === "model"
	const showControlsSection = section === "all" || section === "controls"

	return (
		<div className={cn("flex items-center gap-1.5 min-w-0", showModelSection ? "flex-1 ml-1" : "flex-wrap")}>
			{showModelSection && (
				<>
					{/* Inline Model Select */}
					<Tooltip>
						<TooltipContent side="top">
							Cambia modello ({selectedProvider}) — {selectedModelId || "nessun modello"}
						</TooltipContent>
						<TooltipTrigger asChild>
							<select
								aria-label="Select AI model"
								className="h-6 flex-1 min-w-[130px] max-w-full truncate rounded-md bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.07] hover:border-white/[0.15] px-2 pr-5 font-geist-mono text-[11px] text-[#cfd4dc] hover:text-[#f2f4f7] focus:outline-none focus:border-white/[0.25] cursor-pointer transition-colors"
								data-testid="quick-model-select"
								onChange={handleModelSelect}
								style={{ colorScheme: "dark" }}
								value={selectedModelId || ""}>
								{modelOptions.length === 0 && <option value="">Nessun modello...</option>}
								{modelOptions.map((id) => (
									<option className="bg-[#151618] text-[#eceef0]" key={id} value={id}>
										{id}
									</option>
								))}
							</select>
						</TooltipTrigger>
					</Tooltip>

					{/* Refresh models button */}
					<Tooltip>
						<TooltipContent side="top">Aggiorna lista modelli dal server</TooltipContent>
						<TooltipTrigger asChild>
							<button
								aria-label="Refresh models"
								className="size-6 shrink-0 inline-flex items-center justify-center rounded-md bg-transparent hover:bg-white/[0.07] border-0 text-[#7c828b] hover:text-[#eceef0] cursor-pointer transition-colors"
								disabled={isRefreshing}
								onClick={() => void fetchDynamicModels()}
								type="button">
								<RefreshCwIcon className={isRefreshing ? "animate-spin" : ""} size={11.5} strokeWidth={1.9} />
							</button>
						</TooltipTrigger>
					</Tooltip>

					{/* Quick API Settings trigger */}
					<Tooltip>
						<TooltipContent side="top">Configurazione Provider & API</TooltipContent>
						<TooltipTrigger asChild>
							<button
								aria-label="Open API Settings"
								className="size-6 shrink-0 inline-flex items-center justify-center rounded-md bg-transparent hover:bg-white/[0.07] border-0 text-[#7c828b] hover:text-[#eceef0] cursor-pointer transition-colors"
								onClick={onOpenApiSettings}
								type="button">
								<SlidersHorizontalIcon size={11.5} strokeWidth={1.9} />
							</button>
						</TooltipTrigger>
					</Tooltip>
				</>
			)}

			{showControlsSection && (
				<>
					{/* Inline Context Window Select */}
					<Tooltip>
						<TooltipContent side="top">Context Window ({activeContextWindow.toLocaleString()} token)</TooltipContent>
						<TooltipTrigger asChild>
							<select
								aria-label="Context window size"
								className="h-6 shrink-0 rounded-md bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.07] hover:border-white/[0.15] px-1.5 font-geist-mono text-[11px] tabular-nums text-[#9ea4ad] hover:text-[#f2f4f7] focus:outline-none focus:border-white/[0.25] cursor-pointer transition-colors"
								data-testid="quick-ctx-select"
								onChange={handleContextWindowSelect}
								style={{ colorScheme: "dark" }}
								value={activeContextWindow}>
								{contextOptions.map((opt) => (
									<option className="bg-[#151618] text-[#eceef0]" key={opt.value} value={opt.value}>
										Ctx {opt.label}
									</option>
								))}
							</select>
						</TooltipTrigger>
					</Tooltip>

					{/* Inline Thinking / Reasoning Depth Select */}
					<Tooltip>
						<TooltipContent side="top">
							Livello di Thinking / Profondità di ragionamento
							{isGeneratingOrThinking ? " (non modificabile durante la generazione)" : ""}
						</TooltipContent>
						<TooltipTrigger asChild>
							<select
								aria-label="Thinking reasoning effort"
								className="h-6 shrink-0 rounded-md bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.07] hover:border-white/[0.15] px-1.5 font-geist-mono text-[11px] text-[#9ea4ad] hover:text-[#f2f4f7] focus:outline-none focus:border-white/[0.25] cursor-pointer transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
								data-testid="quick-thinking-select"
								disabled={isGeneratingOrThinking}
								onChange={handleReasoningEffortSelect}
								style={{ colorScheme: "dark" }}
								value={activeReasoningEffort}>
								<option className="bg-[#151618] text-[#eceef0]" value={DEFAULT_REASONING_EFFORT_OPTION}>
									Think: Def
								</option>
								{OPENAI_REASONING_EFFORT_OPTIONS.map((effort) => (
									<option className="bg-[#151618] text-[#eceef0]" key={effort} value={effort}>
										Think: {REASONING_EFFORT_LABELS[effort]}
									</option>
								))}
							</select>
						</TooltipTrigger>
					</Tooltip>

					{/* Orchestrator Toggle Button */}
					<Tooltip>
						<TooltipContent className="max-w-[330px] text-xs flex flex-col gap-1" side="top">
							<div className="font-semibold">
								Orchestratore AI: {orchestratorState.enabled ? "ATTIVO (Modalità Verbosa)" : "DISATTIVO"}
							</div>
							<div className="text-[11px] opacity-90">
								Supervisore: <code className="text-[#6ee7b7]">{orchestratorState.modelId}</code>
							</div>
							{(orchestratorState.planModelId || orchestratorState.actModelId) && (
								<div className="text-[11px] opacity-80">
									Plan: <code>{orchestratorState.planModelId ?? "default"}</code> • Act:{" "}
									<code>{orchestratorState.actModelId ?? "default"}</code>
								</div>
							)}
							{orchestratorState.status !== "idle" && (
								<div className="text-[11px] text-[#fde68a]">
									Stato live: {orchestratorState.phaseDetail || orchestratorState.status}
								</div>
							)}
							<div className="text-[11px] opacity-75">
								Mostra telemetria dietro le quinte (VRAM swap, TTFT, tok/s, ragionamento live), valuta i prompt, valida il Plan e preme automaticamente su Act (o invia correzioni).
							</div>
						</TooltipContent>
						<TooltipTrigger asChild>
							<button
								aria-label="Toggle AI Orchestrator"
								aria-pressed={orchestratorState.enabled}
								className={cn(
									"h-6 shrink-0 inline-flex items-center gap-1 rounded-md border px-2 font-geist-mono text-[11px] cursor-pointer transition-colors",
									orchestratorState.enabled
										? "bg-[#10b981]/15 hover:bg-[#10b981]/25 border-[#10b981]/45 text-[#6ee7b7]"
										: "bg-white/[0.04] hover:bg-white/[0.08] border-white/[0.07] hover:border-white/[0.15] text-[#9ea4ad] hover:text-[#f2f4f7]",
									orchestratorState.status !== "idle" && "animate-pulse",
								)}
								data-testid="orchestrator-toggle-button"
								onClick={toggleOrchestrator}
								type="button">
								<BotIcon size={11.5} strokeWidth={1.9} />
								<span>
									{orchestratorState.status !== "idle" && orchestratorState.phaseDetail
										? `Orch: ${orchestratorState.phaseDetail}`
										: orchestratorState.status === "evaluating_prompt"
											? "Orch: Prompt..."
											: orchestratorState.status === "evaluating_plan"
												? "Orch: Plan..."
												: orchestratorState.enabled
													? "Orch: ON"
													: "Orch: OFF"}
								</span>
							</button>
						</TooltipTrigger>
					</Tooltip>
				</>
			)}
		</div>
	)
}
