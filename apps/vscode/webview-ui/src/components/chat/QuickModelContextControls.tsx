import type { OrchestratorVerdictType } from "@shared/ExtensionMessage"
import { StringRequest } from "@shared/proto/cline/common"
import { OpenAiModelsRequest } from "@shared/proto/cline/models"
import {
	isOpenaiReasoningEffort,
	type Mode,
	OPENAI_REASONING_EFFORT_OPTIONS,
	type OpenaiReasoningEffort,
} from "@shared/storage/types"
import { BotIcon, RefreshCwIcon, ShieldAlertIcon, ShieldCheckIcon, SlidersHorizontalIcon } from "lucide-react"
import type React from "react"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
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

/** Default untouched contextWindow set internally by Cline on OpenAI-compatible models (128,000 !== 131,072). */
const CLINE_DEFAULT_OPENAI_CTX = 128_000

const CONTEXT_WINDOW_PRESETS = [
	{ value: 4096, label: "4k" },
	{ value: 8192, label: "8k" },
	{ value: 16384, label: "16k" },
	{ value: 24576, label: "24k" },
	{ value: 32768, label: "32k" },
	{ value: 43008, label: "42k" },
	{ value: 49152, label: "48k" },
	{ value: 65536, label: "64k" },
	{ value: 98304, label: "96k" },
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

const VERDICT_META: Record<
	OrchestratorVerdictType,
	{ label: string; tone: "positive" | "warning" | "danger" }
> = {
	prompt_ready: { label: "Specifica Prompt Pronta", tone: "positive" },
	approved: { label: "Plan Approvato", tone: "positive" },
	info_approved: { label: "Risposta Validata", tone: "positive" },
	act_completed: { label: "Lavoro Act Verificato", tone: "positive" },
	revision_requested: { label: "Plan da Correggere", tone: "warning" },
	accepted_fallback: { label: "Plan Accettato con Riserva", tone: "warning" },
	act_needs_work: { label: "Act da Rifinire", tone: "warning" },
	blocked: { label: "Plan Bloccato (Richiesto Utente)", tone: "danger" },
}

function formatCtxLabel(val: number): string {
	if (val >= 1_000_000) {
		if (val % 1_048_576 === 0) {
			return `${val / 1_048_576}M`
		}
		return `${(val / 1_000_000).toFixed(val % 1_000_000 === 0 ? 0 : 1)}M`
	}
	if (val >= 1000) {
		if (val % 1024 === 0) {
			return `${val / 1024}k`
		}
		if (val % 1000 === 0) {
			return `${val / 1000}k`
		}
		return `${Math.round(val / 1024)}k`
	}
	return String(val)
}

/**
 * Infer context window from model ID naming conventions such as:
 * "TwIL-8B-Reasoning-96K-80toks" -> 98304, "Qwen3.6-27B-CoderX-48K-115toks" -> 49152, "qwen:32k" -> 32768.
 */
function inferContextWindowFromModelId(modelId: string): number | undefined {
	const kMatch = modelId.match(/(?:^|[-_:/\s(])(\d+)k(?:[-_:/\s)]|$)/i)
	if (kMatch) {
		const kNum = Number.parseInt(kMatch[1], 10)
		if (Number.isFinite(kNum) && kNum > 0) {
			return kNum * 1024
		}
	}
	const mMatch = modelId.match(/(?:^|[-_:/\s(])(\d+)m(?:[-_:/\s)]|$)/i)
	if (mMatch) {
		const mNum = Number.parseInt(mMatch[1], 10)
		if (Number.isFinite(mNum) && mNum > 0 && mNum <= 10) {
			return mNum * 1_000_000
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
	const { apiConfiguration, planActSeparateModelsSetting, turnState, clineMessages = [], showSettings } = useExtensionState()
	const { selectedProvider, selectedModelId, selectedModelInfo } = useNormalizedApiConfiguration(mode)
	const { handleFieldChange, handleModeFieldChange } = useApiConfigurationHandlers()
	const { config, write, commitSelection, reasoningEffort } = useProviderConfig(selectedProvider)
	const { models: catalogModels, refresh: refreshCatalog } = useProviderModels(selectedProvider)
	const { orchestratorState, toggleOrchestrator } = useOrchestratorState()

	const showModelSection = section === "all" || section === "model"
	const showControlsSection = section === "all" || section === "controls"

	const [dynamicModels, setDynamicModels] = useState<string[]>([])
	const [isRefreshing, setIsRefreshing] = useState(false)
	const [optimisticCtx, setOptimisticCtx] = useState<{ key: string; value: number } | null>(null)

	const fetchSeqRef = useRef(0)
	const lastFetchAtRef = useRef(0)
	const isRefreshingRef = useRef(false)
	const refreshCatalogRef = useRef(refreshCatalog)
	refreshCatalogRef.current = refreshCatalog
	const prevShowSettingsRef = useRef(showSettings)
	const prevEndpointKeyRef = useRef<string>("")
	const retryCountRef = useRef(0)
	const manualCtxByKeyRef = useRef<Map<string, number>>(new Map())
	const pendingModelCtxSyncRef = useRef<{ modelId: string; mode: Mode; ctx: number } | null>(null)
	const autoSyncedModelKeyRef = useRef<string>("")

	const isStreamingTurn = useMemo(() => {
		if (turnState !== undefined) {
			return turnState.phase === "streaming"
		}
		const lastMessage = clineMessages[clineMessages.length - 1]
		if (!lastMessage) {
			return false
		}
		if (lastMessage.partial === true) {
			return true
		}
		if (lastMessage.type === "say" && lastMessage.say === "api_req_started") {
			if (!lastMessage.text) {
				return true
			}
			try {
				const info = JSON.parse(lastMessage.text) as {
					cancelReason?: string
					streamingFailedMessage?: string
					tokensIn?: number
					tokensOut?: number
					cost?: number
				}
				if (
					info.cancelReason ||
					info.streamingFailedMessage ||
					info.tokensIn !== undefined ||
					info.tokensOut !== undefined ||
					info.cost !== undefined
				) {
					return false
				}
			} catch {
				// Ignore non-JSON payload
			}
			return true
		}
		return false
	}, [turnState, clineMessages])

	const isGeneratingOrThinking = isStreamingTurn || orchestratorState.status !== "idle"

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

	const effectiveOpenAiBaseUrl = config?.baseUrl || apiConfiguration?.openAiBaseUrl || "http://127.0.0.1:8080/v1"
	const effectiveDynamicBaseUrl = isOpenAiCompat
		? effectiveOpenAiBaseUrl
		: isOllama
			? config?.baseUrl || apiConfiguration?.ollamaBaseUrl || ""
			: isLmStudio
				? config?.baseUrl || apiConfiguration?.lmStudioBaseUrl || ""
				: ""
	const endpointKey = `${selectedProvider}:${effectiveDynamicBaseUrl}`

	const ctxSlotKey = `${planActSeparateModelsSetting ? mode : "shared"}:${selectedProvider}:${selectedModelId || ""}`

	const fetchDynamicModels = useCallback(async () => {
		if (cloudActOnly || !showModelSection) {
			return
		}
		const reqId = ++fetchSeqRef.current
		lastFetchAtRef.current = Date.now()
		isRefreshingRef.current = true
		setIsRefreshing(true)
		try {
			if (isOpenAiCompat) {
				const baseUrl = effectiveOpenAiBaseUrl
				const apiKey = apiConfiguration?.openAiApiKey || ""
				if (baseUrl) {
					const res = await ModelsServiceClient.refreshOpenAiModels(
						OpenAiModelsRequest.create({ baseUrl, apiKey, providerId: selectedProvider }),
					)
					if (reqId === fetchSeqRef.current) {
						const nextValues = res.values ?? []
						if (nextValues.length > 0) {
							retryCountRef.current = 0
							setDynamicModels(nextValues)
						}
					}
				}
			} else if (isOllama) {
				const baseUrl = config?.baseUrl || apiConfiguration?.ollamaBaseUrl || ""
				const res = await ModelsServiceClient.getOllamaModels(StringRequest.create({ value: baseUrl }))
				if (reqId === fetchSeqRef.current) {
					const nextValues = res.values ?? []
					if (nextValues.length > 0) {
						retryCountRef.current = 0
						setDynamicModels(nextValues)
					}
				}
			} else if (isLmStudio) {
				const baseUrl = config?.baseUrl || apiConfiguration?.lmStudioBaseUrl || ""
				const res = await ModelsServiceClient.getLmStudioModels(StringRequest.create({ value: baseUrl }))
				if (reqId === fetchSeqRef.current) {
					const nextValues = res.values ?? []
					if (nextValues.length > 0) {
						retryCountRef.current = 0
						setDynamicModels(nextValues)
					}
				}
			} else {
				await refreshCatalogRef.current()
			}
		} catch (err) {
			console.error("Failed to fetch models for quick selector:", err)
		} finally {
			if (reqId === fetchSeqRef.current) {
				isRefreshingRef.current = false
				setIsRefreshing(false)
			}
		}
	}, [
		cloudActOnly,
		showModelSection,
		isOpenAiCompat,
		isOllama,
		isLmStudio,
		selectedProvider,
		effectiveOpenAiBaseUrl,
		config?.baseUrl,
		apiConfiguration?.openAiApiKey,
		apiConfiguration?.ollamaBaseUrl,
		apiConfiguration?.lmStudioBaseUrl,
	])

	useEffect(() => {
		if (!showModelSection || cloudActOnly) {
			return
		}
		if (prevEndpointKeyRef.current !== endpointKey) {
			prevEndpointKeyRef.current = endpointKey
			fetchSeqRef.current += 1
			retryCountRef.current = 0
			setDynamicModels([])
		}
		if (supportsDynamicFetch) {
			void fetchDynamicModels()
		}
	}, [showModelSection, cloudActOnly, endpointKey, supportsDynamicFetch, fetchDynamicModels])

	// Bounded automatic retries when the local model server starts slightly after Cline Studio.
	useEffect(() => {
		if (!showModelSection || cloudActOnly || !supportsDynamicFetch || dynamicModels.length > 0 || isRefreshing) {
			return
		}
		if (retryCountRef.current >= 4) {
			return
		}
		const delayMs = [2500, 5000, 8000, 12000][retryCountRef.current] ?? 12000
		const timer = setTimeout(() => {
			if (typeof document !== "undefined" && document.hidden) {
				return
			}
			retryCountRef.current += 1
			void fetchDynamicModels()
		}, delayMs)
		return () => clearTimeout(timer)
	}, [showModelSection, cloudActOnly, supportsDynamicFetch, dynamicModels.length, isRefreshing, fetchDynamicModels])

	// Refresh models when regaining focus/visibility if empty or stale.
	useEffect(() => {
		if (!showModelSection || cloudActOnly || !supportsDynamicFetch) {
			return
		}
		const maybeRefreshOnResume = () => {
			if ((typeof document !== "undefined" && document.hidden) || isRefreshingRef.current) {
				return
			}
			const elapsed = Date.now() - lastFetchAtRef.current
			if ((dynamicModels.length === 0 && elapsed > 1500) || elapsed > 15_000) {
				void fetchDynamicModels()
			}
		}
		window.addEventListener("focus", maybeRefreshOnResume)
		document.addEventListener("visibilitychange", maybeRefreshOnResume)
		return () => {
			window.removeEventListener("focus", maybeRefreshOnResume)
			document.removeEventListener("visibilitychange", maybeRefreshOnResume)
		}
	}, [showModelSection, cloudActOnly, supportsDynamicFetch, dynamicModels.length, fetchDynamicModels])

	// Refresh after returning from Settings so URL/key changes or newly started servers take effect immediately.
	useEffect(() => {
		const wasOpen = prevShowSettingsRef.current
		prevShowSettingsRef.current = showSettings
		if (wasOpen && !showSettings && showModelSection && !cloudActOnly && supportsDynamicFetch) {
			void fetchDynamicModels()
		}
	}, [showSettings, showModelSection, cloudActOnly, supportsDynamicFetch, fetchDynamicModels])

	const handleModelSelectInteract = useCallback(() => {
		if (!supportsDynamicFetch || cloudActOnly || isRefreshingRef.current) {
			return
		}
		const elapsed = Date.now() - lastFetchAtRef.current
		if ((dynamicModels.length === 0 && elapsed > 1000) || elapsed > 10_000) {
			void fetchDynamicModels()
		}
	}, [supportsDynamicFetch, cloudActOnly, dynamicModels.length, fetchDynamicModels])

	const modelOptions = useMemo(() => {
		const set = new Set<string>()
		if (isOpenAiCompat) {
			if (selectedModelId && selectedModelId !== "gpt-4o") {
				set.add(selectedModelId)
			}
			for (const id of dynamicModels) {
				if (id?.trim() && id.trim() !== "gpt-4o") {
					set.add(id.trim())
				}
			}
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
	const currentOverrides = useMemo(() => {
		if (committedSelection?.modelId && selectedModelId && committedSelection.modelId !== selectedModelId) {
			return {}
		}
		return fromProtobufProviderModelOverrides(committedSelection?.overrides) ?? {}
	}, [committedSelection?.modelId, committedSelection?.overrides, selectedModelId])

	const inferredModelCtx = useMemo(
		() => (selectedModelId ? inferContextWindowFromModelId(selectedModelId) : undefined),
		[selectedModelId],
	)

	const configuredBackendCtx = useMemo(() => {
		if (isOpenAiCompat) {
			return modeFields.openAiModelInfo?.contextWindow
		}
		if (isOllama) {
			const parsed = Number.parseInt(apiConfiguration?.ollamaApiOptionsCtxNum || "", 10)
			return Number.isFinite(parsed) && parsed > 0 ? parsed : config?.contextWindow
		}
		if (isLmStudio) {
			const parsed = Number.parseInt(apiConfiguration?.lmStudioMaxTokens || "", 10)
			return Number.isFinite(parsed) && parsed > 0 ? parsed : config?.contextWindow
		}
		return currentOverrides.contextWindow
	}, [
		isOpenAiCompat,
		isOllama,
		isLmStudio,
		modeFields.openAiModelInfo?.contextWindow,
		apiConfiguration?.ollamaApiOptionsCtxNum,
		apiConfiguration?.lmStudioMaxTokens,
		config?.contextWindow,
		currentOverrides.contextWindow,
	])

	const activeContextWindow = useMemo(() => {
		if (optimisticCtx && optimisticCtx.key === ctxSlotKey && optimisticCtx.value > 0) {
			return optimisticCtx.value
		}

		const manualChoice = manualCtxByKeyRef.current.get(ctxSlotKey)
		if (manualChoice && manualChoice > 0) {
			return manualChoice
		}

		if (currentOverrides.contextWindow && currentOverrides.contextWindow > 0) {
			return currentOverrides.contextWindow
		}

		if (isOpenAiCompat) {
			const openAiCtx = modeFields.openAiModelInfo?.contextWindow
			// If OpenAI modelInfo was explicitly set (not Cline's untouched 128_000 default), respect it over filename inference
			if (openAiCtx && openAiCtx > 0 && openAiCtx !== CLINE_DEFAULT_OPENAI_CTX) {
				return openAiCtx
			}
			if (inferredModelCtx) {
				return inferredModelCtx
			}
			if (openAiCtx === CLINE_DEFAULT_OPENAI_CTX) {
				return 131072
			}
		} else if (isOllama) {
			const ollamaCtx = Number.parseInt(apiConfiguration?.ollamaApiOptionsCtxNum || "", 10) || config?.contextWindow
			if (ollamaCtx && Number.isFinite(ollamaCtx) && ollamaCtx > 0) {
				return ollamaCtx
			}
			if (inferredModelCtx) {
				return inferredModelCtx
			}
		} else if (isLmStudio) {
			const lmStudioCtx = Number.parseInt(apiConfiguration?.lmStudioMaxTokens || "", 10) || config?.contextWindow
			if (lmStudioCtx && Number.isFinite(lmStudioCtx) && lmStudioCtx > 0) {
				return lmStudioCtx
			}
			if (inferredModelCtx) {
				return inferredModelCtx
			}
		} else if (inferredModelCtx) {
			return inferredModelCtx
		}

		if (selectedModelInfo?.contextWindow && selectedModelInfo.contextWindow > 0) {
			return selectedModelInfo.contextWindow === CLINE_DEFAULT_OPENAI_CTX ? 131072 : selectedModelInfo.contextWindow
		}
		return 131072
	}, [
		optimisticCtx,
		ctxSlotKey,
		currentOverrides.contextWindow,
		isOpenAiCompat,
		isOllama,
		isLmStudio,
		modeFields.openAiModelInfo?.contextWindow,
		apiConfiguration?.ollamaApiOptionsCtxNum,
		apiConfiguration?.lmStudioMaxTokens,
		config?.contextWindow,
		inferredModelCtx,
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

	const persistContextWindowToBackend = useCallback(
		async (targetModelId: string, nextCtx: number, targetMode: Mode = mode) => {
			await write({ contextWindow: nextCtx })

			if (targetModelId) {
				await commitForTargetModes(targetModelId, {
					...currentOverrides,
					contextWindow: nextCtx,
				})
			}

			if (isOpenAiCompat) {
				const targetModeFields = getModeSpecificFields(apiConfiguration, targetMode)
				const existingOpenAiInfo = targetModeFields.openAiModelInfo
				const updatedOpenAiInfo = {
					...selectedModelInfo,
					...existingOpenAiInfo,
					maxTokens: existingOpenAiInfo?.maxTokens ?? selectedModelInfo.maxTokens ?? -1,
					supportsImages: existingOpenAiInfo?.supportsImages ?? selectedModelInfo.supportsImages ?? true,
					supportsPromptCache:
						existingOpenAiInfo?.supportsPromptCache ?? selectedModelInfo.supportsPromptCache ?? false,
					contextWindow: nextCtx,
				}
				await handleModeFieldChange(
					{ plan: "planModeOpenAiModelInfo", act: "actModeOpenAiModelInfo" },
					updatedOpenAiInfo,
					targetMode,
				)
			} else if (isOllama) {
				await handleFieldChange("ollamaApiOptionsCtxNum", String(nextCtx))
			} else if (isLmStudio) {
				await handleFieldChange("lmStudioMaxTokens", String(nextCtx))
			}
		},
		[
			apiConfiguration,
			commitForTargetModes,
			currentOverrides,
			handleFieldChange,
			handleModeFieldChange,
			isLmStudio,
			isOllama,
			isOpenAiCompat,
			mode,
			selectedModelInfo,
			write,
		],
	)

	// Clear optimistic state once backend config reflects the selected value
	useEffect(() => {
		if (optimisticCtx && configuredBackendCtx === optimisticCtx.value) {
			setOptimisticCtx(null)
		}
	}, [optimisticCtx, configuredBackendCtx])

	// Apply queued context window sync right after a model switch updates apiConfiguration (avoids stale closure overwrites)
	useEffect(() => {
		const pending = pendingModelCtxSyncRef.current
		if (pending && selectedModelId === pending.modelId) {
			pendingModelCtxSyncRef.current = null
			autoSyncedModelKeyRef.current = ctxSlotKey
			void persistContextWindowToBackend(pending.modelId, pending.ctx, pending.mode).catch((err) => {
				console.error("Failed to sync context window after model switch:", err)
			})
			return
		}

		// Auto-sync inferred model Ctx (e.g., -48K- or -96K-) on initial mount if backend still holds untouched default
		if (
			selectedModelId &&
			inferredModelCtx &&
			autoSyncedModelKeyRef.current !== ctxSlotKey &&
			configuredBackendCtx !== activeContextWindow &&
			(configuredBackendCtx === undefined || configuredBackendCtx === CLINE_DEFAULT_OPENAI_CTX)
		) {
			autoSyncedModelKeyRef.current = ctxSlotKey
			void persistContextWindowToBackend(selectedModelId, activeContextWindow, mode).catch((err) => {
				console.error("Failed to auto-sync inferred context window:", err)
			})
		}
	}, [
		selectedModelId,
		ctxSlotKey,
		inferredModelCtx,
		configuredBackendCtx,
		activeContextWindow,
		mode,
		persistContextWindowToBackend,
	])

	const handleModelSelect = useCallback(
		async (e: React.ChangeEvent<HTMLSelectElement>) => {
			const nextModelId = e.target.value.trim()
			if (!nextModelId || nextModelId === selectedModelId) {
				return
			}

			const nextSlotKey = `${planActSeparateModelsSetting ? mode : "shared"}:${selectedProvider}:${nextModelId}`
			const inferredCtx = inferContextWindowFromModelId(nextModelId)
			const catalogCtx = catalogModels[nextModelId]?.contextWindow
			const targetCtx = inferredCtx ?? catalogCtx ?? activeContextWindow

			manualCtxByKeyRef.current.delete(nextSlotKey)
			setOptimisticCtx({ key: nextSlotKey, value: targetCtx })
			pendingModelCtxSyncRef.current = { modelId: nextModelId, mode, ctx: targetCtx }

			try {
				const { contextWindow: _prevCtx, ...restOverrides } = currentOverrides
				const nextOverrides = { ...restOverrides, contextWindow: targetCtx }

				await write({ contextWindow: targetCtx })
				await commitForTargetModes(nextModelId, nextOverrides)

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
			} catch (err) {
				pendingModelCtxSyncRef.current = null
				setOptimisticCtx(null)
				console.error("Failed to switch model:", err)
			}
		},
		[
			activeContextWindow,
			catalogModels,
			commitForTargetModes,
			currentOverrides,
			handleModeFieldChange,
			isLmStudio,
			isOllama,
			isOpenAiCompat,
			mode,
			planActSeparateModelsSetting,
			selectedModelId,
			selectedProvider,
			write,
		],
	)

	const handleContextWindowSelect = useCallback(
		async (e: React.ChangeEvent<HTMLSelectElement>) => {
			const nextCtx = Number.parseInt(e.target.value, 10)
			if (!Number.isFinite(nextCtx) || nextCtx <= 0) {
				return
			}

			manualCtxByKeyRef.current.set(ctxSlotKey, nextCtx)
			setOptimisticCtx({ key: ctxSlotKey, value: nextCtx })

			try {
				await persistContextWindowToBackend(selectedModelId || "", nextCtx, mode)
			} catch (err) {
				setOptimisticCtx(null)
				console.error("Failed to update context window:", err)
			}
		},
		[ctxSlotKey, mode, persistContextWindowToBackend, selectedModelId],
	)

	const handleReasoningEffortSelect = useCallback(
		async (e: React.ChangeEvent<HTMLSelectElement>) => {
			const nextEffort = e.target.value as ReasoningEffortSelection
			try {
				await handleModeFieldChange(
					{ plan: "planModeReasoningEffort", act: "actModeReasoningEffort" },
					isOpenaiReasoningEffort(nextEffort) ? nextEffort : undefined,
					mode,
				)
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

	const lastEval = orchestratorState.lastEvaluation
	const verdictMeta = lastEval ? VERDICT_META[lastEval.verdict] : undefined

	const orchButtonToneClass = !orchestratorState.enabled
		? "bg-white/[0.04] hover:bg-white/[0.08] border-white/[0.07] hover:border-white/[0.15] text-[#9ea4ad] hover:text-[#f2f4f7]"
		: orchestratorState.status !== "idle"
			? "bg-[#f59e0b]/15 hover:bg-[#f59e0b]/25 border-[#f59e0b]/45 text-[#fde68a] animate-pulse"
			: verdictMeta?.tone === "danger"
				? "bg-[#ef4444]/15 hover:bg-[#ef4444]/25 border-[#ef4444]/50 text-[#fca5a5]"
				: verdictMeta?.tone === "warning"
					? "bg-[#f59e0b]/15 hover:bg-[#f59e0b]/25 border-[#f59e0b]/45 text-[#fcd34d]"
					: "bg-[#10b981]/15 hover:bg-[#10b981]/25 border-[#10b981]/45 text-[#6ee7b7]"

	const orchButtonLabel =
		orchestratorState.status !== "idle" && orchestratorState.phaseDetail
			? `Orch: ${orchestratorState.phaseDetail}`
			: orchestratorState.status === "evaluating_prompt"
				? "Orch: Prompt..."
				: orchestratorState.status === "evaluating_plan"
					? "Orch: Plan..."
					: orchestratorState.status === "evaluating_act"
						? "Orch: Act..."
						: !orchestratorState.enabled
							? "Orch: OFF"
							: lastEval
								? `Orch: ${lastEval.verdict === "blocked" ? "STOP" : "ON"} • ${lastEval.score}/10`
								: "Orch: ON"

	return (
		<div className={cn("flex items-center gap-1.5 min-w-0", showModelSection ? "flex-1 ml-1" : "flex-wrap")}>
			{showModelSection && (
				<>
					{/* Inline Model Select */}
					<Tooltip>
						<TooltipContent side="top">
							Cambia modello ({selectedProvider}) — {selectedModelId || "nessun modello"}
							{isGeneratingOrThinking ? " (non modificabile durante la generazione)" : ""}
						</TooltipContent>
						<TooltipTrigger asChild>
							<select
								aria-label="Select AI model"
								className="h-6 flex-1 min-w-[130px] max-w-full truncate rounded-md bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.07] hover:border-white/[0.15] px-2 pr-5 font-geist-mono text-[11px] text-[#cfd4dc] hover:text-[#f2f4f7] focus:outline-none focus:border-white/[0.25] cursor-pointer transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
								data-testid="quick-model-select"
								disabled={isGeneratingOrThinking}
								onChange={handleModelSelect}
								onFocus={handleModelSelectInteract}
								onMouseDown={handleModelSelectInteract}
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
								className="size-6 shrink-0 inline-flex items-center justify-center rounded-md bg-transparent hover:bg-white/[0.07] border-0 text-[#7c828b] hover:text-[#eceef0] cursor-pointer transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
								disabled={isRefreshing || isGeneratingOrThinking}
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
						<TooltipContent side="top">
							Context Window attiva: {activeContextWindow.toLocaleString()} token ({formatCtxLabel(activeContextWindow)})
							{isGeneratingOrThinking ? " — non modificabile durante la generazione" : ""}
						</TooltipContent>
						<TooltipTrigger asChild>
							<select
								aria-label="Context window size"
								className="h-6 shrink-0 rounded-md bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.07] hover:border-white/[0.15] px-1.5 font-geist-mono text-[11px] tabular-nums text-[#9ea4ad] hover:text-[#f2f4f7] focus:outline-none focus:border-white/[0.25] cursor-pointer transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
								data-testid="quick-ctx-select"
								disabled={isGeneratingOrThinking}
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

					{/* Orchestrator Toggle & Supervisor Verdict Button */}
					<Tooltip>
						<TooltipContent className="max-w-[360px] text-xs flex flex-col gap-1.5 p-2.5" side="top">
							<div className="flex items-center justify-between gap-2 font-semibold">
								<span>Orchestratore AI: {orchestratorState.enabled ? "ATTIVO" : "DISATTIVO"}</span>
								{orchestratorState.revisionCount > 0 && (
									<span className="rounded bg-white/[0.08] px-1.5 py-0.5 font-geist-mono text-[10px] text-[#fcd34d]">
										Rev {orchestratorState.revisionCount}/{orchestratorState.maxRevisions}
									</span>
								)}
							</div>

							{lastEval && verdictMeta && (
								<div
									className={cn(
										"rounded-md border p-2 flex flex-col gap-1",
										verdictMeta.tone === "danger"
											? "bg-[#ef4444]/15 border-[#ef4444]/40 text-[#fecaca]"
											: verdictMeta.tone === "warning"
												? "bg-[#f59e0b]/15 border-[#f59e0b]/40 text-[#fef3c7]"
												: "bg-[#10b981]/15 border-[#10b981]/40 text-[#d1fae5]",
									)}>
									<div className="flex items-center justify-between gap-2 font-semibold text-[11px]">
										<span className="inline-flex items-center gap-1">
											{verdictMeta.tone === "positive" ? (
												<ShieldCheckIcon size={12} />
											) : (
												<ShieldAlertIcon size={12} />
											)}
											Valutazione del Supervisore: {verdictMeta.label}
										</span>
										<span className="font-geist-mono font-bold">{lastEval.score}/10</span>
									</div>
									<div className="text-[11px] leading-snug opacity-95 line-clamp-4">{lastEval.summary}</div>
								</div>
							)}

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
								<div className="text-[11px] font-medium text-[#fde68a]">
									⏳ Stato live: {orchestratorState.phaseDetail || orchestratorState.status}
								</div>
							)}
							<div className="text-[10.5px] opacity-70">
								Clicca per {orchestratorState.enabled ? "disattivare" : "attivare"} la supervisione automatica di Prompt, Plan e Act.
							</div>
						</TooltipContent>
						<TooltipTrigger asChild>
							<button
								aria-label="Toggle AI Orchestrator"
								aria-pressed={orchestratorState.enabled}
								className={cn(
									"h-6 shrink-0 max-w-[180px] inline-flex items-center gap-1 rounded-md border px-2 font-geist-mono text-[11px] tabular-nums cursor-pointer transition-colors",
									orchButtonToneClass,
								)}
								data-testid="orchestrator-toggle-button"
								onClick={toggleOrchestrator}
								type="button">
								{orchestratorState.enabled && lastEval && orchestratorState.status === "idle" ? (
									verdictMeta?.tone === "positive" ? (
										<ShieldCheckIcon className="shrink-0" size={11.5} strokeWidth={2} />
									) : (
										<ShieldAlertIcon className="shrink-0" size={11.5} strokeWidth={2} />
									)
								) : (
									<BotIcon className="shrink-0" size={11.5} strokeWidth={1.9} />
								)}
								<span className="truncate">{orchButtonLabel}</span>
							</button>
						</TooltipTrigger>
					</Tooltip>
				</>
			)}
		</div>
	)
}