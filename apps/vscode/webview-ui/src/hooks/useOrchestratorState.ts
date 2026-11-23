import type { OrchestratorUiState } from "@shared/ExtensionMessage"
import { useCallback, useEffect, useState } from "react"
import { PLATFORM_CONFIG } from "@/config/platform.config"
import { useExtensionState } from "@/context/ExtensionStateContext"

const DEFAULT_ORCHESTRATOR_STATE: OrchestratorUiState = {
	enabled: false,
	modelId: "Qwen3.6-35B-Coding-Avanzato-60toks",
	status: "idle",
	revisionCount: 0,
	maxRevisions: 3,
}

/**
 * Syncs the LocalAI Orchestrator state between the Cline webview UI and the VS Code extension host.
 */
export function useOrchestratorState() {
	const { orchestratorState: extOrchestratorState } = useExtensionState()
	const [localState, setLocalState] = useState<OrchestratorUiState>(
		extOrchestratorState ?? DEFAULT_ORCHESTRATOR_STATE,
	)

	useEffect(() => {
		if (extOrchestratorState) {
			setLocalState(extOrchestratorState)
		}
	}, [extOrchestratorState])

	useEffect(() => {
		const handleMessage = (event: MessageEvent) => {
			const msg = event.data
			if (msg?.type === "studio_orchestrator_state" && msg.orchestratorState) {
				setLocalState(msg.orchestratorState as OrchestratorUiState)
			}
		}

		window.addEventListener("message", handleMessage)
		PLATFORM_CONFIG.postMessage({ type: "studio_get_orchestrator_state" })
		return () => window.removeEventListener("message", handleMessage)
	}, [])

	const toggleOrchestrator = useCallback(() => {
		const nextEnabled = !localState.enabled
		setLocalState((prev) => ({ ...prev, enabled: nextEnabled }))
		PLATFORM_CONFIG.postMessage({
			type: "studio_set_orchestrator_state",
			orchestratorEnabled: nextEnabled,
		})
	}, [localState.enabled])

	return {
		orchestratorState: localState,
		toggleOrchestrator,
	}
}
