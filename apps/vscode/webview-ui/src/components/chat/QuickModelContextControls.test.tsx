import { act, fireEvent, render, screen } from "@testing-library/react"
import type { ReactNode } from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { QuickModelContextControls } from "./QuickModelContextControls"

const mocks = vi.hoisted(() => ({
	refreshOpenAiModels: vi.fn(),
	useExtensionState: vi.fn(),
	useNormalizedApiConfiguration: vi.fn(),
	useProviderConfig: vi.fn(),
	useProviderModels: vi.fn(),
	useOrchestratorState: vi.fn(),
	refreshCatalog: vi.fn(),
}))

vi.mock("@/context/ExtensionStateContext", () => ({
	useExtensionState: mocks.useExtensionState,
}))

vi.mock("@/hooks/useNormalizedApiConfiguration", () => ({
	useNormalizedApiConfiguration: mocks.useNormalizedApiConfiguration,
}))

vi.mock("@/components/settings/utils/useApiConfigurationHandlers", () => ({
	useApiConfigurationHandlers: () => ({
		handleFieldChange: vi.fn(),
		handleModeFieldChange: vi.fn(),
	}),
}))

vi.mock("@/hooks/useProviderConfig", () => ({
	fromProtobufProviderModelOverrides: () => undefined,
	useProviderConfig: mocks.useProviderConfig,
}))

vi.mock("@/hooks/useProviderModels", () => ({
	useProviderModels: mocks.useProviderModels,
}))

vi.mock("@/hooks/useOrchestratorState", () => ({
	useOrchestratorState: mocks.useOrchestratorState,
}))

vi.mock("@/services/grpc-client", () => ({
	ModelsServiceClient: {
		refreshOpenAiModels: mocks.refreshOpenAiModels,
		getOllamaModels: vi.fn(async () => ({ values: [] })),
		getLmStudioModels: vi.fn(async () => ({ values: [] })),
	},
}))

vi.mock("@/components/ui/tooltip", () => ({
	Tooltip: ({ children }: { children?: ReactNode }) => <>{children}</>,
	TooltipContent: () => null,
	TooltipTrigger: ({ children }: { children?: ReactNode }) => <>{children}</>,
}))

describe("QuickModelContextControls", () => {
	beforeEach(() => {
		vi.useFakeTimers()
		vi.clearAllMocks()
		mocks.refreshCatalog.mockResolvedValue(undefined)
		mocks.useExtensionState.mockReturnValue({
			apiConfiguration: { openAiBaseUrl: "http://127.0.0.1:8080/v1" },
			planActSeparateModelsSetting: true,
			turnState: undefined,
			clineMessages: [],
			showSettings: false,
		})
		mocks.useNormalizedApiConfiguration.mockReturnValue({
			selectedProvider: "openai",
			selectedModelId: "Qwen3.6-27B-CoderX-48K-115toks",
			selectedModelInfo: { contextWindow: 49152 },
		})
		mocks.useProviderConfig.mockReturnValue({
			config: { baseUrl: "http://127.0.0.1:8080/v1" },
			write: vi.fn(),
			commitSelection: vi.fn(),
			reasoningEffort: undefined,
		})
		mocks.useProviderModels.mockReturnValue({
			models: {},
			refresh: mocks.refreshCatalog,
		})
		mocks.useOrchestratorState.mockReturnValue({
			orchestratorState: { enabled: false, status: "idle", modelId: "TwIL-8B", revisionCount: 0, maxRevisions: 2 },
			toggleOrchestrator: vi.fn(),
		})
	})

	afterEach(() => {
		vi.useRealTimers()
	})

	it("retries automatically when initial model fetch fails and populates all models once server comes up", async () => {
		mocks.refreshOpenAiModels
			.mockResolvedValueOnce({ values: [] })
			.mockResolvedValueOnce({ values: ["Gemma4-12B-Agentic-47toks", "Qwen3.6-27B-CoderX-48K-115toks", "TwIL-8B-Reasoning-96K-80toks"] })

		render(<QuickModelContextControls cloudActOnly={false} mode="act" onOpenApiSettings={vi.fn()} section="model" />)
		await act(async () => {})

		const select = screen.getByTestId("quick-model-select")
		expect(select.querySelectorAll("option")).toHaveLength(1)

		await act(async () => {
			vi.advanceTimersByTime(2500)
		})

		expect(mocks.refreshOpenAiModels).toHaveBeenCalledTimes(2)
		const options = Array.from(select.querySelectorAll("option")).map((o) => o.value)
		expect(options).toEqual(["Qwen3.6-27B-CoderX-48K-115toks", "Gemma4-12B-Agentic-47toks", "TwIL-8B-Reasoning-96K-80toks"])
	})

	it("refreshes empty model list when user interacts with the dropdown", async () => {
		mocks.refreshOpenAiModels.mockResolvedValueOnce({ values: [] }).mockResolvedValueOnce({
			values: ["Model-A", "Model-B"],
		})

		render(<QuickModelContextControls cloudActOnly={false} mode="act" onOpenApiSettings={vi.fn()} section="model" />)
		await act(async () => {})

		await act(async () => {
			vi.advanceTimersByTime(1100)
		})

		const select = screen.getByTestId("quick-model-select")
		await act(async () => {
			fireEvent.mouseDown(select)
		})

		expect(mocks.refreshOpenAiModels).toHaveBeenCalledTimes(2)
		const options = Array.from(select.querySelectorAll("option")).map((o) => o.value)
		expect(options).toContain("Model-A")
		expect(options).toContain("Model-B")
	})

	it("does not fetch dynamic models for section='controls'", async () => {
		mocks.refreshOpenAiModels.mockResolvedValue({ values: ["Model-A"] })

		render(<QuickModelContextControls cloudActOnly={false} mode="act" onOpenApiSettings={vi.fn()} section="controls" />)
		await act(async () => {})

		expect(mocks.refreshOpenAiModels).not.toHaveBeenCalled()
	})
})
