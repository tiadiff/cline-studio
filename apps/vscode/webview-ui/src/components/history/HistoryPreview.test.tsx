import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import HistoryPreview from "./HistoryPreview"

const mocks = vi.hoisted(() => ({
	useExtensionState: vi.fn(),
	isCostVisible: vi.fn(() => true),
}))

vi.mock("@/context/ExtensionStateContext", () => ({
	useExtensionState: mocks.useExtensionState,
}))

vi.mock("@/hooks/useResolvedCloudStatuses", () => ({
	useResolvedCloudStatuses: () => new Map(),
}))

vi.mock("@/hooks/useUsageCostVisibility", () => ({
	useUsageCostVisibility: () => mocks.isCostVisible,
}))

vi.mock("@/services/grpc-client", () => ({
	TaskServiceClient: {
		showTaskWithId: vi.fn(async () => ({})),
	},
}))

describe("HistoryPreview", () => {
	beforeEach(() => {
		vi.clearAllMocks()
		mocks.isCostVisible.mockReturnValue(true)
	})

	it("renders backend totalHistoryCost when available", () => {
		mocks.useExtensionState.mockReturnValue({
			taskHistory: [
				{ id: "1", ts: 1000, task: "Task 1", totalCost: 1.25 },
				{ id: "2", ts: 2000, task: "Task 2", totalCost: 2.5 },
			],
			totalHistoryCost: 42.88,
			totalHistoryCount: 99,
		})

		render(<HistoryPreview showHistoryView={vi.fn()} />)

		const badge = screen.getByTestId("history-total-cost")
		expect(badge).toHaveTextContent("Totale: $42.88")
		expect(badge).toHaveAttribute("title", "Totale costi su 99 conversazioni")
	})

	it("falls back to summing visible non-cloud taskHistory costs when totalHistoryCost is undefined", () => {
		mocks.useExtensionState.mockReturnValue({
			taskHistory: [
				{ id: "1", ts: 1000, task: "Task 1", totalCost: 1.25, apiProvider: "openai" },
				{ id: "2", ts: 2000, task: "Task 2", totalCost: 3.1, apiProvider: "openai" },
				{ id: "3", ts: 3000, task: "Cloud Task", totalCost: 9.99, executionTarget: "cloud" },
			],
		})

		render(<HistoryPreview showHistoryView={vi.fn()} />)

		expect(screen.getByTestId("history-total-cost")).toHaveTextContent("Totale: $4.35")
	})
})
