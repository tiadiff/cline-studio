import { describe, expect, it, vi, beforeEach } from "vitest"
import { LocalAiRecoveryCoordinator } from "./local-ai-recovery-coordinator"

describe("LocalAiRecoveryCoordinator", () => {
	let compactTask: any
	let sendAskResponse: any

	beforeEach(() => {
		compactTask = vi.fn().mockResolvedValue(undefined)
		sendAskResponse = vi.fn().mockResolvedValue(undefined)
		// Mock global fetch
		global.fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ ok: true }) } as any) as any
	})

	it("executes recovery: flushes vram, compacts task and sends resume prompt", async () => {
		const coordinator = new LocalAiRecoveryCoordinator({
			compactTask,
			sendAskResponse,
			maxAttempts: 5,
			windowMs: 180000,
		})

		const recovered = await coordinator.handleTurnError()

		expect(recovered).toBe(true)
		expect(global.fetch).toHaveBeenCalledWith("http://127.0.0.1:8080/_localai/free-vram", expect.anything())
		expect(compactTask).toHaveBeenCalledTimes(1)
		expect(sendAskResponse).toHaveBeenCalledWith("compact context and resume task")
	})

	it("enforces max 5 recovery attempts in 3 minutes", async () => {
		const coordinator = new LocalAiRecoveryCoordinator({
			compactTask,
			sendAskResponse,
			maxAttempts: 5,
			windowMs: 180000,
		})

		for (let i = 0; i < 5; i++) {
			const res = await coordinator.handleTurnError()
			expect(res).toBe(true)
		}

		// 6th attempt should be blocked
		const res6 = await coordinator.handleTurnError()
		expect(res6).toBe(false)
		expect(sendAskResponse).toHaveBeenCalledTimes(5)
	})

	it("resets recovery attempts after window expires", async () => {
		let currentTime = 1000
		vi.spyOn(Date, "now").mockImplementation(() => currentTime)

		const coordinator = new LocalAiRecoveryCoordinator({
			compactTask,
			sendAskResponse,
			maxAttempts: 5,
			windowMs: 180000,
		})

		for (let i = 0; i < 5; i++) {
			await coordinator.handleTurnError()
		}

		// Blocked at current time
		expect(await coordinator.handleTurnError()).toBe(false)

		// Advance time by 3 minutes + 1 ms
		currentTime += 180001

		// Should now succeed
		expect(await coordinator.handleTurnError()).toBe(true)
		expect(sendAskResponse).toHaveBeenCalledTimes(6)

		vi.restoreAllMocks()
	})

	it("waits for session idle before compacting and still resumes if compaction fails", async () => {
		const order: string[] = []
		let running = true
		const waitForSessionIdle = vi.fn().mockImplementation(async () => {
			order.push("waitForSessionIdle")
			running = false
		})
		const failingCompact = vi.fn().mockImplementation(async () => {
			order.push(`compact(running=${running})`)
			throw new Error("compaction failed")
		})
		const trackedSend = vi.fn().mockImplementation(async () => {
			order.push("sendAskResponse")
		})

		const coordinator = new LocalAiRecoveryCoordinator({
			compactTask: failingCompact,
			sendAskResponse: trackedSend,
			isSessionRunning: () => running,
			waitForSessionIdle,
			maxAttempts: 5,
			windowMs: 180000,
		})

		const recovered = await coordinator.handleTurnError("custom resume")
		expect(recovered).toBe(true)
		expect(order).toEqual(["waitForSessionIdle", "compact(running=false)", "sendAskResponse"])
		expect(trackedSend).toHaveBeenCalledWith("custom resume")
	})
})
